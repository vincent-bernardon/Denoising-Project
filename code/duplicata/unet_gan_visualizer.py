"""
Visualiseur pour modèles U-Net + GAN
Affiche les résultats de débruitage sur CIFAR-10 ou STL-10
"""
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

from UNET.unet_model import UNet
from utils import add_noise_to_images


def load_unet_model(model_path, device):
    """Charge un modèle U-Net depuis un fichier .pth"""
    model = UNet(n_channels=3, n_classes=3, base_features=64)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def compute_psnr(img1, img2):
    """Calcule le PSNR entre deux images [0, 1]"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def visualize_denoising(model, dataset, device, noise_configs, n_samples=10, 
                       dataset_name='cifar10', save_path=None):
    """
    Visualise le débruitage pour les 3 types de bruit
    
    Args:
        model: Modèle U-Net chargé
        dataset: Dataset (CIFAR-10 ou STL-10)
        device: Device (cuda/cpu)
        noise_configs: Liste des configurations de bruit
        n_samples: Nombre d'images à afficher par type de bruit
        dataset_name: Nom du dataset pour le titre
        save_path: Chemin pour sauvegarder les images (optionnel)
    """
    model.eval()
    
    # Obtenir les labels pour sélectionner une image par classe
    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    else:
        targets = np.array(dataset.labels)
    
    # Classes CIFAR-10 / STL-10
    if dataset_name == 'cifar10':
        class_names = ['plane', 'car', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
    else:
        class_names = ['plane', 'bird', 'car', 'cat', 'deer',
                      'dog', 'horse', 'monkey', 'ship', 'truck']
    
    # Sélectionner une image par classe
    t_idx = {}
    for i in range(min(n_samples, 10)):
        idx_list = np.where(targets == i)[0]
        if len(idx_list) > 0:
            t_idx[i] = idx_list[0]
    
    n_samples = len(t_idx)
    
    # Pour chaque type de bruit
    for noise_idx, noise_config in enumerate(noise_configs):
        fig = plt.figure(figsize=(20, 6))
        fig.suptitle(f"U-Net + GAN Denoising - {noise_config['name']} - {dataset_name.upper()}\n"
                    f"Params: {noise_config['params']}", 
                    fontsize=16, fontweight='bold')
        
        psnr_noisy_list = []
        psnr_denoised_list = []
        
        for idx, (class_id, img_idx) in enumerate(t_idx.items()):
            # Récupérer l'image originale
            img_tensor, label = dataset[img_idx]
            
            # Pour STL-10, redimensionner à 32x32 si nécessaire
            if img_tensor.shape[1] > 32:
                resize = transforms.Resize(32)
                img_tensor = resize(img_tensor)
            
            img_clean = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            
            # Ajouter du bruit
            img_noisy = add_noise_to_images(
                img_clean[np.newaxis, ...],
                noise_type=noise_config['type'],
                **noise_config['params']
            )[0]
            
            # Convertir en tensors
            img_noisy_tensor = torch.FloatTensor(img_noisy).permute(2, 0, 1) / 255.0
            img_clean_tensor = torch.FloatTensor(img_clean).permute(2, 0, 1) / 255.0
            
            # Débruiter
            with torch.no_grad():
                img_noisy_batch = img_noisy_tensor.unsqueeze(0).to(device)
                img_denoised = model(img_noisy_batch).cpu().squeeze()
            
            # Calculer PSNR
            psnr_noisy = compute_psnr(img_noisy_tensor, img_clean_tensor)
            psnr_denoised = compute_psnr(img_denoised, img_clean_tensor)
            psnr_noisy_list.append(psnr_noisy.item())
            psnr_denoised_list.append(psnr_denoised.item())
            
            # Affichage - Ligne 1: Original
            ax = plt.subplot(4, n_samples, idx + 1)
            plt.imshow(img_clean_tensor.permute(1, 2, 0).numpy())
            ax.axis('off')
            if idx == 0:
                ax.set_ylabel('Original', fontsize=12, fontweight='bold')
            ax.set_title(f'{class_names[class_id]}', fontsize=10)
            
            # Ligne 2: Noisy
            ax = plt.subplot(4, n_samples, idx + 1 + n_samples)
            plt.imshow(img_noisy_tensor.permute(1, 2, 0).numpy())
            ax.axis('off')
            if idx == 0:
                ax.set_ylabel('Noisy', fontsize=12, fontweight='bold')
            ax.set_title(f'PSNR: {psnr_noisy:.2f}dB', fontsize=9, color='red')
            
            # Ligne 3: Denoised (GAN)
            ax = plt.subplot(4, n_samples, idx + 1 + 2 * n_samples)
            plt.imshow(img_denoised.permute(1, 2, 0).numpy().clip(0, 1))
            ax.axis('off')
            if idx == 0:
                ax.set_ylabel('Denoised (GAN)', fontsize=12, fontweight='bold')
            improvement = psnr_denoised - psnr_noisy
            color = 'green' if improvement > 0 else 'orange'
            ax.set_title(f'PSNR: {psnr_denoised:.2f}dB (+{improvement:.2f})', 
                        fontsize=9, color=color)
            
            # Ligne 4: Différence (amplifiée)
            ax = plt.subplot(4, n_samples, idx + 1 + 3 * n_samples)
            diff = torch.abs(img_denoised - img_clean_tensor)
            diff_amplified = (diff * 5).clamp(0, 1)  # Amplifier x5 pour mieux voir
            plt.imshow(diff_amplified.permute(1, 2, 0).numpy(), cmap='hot')
            ax.axis('off')
            if idx == 0:
                ax.set_ylabel('Diff (×5)', fontsize=12, fontweight='bold')
            mse = torch.mean(diff ** 2).item()
            ax.set_title(f'MSE: {mse:.4f}', fontsize=9)
        
        # Statistiques globales
        avg_psnr_noisy = np.mean(psnr_noisy_list)
        avg_psnr_denoised = np.mean(psnr_denoised_list)
        avg_improvement = avg_psnr_denoised - avg_psnr_noisy
        
        fig.text(0.5, 0.02, 
                f'Average PSNR - Noisy: {avg_psnr_noisy:.2f}dB | '
                f'Denoised: {avg_psnr_denoised:.2f}dB | '
                f'Improvement: +{avg_improvement:.2f}dB',
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        # Sauvegarder si demandé
        if save_path:
            save_file = f"{save_path}_{noise_config['type']}.png"
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"✓ Sauvegardé: {save_file}")
        
        plt.show()
        
        print(f"\n{noise_config['name']}:")
        print(f"  - PSNR moyen (noisy):    {avg_psnr_noisy:.2f} dB")
        print(f"  - PSNR moyen (denoised): {avg_psnr_denoised:.2f} dB")
        print(f"  - Amélioration:          +{avg_improvement:.2f} dB")


def compare_models(model_paths, dataset, device, noise_config, n_samples=10, 
                   dataset_name='cifar10'):
    """
    Compare plusieurs modèles (U-Net classique vs U-Net+GAN)
    
    Args:
        model_paths: Liste de tuples (nom, path)
        dataset: Dataset
        device: Device
        noise_config: Configuration du bruit
        n_samples: Nombre d'images
        dataset_name: Nom du dataset
    """
    # Charger les modèles
    models = []
    for name, path in model_paths:
        if Path(path).exists():
            model = load_unet_model(path, device)
            models.append((name, model))
            print(f"✓ Modèle chargé: {name}")
        else:
            print(f"/!\\  Modèle non trouvé: {path}")
    
    if len(models) == 0:
        print("X Aucun modèle disponible pour la comparaison")
        return
    
    # Obtenir les labels
    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    else:
        targets = np.array(dataset.labels)
    
    # Sélectionner images
    t_idx = {}
    for i in range(min(n_samples, 10)):
        idx_list = np.where(targets == i)[0]
        if len(idx_list) > 0:
            t_idx[i] = idx_list[0]
    
    n_samples = len(t_idx)
    n_models = len(models)
    
    # Créer la figure
    fig = plt.figure(figsize=(20, 4 * (n_models + 2)))
    fig.suptitle(f"Model Comparison - {noise_config['name']} - {dataset_name.upper()}\n"
                f"Params: {noise_config['params']}", 
                fontsize=16, fontweight='bold')
    
    results = {name: {'psnr': []} for name, _ in models}
    
    for idx, (class_id, img_idx) in enumerate(t_idx.items()):
        # Image originale
        img_tensor, label = dataset[img_idx]
        
        if img_tensor.shape[1] > 32:
            resize = transforms.Resize(32)
            img_tensor = resize(img_tensor)
        
        img_clean = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # Ajouter du bruit
        img_noisy = add_noise_to_images(
            img_clean[np.newaxis, ...],
            noise_type=noise_config['type'],
            **noise_config['params']
        )[0]
        
        img_noisy_tensor = torch.FloatTensor(img_noisy).permute(2, 0, 1) / 255.0
        img_clean_tensor = torch.FloatTensor(img_clean).permute(2, 0, 1) / 255.0
        
        # Original
        ax = plt.subplot(n_models + 2, n_samples, idx + 1)
        plt.imshow(img_clean_tensor.permute(1, 2, 0).numpy())
        ax.axis('off')
        if idx == 0:
            ax.set_ylabel('Original', fontsize=11, fontweight='bold')
        
        # Noisy
        ax = plt.subplot(n_models + 2, n_samples, idx + 1 + n_samples)
        plt.imshow(img_noisy_tensor.permute(1, 2, 0).numpy())
        ax.axis('off')
        if idx == 0:
            ax.set_ylabel('Noisy', fontsize=11, fontweight='bold')
        psnr_noisy = compute_psnr(img_noisy_tensor, img_clean_tensor)
        ax.set_title(f'{psnr_noisy:.2f}dB', fontsize=9, color='red')
        
        # Modèles
        for model_idx, (model_name, model) in enumerate(models):
            with torch.no_grad():
                img_noisy_batch = img_noisy_tensor.unsqueeze(0).to(device)
                img_denoised = model(img_noisy_batch).cpu().squeeze()
            
            psnr_denoised = compute_psnr(img_denoised, img_clean_tensor)
            results[model_name]['psnr'].append(psnr_denoised.item())
            
            ax = plt.subplot(n_models + 2, n_samples, 
                           idx + 1 + (model_idx + 2) * n_samples)
            plt.imshow(img_denoised.permute(1, 2, 0).numpy().clip(0, 1))
            ax.axis('off')
            if idx == 0:
                ax.set_ylabel(model_name, fontsize=11, fontweight='bold')
            improvement = psnr_denoised - psnr_noisy
            color = 'green' if improvement > 0 else 'orange'
            ax.set_title(f'{psnr_denoised:.2f}dB (+{improvement:.2f})', 
                        fontsize=9, color=color)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Afficher les statistiques
    print(f"\n{'='*60}")
    print(f"Comparaison - {noise_config['name']}")
    print(f"{'='*60}")
    for model_name in results:
        avg_psnr = np.mean(results[model_name]['psnr'])
        print(f"{model_name:20s}: {avg_psnr:.2f} dB")
    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualiseur U-Net + GAN')
    parser.add_argument('--model', type=str, required=True,
                        help='Chemin vers le modèle .pth (U-Net ou GAN)')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'stl10'],
                        help='Dataset à utiliser')
    parser.add_argument('--compare', type=str, nargs='+',
                        help='Comparer plusieurs modèles (chemins séparés par espaces)')
    parser.add_argument('--n-samples', type=int, default=10,
                        help='Nombre d\'images à afficher (max 10)')
    parser.add_argument('--save', type=str, default=None,
                        help='Chemin de base pour sauvegarder les images')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("VISUALISEUR U-NET + GAN")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    noise_configs = [
        {'name': 'Gaussien', 'type': 'gaussian', 'params': {'std': 25}},
        {'name': 'Salt & Pepper', 'type': 'salt_pepper', 
         'params': {'salt_prob': 0.02, 'pepper_prob': 0.02}},
        {'name': 'Mixte', 'type': 'mixed', 
         'params': {'gaussian_std': 20, 'salt_prob': 0.01, 'pepper_prob': 0.01}}
    ]
    
    print(f"\nChargement de {args.dataset.upper()}...")
    data_dir = './dataset'
    
    if args.dataset == 'cifar10':
        test_dataset = torchvision.datasets.CIFAR10(
            data_dir, train=False, download=True, transform=transforms.ToTensor()
        )
    else:
        test_dataset = torchvision.datasets.STL10(
            data_dir, split='test', download=True, transform=transforms.ToTensor()
        )
    
    print(f"✓ {len(test_dataset)} images chargées")
    
    # Mode comparaison ou visualisation simple
    if args.compare:
        print("\nMode comparaison activé")
        print(f"Nombre de modèles: {len(args.compare) + 1}")
        
        # Construire la liste des modèles
        model_paths = [('U-Net', args.model)]
        for i, path in enumerate(args.compare):
            name = f'Model {i+2}'
            if 'gan' in path.lower():
                name = f'U-Net+GAN {i+1}'
            model_paths.append((name, path))
        
        # Comparer sur chaque type de bruit
        for noise_config in noise_configs:
            print(f"\n{'='*60}")
            print(f"Test: {noise_config['name']}")
            print(f"{'='*60}")
            compare_models(model_paths, test_dataset, device, noise_config,
                          n_samples=args.n_samples, dataset_name=args.dataset)
    
    else:
        # Visualisation simple
        print(f"\nChargement du modèle: {args.model}")
        
        if not Path(args.model).exists():
            print(f"Erreur: Fichier non trouvé: {args.model}")
            exit(1)
        
        model = load_unet_model(args.model, device)
        print("✓ Modèle chargé avec succès")
        
        # Visualiser pour les 3 types de bruit
        print(f"\n{'='*80}")
        print("VISUALISATION DES RÉSULTATS")
        print(f"{'='*80}")
        
        visualize_denoising(
            model, test_dataset, device, noise_configs,
            n_samples=args.n_samples,
            dataset_name=args.dataset,
            save_path=args.save
        )
    
    print(f"\n{'='*80}")
    print("VISUALISATION TERMINÉE ✓")
    print(f"{'='*80}")
