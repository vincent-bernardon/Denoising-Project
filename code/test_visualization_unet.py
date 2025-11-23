"""
Visualisation des résultats de débruitage U-Net
Pour CIFAR-10 ou STL-10
"""
import torch
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse

from unet_model import UNet
from utils import add_noise_to_images


def load_unet_model(model_path, device):
    """Charge un modèle U-Net"""
    model = UNet(n_channels=3, n_classes=3, base_features=64)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ Modèle chargé: {model_path}")
    return model


def calculate_psnr(img1, img2, max_pixel_value=255.0):
    """Calcule le PSNR entre deux images"""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_pixel_value / np.sqrt(mse))


def denoise_with_patches(model, img_tensor, device, patch_size=32, stride=16):
    """
    Débruite une image en la découpant en patches, puis recompose l'image
    Utilisé pour STL-10 (96x96) avec des modèles entraînés sur patches 32x32
    """
    _, h, w = img_tensor.shape
    
    # Si l'image est déjà 32x32, pas besoin de patches
    if h == 32 and w == 32:
        with torch.no_grad():
            img_batch = img_tensor.unsqueeze(0).to(device)
            denoised = model(img_batch).cpu().squeeze()
        return denoised
    
    # Créer une image de sortie et un compteur pour la moyenne pondérée
    denoised_img = torch.zeros_like(img_tensor)
    weight_map = torch.zeros((h, w))
    
    # Extraire et débruiter chaque patch
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            # Extraire le patch
            patch = img_tensor[:, i:i+patch_size, j:j+patch_size]
            
            # Débruiter
            with torch.no_grad():
                patch_batch = patch.unsqueeze(0).to(device)
                denoised_patch = model(patch_batch).cpu().squeeze()
            
            # Ajouter au résultat avec pondération
            denoised_img[:, i:i+patch_size, j:j+patch_size] += denoised_patch
            weight_map[i:i+patch_size, j:j+patch_size] += 1
    
    # Normaliser par le nombre de superpositions
    weight_map = weight_map.unsqueeze(0)  # Ajouter dimension de canal
    denoised_img = denoised_img / weight_map
    
    return denoised_img


def visualize_denoising(model, dataset, device, noise_config, n_samples=3, dataset_name='stl10'):
    """
    Visualise le débruitage pour un type de bruit (style identique à test_visualization.py)
    
    Args:
        model: Modèle U-Net chargé
        dataset: Dataset (CIFAR-10 ou STL-10)
        device: Device (cuda/cpu)
        noise_config: Configuration du bruit
        n_samples: Nombre d'images à afficher (défaut: 3)
        dataset_name: Nom du dataset
    """
    model.eval()
    
    # Classes
    if dataset_name == 'cifar10':
        class_names = ['plane', 'car', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
    else:  # stl10
        class_names = ['plane', 'bird', 'car', 'cat', 'deer',
                      'dog', 'horse', 'monkey', 'ship', 'truck']
    
    # Obtenir les labels
    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    else:
        targets = np.array(dataset.labels)
    
    # Sélectionner n_samples images aléatoires
    selected_indices = np.random.choice(len(dataset), size=n_samples, replace=False)
    
    # Dictionnaire pour les noms d'affichage des types de bruit
    noise_display_names = {
        'gaussian': 'Gaussien',
        'salt_pepper': 'Salt & Pepper',
        'mixed': 'Mixte'
    }
    noise_label = noise_display_names.get(noise_config['type'], noise_config['type'].capitalize())
    
    # Créer la figure (même style que image_visualizer.py)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]
    
    psnr_noisy_list = []
    psnr_denoised_list = []
    mse_noisy_list = []
    mse_denoised_list = []
    
    for i, img_idx in enumerate(selected_indices):
        # Récupérer l'image
        img_tensor, label = dataset[img_idx]
        
        # Garder la résolution originale (96x96 pour STL-10)
        img_clean = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # Ajouter du bruit
        img_noisy = add_noise_to_images(
            img_clean[np.newaxis, ...],
            noise_type=noise_config['type'],
            **noise_config['params']
        )[0]
        
        # Convertir en tensors
        img_noisy_tensor = torch.FloatTensor(img_noisy).permute(2, 0, 1) / 255.0
        
        # Débruiter avec système de patches si nécessaire
        img_denoised = denoise_with_patches(model, img_noisy_tensor, device, patch_size=32, stride=16)
        img_denoised_np = (img_denoised.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        
        # Calculer métriques
        mse_noisy = np.mean((img_noisy.astype(float) - img_clean.astype(float)) ** 2)
        psnr_noisy = calculate_psnr(img_noisy, img_clean, max_pixel_value=255.0)
        mse_denoised = np.mean((img_denoised_np.astype(float) - img_clean.astype(float)) ** 2)
        psnr_denoised = calculate_psnr(img_denoised_np, img_clean, max_pixel_value=255.0)
        improvement = psnr_denoised - psnr_noisy
        
        psnr_noisy_list.append(psnr_noisy)
        psnr_denoised_list.append(psnr_denoised)
        mse_noisy_list.append(mse_noisy)
        mse_denoised_list.append(mse_denoised)
        
        # Colonne 1: Image bruitée
        axes[i, 0].imshow(img_noisy.astype(np.uint8))
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title(f'Image bruitée ({noise_label})', fontsize=12, fontweight='bold')
        
        # Métriques sous l'image bruitée
        metrics_text = f'MSE: {mse_noisy:.1f}\nPSNR: {psnr_noisy:.2f} dB'
        axes[i, 0].text(0.5, -0.05, metrics_text, ha='center', va='top', 
                       transform=axes[i, 0].transAxes, fontsize=9, 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Colonne 2: Image débruitée (reconstruite)
        axes[i, 1].imshow(img_denoised_np.astype(np.uint8))
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].set_title('Image débruitée (U-Net)', fontsize=12, fontweight='bold')
        
        # Métriques sous l'image débruitée avec amélioration
        color = 'green' if improvement > 0 else 'red'
        metrics_text = f'MSE: {mse_denoised:.1f}\nPSNR: {psnr_denoised:.2f} dB'
        if improvement > 0:
            metrics_text += f'\n(+{improvement:.2f} dB)'
        else:
            metrics_text += f'\n({improvement:.2f} dB)'
        axes[i, 1].text(0.5, -0.05, metrics_text, ha='center', va='top', 
                       transform=axes[i, 1].transAxes, fontsize=9, color=color,
                       bbox=dict(boxstyle='round', facecolor='lightgreen' if improvement > 0 else 'lightcoral', alpha=0.5))
        
        # Colonne 3: Image propre (ground truth)
        axes[i, 2].imshow(img_clean.astype(np.uint8))
        axes[i, 2].axis('off')
        if i == 0:
            axes[i, 2].set_title('Image propre (GT)', fontsize=12, fontweight='bold')
        
        # "Référence" sous l'image propre
        axes[i, 2].text(0.5, -0.05, 'Référence', ha='center', va='top', 
                       transform=axes[i, 2].transAxes, fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    # Statistiques
    avg_psnr_noisy = np.mean(psnr_noisy_list)
    avg_psnr_denoised = np.mean(psnr_denoised_list)
    avg_mse_noisy = np.mean(mse_noisy_list)
    avg_mse_denoised = np.mean(mse_denoised_list)
    avg_gain = avg_psnr_denoised - avg_psnr_noisy
    
    print(f"\n{noise_config['name']} - Résultats:")
    print(f"  MSE moyen (noisy):      {avg_mse_noisy:.2f}")
    print(f"  MSE moyen (denoised):   {avg_mse_denoised:.2f}")
    print(f"  PSNR moyen (noisy):     {avg_psnr_noisy:.2f} dB")
    print(f"  PSNR moyen (denoised):  {avg_psnr_denoised:.2f} dB")
    print(f"  Gain moyen:             +{avg_gain:.2f} dB")


def test_visualization_unet():
    """Teste la visualisation avec le modèle U-Net sauvegardé"""
    
    parser = argparse.ArgumentParser(description='Visualisation U-Net')
    parser.add_argument('--model', type=str, required=True,
                        help='Chemin vers le modèle U-Net (.pth)')
    parser.add_argument('--dataset', type=str, default='stl10',
                        choices=['cifar10', 'stl10'],
                        help='Dataset à utiliser (défaut: stl10)')
    parser.add_argument('--n-samples', type=int, default=3,
                        help='Nombre d\'images à afficher (défaut: 3)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TEST DE VISUALISATION U-NET")
    print("=" * 80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Charger le modèle
    print(f"\nChargement du modèle U-Net...")
    model = load_unet_model(args.model, device)
    
    # Charger le dataset
    print(f"\nChargement de {args.dataset.upper()}...")
    
    if args.dataset == 'cifar10':
        data_dir = './code/code'
        test_dataset = torchvision.datasets.CIFAR10(
            data_dir, train=False, download=False, transform=transforms.ToTensor()
        )
    else:  # stl10
        data_dir = './code/code/stl10'
        test_dataset = torchvision.datasets.STL10(
            data_dir, split='test', download=False, transform=transforms.ToTensor()
        )
    
    print(f"✅ {len(test_dataset)} images chargées")
    
    # Configuration des bruits
    noise_configs = [
        {
            'name': 'Gaussien',
            'type': 'gaussian',
            'params': {'std': 25}
        },
        {
            'name': 'Salt & Pepper',
            'type': 'salt_pepper',
            'params': {'salt_prob': 0.02, 'pepper_prob': 0.02}
        },
        {
            'name': 'Mixte',
            'type': 'mixed',
            'params': {'gaussian_std': 20, 'salt_prob': 0.01, 'pepper_prob': 0.01}
        }
    ]
    
    # Tester les 3 types de bruit
    for i, noise_config in enumerate(noise_configs, 1):
        print("\n" + "=" * 80)
        print(f"TEST {i}/3 : {noise_config['name']}")
        print("=" * 80)
        
        visualize_denoising(
            model=model,
            dataset=test_dataset,
            device=device,
            noise_config=noise_config,
            n_samples=args.n_samples,
            dataset_name=args.dataset
        )
    
    print("\n" + "=" * 80)
    print("TEST TERMINÉ ✔")
    print("=" * 80)


if __name__ == "__main__":
    test_visualization_unet()
