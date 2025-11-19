"""
Comparaison directe U-Net vs U-Net+GAN sur les mêmes images
Visualise côte à côte les résultats des deux modèles
"""
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

from unet_model import UNet
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


def compare_unet_vs_gan(unet_path, gan_path, dataset, device, noise_configs, n_samples=10):
    """
    Compare U-Net classique vs U-Net+GAN côte à côte
    
    Args:
        unet_path: Chemin vers modèle U-Net classique
        gan_path: Chemin vers modèle U-Net+GAN
        dataset: Dataset de test
        device: Device (cuda/cpu)
        noise_configs: Liste des configurations de bruit
        n_samples: Nombre d'images à comparer
    """
    # Charger les deux modèles
    print("\n📦 Chargement des modèles...")
    print(f"  - U-Net classique: {unet_path}")
    unet_model = load_unet_model(unet_path, device)
    print("    ✓ Chargé")
    
    print(f"  - U-Net + GAN: {gan_path}")
    gan_model = load_unet_model(gan_path, device)
    print("    ✓ Chargé")
    
    # Obtenir les labels
    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    else:
        targets = np.array(dataset.labels)
    
    # Noms des classes
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
    for noise_config in noise_configs:
        print(f"\n{'='*80}")
        print(f"Comparaison: {noise_config['name']}")
        print(f"{'='*80}")
        
        # Créer une figure avec 5 lignes
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(f"U-Net vs U-Net+GAN - {noise_config['name']}\n"
                    f"Params: {noise_config['params']}", 
                    fontsize=16, fontweight='bold')
        
        psnr_noisy_list = []
        psnr_unet_list = []
        psnr_gan_list = []
        improvement_list = []
        
        for idx, (class_id, img_idx) in enumerate(t_idx.items()):
            # Récupérer l'image originale
            img_tensor, label = dataset[img_idx]
            
            # Redimensionner à 32x32 si nécessaire (pour STL-10)
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
            
            # Débruiter avec U-Net classique
            with torch.no_grad():
                img_noisy_batch = img_noisy_tensor.unsqueeze(0).to(device)
                img_unet = unet_model(img_noisy_batch).cpu().squeeze()
            
            # Débruiter avec U-Net+GAN
            with torch.no_grad():
                img_gan = gan_model(img_noisy_batch).cpu().squeeze()
            
            # Calculer PSNR
            psnr_noisy = compute_psnr(img_noisy_tensor, img_clean_tensor)
            psnr_unet = compute_psnr(img_unet, img_clean_tensor)
            psnr_gan = compute_psnr(img_gan, img_clean_tensor)
            improvement = psnr_gan - psnr_unet
            
            psnr_noisy_list.append(psnr_noisy.item())
            psnr_unet_list.append(psnr_unet.item())
            psnr_gan_list.append(psnr_gan.item())
            improvement_list.append(improvement.item())
            
            # Ligne 1: Image originale
            ax = plt.subplot(5, n_samples, idx + 1)
            plt.imshow(img_clean_tensor.permute(1, 2, 0).numpy())
            ax.axis('off')
            if idx == 0:
                ax.set_ylabel('Original', fontsize=13, fontweight='bold', rotation=0, 
                            ha='right', va='center')
            ax.set_title(f'{class_names[class_id]}', fontsize=11, fontweight='bold')
            
            # Ligne 2: Image bruitée
            ax = plt.subplot(5, n_samples, idx + 1 + n_samples)
            plt.imshow(img_noisy_tensor.permute(1, 2, 0).numpy())
            ax.axis('off')
            if idx == 0:
                ax.set_ylabel('Noisy', fontsize=13, fontweight='bold', rotation=0,
                            ha='right', va='center')
            ax.set_title(f'PSNR: {psnr_noisy:.2f}dB', fontsize=10, color='red')
            
            # Ligne 3: U-Net seul
            ax = plt.subplot(5, n_samples, idx + 1 + 2 * n_samples)
            plt.imshow(img_unet.permute(1, 2, 0).numpy().clip(0, 1))
            ax.axis('off')
            if idx == 0:
                ax.set_ylabel('U-Net Only', fontsize=13, fontweight='bold', 
                            rotation=0, ha='right', va='center')
            gain_unet = psnr_unet - psnr_noisy
            ax.set_title(f'PSNR: {psnr_unet:.2f}dB (+{gain_unet:.2f})', 
                        fontsize=10, color='blue')
            
            # Ligne 4: U-Net + GAN
            ax = plt.subplot(5, n_samples, idx + 1 + 3 * n_samples)
            plt.imshow(img_gan.permute(1, 2, 0).numpy().clip(0, 1))
            ax.axis('off')
            if idx == 0:
                ax.set_ylabel('U-Net + GAN', fontsize=13, fontweight='bold',
                            rotation=0, ha='right', va='center')
            gain_gan = psnr_gan - psnr_noisy
            color = 'green' if psnr_gan > psnr_unet else 'orange'
            ax.set_title(f'PSNR: {psnr_gan:.2f}dB (+{gain_gan:.2f})', 
                        fontsize=10, color=color)
            
            # Ligne 5: Différence U-Net vs GAN (amplifiée)
            ax = plt.subplot(5, n_samples, idx + 1 + 4 * n_samples)
            diff = torch.abs(img_gan - img_unet)
            diff_amplified = (diff * 10).clamp(0, 1)  # Amplifier x10
            plt.imshow(diff_amplified.permute(1, 2, 0).numpy(), cmap='hot')
            ax.axis('off')
            if idx == 0:
                ax.set_ylabel('Diff (×10)', fontsize=13, fontweight='bold',
                            rotation=0, ha='right', va='center')
            mse_diff = torch.mean(diff ** 2).item()
            color_diff = 'green' if improvement > 0 else 'red'
            ax.set_title(f'Δ: {improvement:.2f}dB', fontsize=10, 
                        color=color_diff, fontweight='bold')
        
        # Statistiques globales
        avg_psnr_noisy = np.mean(psnr_noisy_list)
        avg_psnr_unet = np.mean(psnr_unet_list)
        avg_psnr_gan = np.mean(psnr_gan_list)
        avg_improvement = np.mean(improvement_list)
        
        stats_text = (f'Average PSNR - Noisy: {avg_psnr_noisy:.2f}dB | '
                     f'U-Net: {avg_psnr_unet:.2f}dB | '
                     f'U-Net+GAN: {avg_psnr_gan:.2f}dB | '
                     f'GAN Improvement: {avg_improvement:+.2f}dB')
        
        color_bg = 'lightgreen' if avg_improvement > 0 else 'wheat'
        
        fig.text(0.5, 0.02, stats_text,
                ha='center', fontsize=13, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=color_bg, alpha=0.7))
        
        plt.tight_layout(rect=[0, 0.04, 1, 0.96])
        plt.show()
        
        # Afficher les résultats détaillés
        print(f"\n{noise_config['name']} - Résultats:")
        print(f"  {'Image':<12} {'Noisy':<10} {'U-Net':<10} {'GAN':<10} {'Δ (GAN-UNet)':<15}")
        print(f"  {'-'*60}")
        for i, (pn, pu, pg, imp) in enumerate(zip(psnr_noisy_list, psnr_unet_list, 
                                                    psnr_gan_list, improvement_list)):
            symbol = '✓' if imp > 0 else '✗'
            print(f"  {class_names[i]:<12} {pn:>6.2f} dB  {pu:>6.2f} dB  "
                  f"{pg:>6.2f} dB  {imp:>+6.2f} dB {symbol}")
        print(f"  {'-'*60}")
        print(f"  {'MOYENNE':<12} {avg_psnr_noisy:>6.2f} dB  {avg_psnr_unet:>6.2f} dB  "
              f"{avg_psnr_gan:>6.2f} dB  {avg_improvement:>+6.2f} dB")
        
        if avg_improvement > 0:
            print(f"\n  ✅ Le GAN améliore le débruitage de {avg_improvement:.2f} dB en moyenne !")
        else:
            print(f"\n  ⚠️  Le GAN n'améliore pas significativement le débruitage")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Comparaison U-Net vs U-Net+GAN sur les mêmes images'
    )
    parser.add_argument('--unet', type=str, required=True,
                        help='Chemin vers le modèle U-Net classique (.pth)')
    parser.add_argument('--gan', type=str, required=True,
                        help='Chemin vers le modèle U-Net+GAN (.pth)')
    parser.add_argument('--dataset', type=str, default='stl10',
                        choices=['cifar10', 'stl10'],
                        help='Dataset à utiliser (défaut: stl10)')
    parser.add_argument('--n-samples', type=int, default=10,
                        help='Nombre d\'images à comparer (max 10, défaut: 10)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("COMPARAISON U-NET vs U-NET+GAN")
    print("=" * 80)
    
    # Vérifier que les fichiers existent
    if not Path(args.unet).exists():
        print(f"❌ Erreur: Modèle U-Net non trouvé: {args.unet}")
        exit(1)
    
    if not Path(args.gan).exists():
        print(f"❌ Erreur: Modèle GAN non trouvé: {args.gan}")
        exit(1)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Device: {device}")
    
    # Configuration des bruits
    noise_configs = [
        {'name': 'Gaussien', 'type': 'gaussian', 'params': {'std': 25}},
        {'name': 'Salt & Pepper', 'type': 'salt_pepper', 
         'params': {'salt_prob': 0.02, 'pepper_prob': 0.02}},
        {'name': 'Mixte', 'type': 'mixed', 
         'params': {'gaussian_std': 20, 'salt_prob': 0.01, 'pepper_prob': 0.01}}
    ]
    
    # Charger le dataset
    print(f"\n📦 Chargement de {args.dataset.upper()}...")
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
    
    # Comparer les modèles
    compare_unet_vs_gan(
        args.unet, args.gan, test_dataset, device, 
        noise_configs, n_samples=args.n_samples
    )
    
    print(f"\n{'='*80}")
    print("COMPARAISON TERMINÉE ✓")
    print(f"{'='*80}")
