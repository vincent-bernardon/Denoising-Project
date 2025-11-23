"""
Visualisation comparative U-Net vs U-Net+GAN
Style identique à test_visualization_unet.py mais avec 2 colonnes de débruitage
"""
import torch
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse

from unet_model import UNet
from utils import add_noise_to_images
from patch_utils import denoise_with_patches


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


def visualize_comparison(unet_model, gan_model, dataset, device, noise_config, n_samples=3, dataset_name='stl10'):
    """
    Visualise la comparaison U-Net vs U-Net+GAN (style identique à test_visualization_unet.py)
    
    Args:
        unet_model: Modèle U-Net classique
        gan_model: Modèle U-Net+GAN
        dataset: Dataset (CIFAR-10 ou STL-10)
        device: Device (cuda/cpu)
        noise_config: Configuration du bruit
        n_samples: Nombre d'images à afficher (défaut: 3)
        dataset_name: Nom du dataset
    """
    unet_model.eval()
    gan_model.eval()
    
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
    
    # Créer la figure (même style que test_visualization_unet.py mais avec 4 colonnes)
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]
    
    psnr_noisy_list = []
    psnr_unet_list = []
    psnr_gan_list = []
    mse_noisy_list = []
    mse_unet_list = []
    mse_gan_list = []
    
    for i, img_idx in enumerate(selected_indices):
        # Récupérer l'image
        img_tensor, label = dataset[img_idx]
        
        # Garder la résolution originale
        img_clean = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # Ajouter du bruit
        img_noisy = add_noise_to_images(
            img_clean[np.newaxis, ...],
            noise_type=noise_config['type'],
            **noise_config['params']
        )[0]
        
        # Convertir en tensors
        img_noisy_tensor = torch.FloatTensor(img_noisy).permute(2, 0, 1) / 255.0
        
        # Débruiter avec U-Net classique
        img_unet = denoise_with_patches(unet_model, img_noisy_tensor, device, patch_size=32, stride=16)
        img_unet_np = (img_unet.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        
        # Débruiter avec U-Net+GAN
        img_gan = denoise_with_patches(gan_model, img_noisy_tensor, device, patch_size=32, stride=16)
        img_gan_np = (img_gan.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        
        # Calculer métriques
        mse_noisy = np.mean((img_noisy.astype(float) - img_clean.astype(float)) ** 2)
        psnr_noisy = calculate_psnr(img_noisy, img_clean, max_pixel_value=255.0)
        
        mse_unet = np.mean((img_unet_np.astype(float) - img_clean.astype(float)) ** 2)
        psnr_unet = calculate_psnr(img_unet_np, img_clean, max_pixel_value=255.0)
        improvement_unet = psnr_unet - psnr_noisy
        
        mse_gan = np.mean((img_gan_np.astype(float) - img_clean.astype(float)) ** 2)
        psnr_gan = calculate_psnr(img_gan_np, img_clean, max_pixel_value=255.0)
        improvement_gan = psnr_gan - psnr_noisy
        
        psnr_noisy_list.append(psnr_noisy)
        psnr_unet_list.append(psnr_unet)
        psnr_gan_list.append(psnr_gan)
        mse_noisy_list.append(mse_noisy)
        mse_unet_list.append(mse_unet)
        mse_gan_list.append(mse_gan)
        
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
        
        # Colonne 2: U-Net classique
        axes[i, 1].imshow(img_unet_np.astype(np.uint8))
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].set_title('Image débruitée (U-Net)', fontsize=12, fontweight='bold')
        
        # Métriques sous l'image débruitée avec amélioration
        color_unet = 'green' if improvement_unet > 0 else 'red'
        metrics_text = f'MSE: {mse_unet:.1f}\nPSNR: {psnr_unet:.2f} dB'
        if improvement_unet > 0:
            metrics_text += f'\n(+{improvement_unet:.2f} dB)'
        else:
            metrics_text += f'\n({improvement_unet:.2f} dB)'
        axes[i, 1].text(0.5, -0.05, metrics_text, ha='center', va='top', 
                       transform=axes[i, 1].transAxes, fontsize=9, color=color_unet,
                       bbox=dict(boxstyle='round', facecolor='lightgreen' if improvement_unet > 0 else 'lightcoral', alpha=0.5))
        
        # Colonne 3: U-Net+GAN
        axes[i, 2].imshow(img_gan_np.astype(np.uint8))
        axes[i, 2].axis('off')
        if i == 0:
            axes[i, 2].set_title('Image débruitée (U-Net+GAN)', fontsize=12, fontweight='bold')
        
        # Métriques sous l'image débruitée avec amélioration
        color_gan = 'green' if improvement_gan > 0 else 'red'
        metrics_text = f'MSE: {mse_gan:.1f}\nPSNR: {psnr_gan:.2f} dB'
        if improvement_gan > 0:
            metrics_text += f'\n(+{improvement_gan:.2f} dB)'
        else:
            metrics_text += f'\n({improvement_gan:.2f} dB)'
        
        # Comparaison U-Net+GAN vs U-Net
        diff_psnr = psnr_gan - psnr_unet
        if abs(diff_psnr) > 0.01:
            if diff_psnr > 0:
                metrics_text += f'\n[+{diff_psnr:.2f} dB vs U-Net]'
            else:
                metrics_text += f'\n[{diff_psnr:.2f} dB vs U-Net]'
        
        axes[i, 2].text(0.5, -0.05, metrics_text, ha='center', va='top', 
                       transform=axes[i, 2].transAxes, fontsize=9, color=color_gan,
                       bbox=dict(boxstyle='round', facecolor='lightgreen' if improvement_gan > 0 else 'lightcoral', alpha=0.5))
        
        # Colonne 4: Image propre (ground truth)
        axes[i, 3].imshow(img_clean.astype(np.uint8))
        axes[i, 3].axis('off')
        if i == 0:
            axes[i, 3].set_title('Image propre (GT)', fontsize=12, fontweight='bold')
        
        # "Référence" sous l'image propre
        axes[i, 3].text(0.5, -0.05, 'Référence', ha='center', va='top', 
                       transform=axes[i, 3].transAxes, fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    # Statistiques
    avg_psnr_noisy = np.mean(psnr_noisy_list)
    avg_psnr_unet = np.mean(psnr_unet_list)
    avg_psnr_gan = np.mean(psnr_gan_list)
    avg_mse_noisy = np.mean(mse_noisy_list)
    avg_mse_unet = np.mean(mse_unet_list)
    avg_mse_gan = np.mean(mse_gan_list)
    
    avg_gain_unet = avg_psnr_unet - avg_psnr_noisy
    avg_gain_gan = avg_psnr_gan - avg_psnr_noisy
    gan_vs_unet = avg_psnr_gan - avg_psnr_unet
    
    print(f"\n{noise_config['name']} - Résultats:")
    print(f"  MSE moyen (noisy):      {avg_mse_noisy:.2f}")
    print(f"  MSE moyen (U-Net):      {avg_mse_unet:.2f}")
    print(f"  MSE moyen (U-Net+GAN):  {avg_mse_gan:.2f}")
    print(f"  PSNR moyen (noisy):     {avg_psnr_noisy:.2f} dB")
    print(f"  PSNR moyen (U-Net):     {avg_psnr_unet:.2f} dB")
    print(f"  PSNR moyen (U-Net+GAN): {avg_psnr_gan:.2f} dB")
    print(f"  Gain U-Net:             +{avg_gain_unet:.2f} dB")
    print(f"  Gain U-Net+GAN:         +{avg_gain_gan:.2f} dB")
    if abs(gan_vs_unet) > 0.01:
        symbol = '+' if gan_vs_unet > 0 else ''
        print(f"  U-Net+GAN vs U-Net:     {symbol}{gan_vs_unet:.2f} dB")


def test_visualization_unet_gan():
    """Teste la visualisation comparative U-Net vs U-Net+GAN"""
    
    parser = argparse.ArgumentParser(description='Visualisation U-Net vs U-Net+GAN')
    parser.add_argument('--unet', type=str, required=True,
                        help='Chemin vers le modèle U-Net classique (.pth)')
    parser.add_argument('--gan', type=str, required=True,
                        help='Chemin vers le modèle U-Net+GAN (.pth)')
    parser.add_argument('--dataset', type=str, default='stl10',
                        choices=['cifar10', 'stl10'],
                        help='Dataset à utiliser (défaut: stl10)')
    parser.add_argument('--n-samples', type=int, default=3,
                        help='Nombre d\'images à afficher (défaut: 3)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("VISUALISATION COMPARATIVE U-NET vs U-NET+GAN")
    print("=" * 80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Vérifier la disponibilité des modèles GAN pour le dataset choisi
    if args.dataset == 'cifar10':
        print("\n⚠️  ATTENTION: Aucun modèle U-Net+GAN disponible pour CIFAR-10")
        print("    Pour entraîner un modèle U-Net+GAN sur CIFAR-10:")
        print("    1. Entraîner d'abord un U-Net classique:")
        print("       python code/unet_train_multi_noise.py --dataset cifar10 --epochs 100")
        print("    2. Puis entraîner le GAN:")
        print("       python code/unet_gan_train.py --dataset cifar10 --pretrained-unet ./code/unet_denoising_cifar10_multinoise.pth --epochs 50")
        print("\n    Actuellement, seul STL-10 dispose d'un modèle U-Net+GAN entraîné.")
        print("    Utilisez --dataset stl10 pour tester la visualisation comparative.")
        return
    
    # Charger les modèles
    print(f"\nChargement des modèles...")
    unet_model = load_unet_model(args.unet, device)
    gan_model = load_unet_model(args.gan, device)
    
    # Charger le dataset STL-10
    print(f"\nChargement de {args.dataset.upper()}...")
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
        
        visualize_comparison(
            unet_model=unet_model,
            gan_model=gan_model,
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
    test_visualization_unet_gan()
