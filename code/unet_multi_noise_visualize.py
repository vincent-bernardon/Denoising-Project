"""
Script pour évaluer les performances du modèle U-Net sur STL-10
Reconstruit les images complètes 96x96 à partir des patches 32x32 débruités
Calcule le PSNR moyen et MSE moyen pour différents types de bruit
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from unet_model import UNet
from utils import add_noise_to_images


def reconstruct_image_from_patches(patches, image_size=96, patch_size=32, stride=16):
    """
    Reconstruit une image complète à partir de patches avec superposition
    Utilise une moyenne pondérée pour les zones de superposition
    
    Args:
        patches: Liste de patches [n_patches, C, patch_size, patch_size]
        image_size: Taille de l'image finale (96 pour STL-10)
        patch_size: Taille de chaque patch (32)
        stride: Stride utilisé lors du découpage (16)
    
    Returns:
        Image reconstruite [C, image_size, image_size]
    """
    n_patches_per_dim = (image_size - patch_size) // stride + 1
    n_channels = patches[0].shape[0]
    
    # Image accumulée et compteur pour la moyenne pondérée
    reconstructed = np.zeros((n_channels, image_size, image_size), dtype=np.float32)
    weight_map = np.zeros((image_size, image_size), dtype=np.float32)
    
    patch_idx = 0
    for i in range(n_patches_per_dim):
        for j in range(n_patches_per_dim):
            top = i * stride
            left = j * stride
            
            # Ajouter le patch à l'image
            reconstructed[:, top:top+patch_size, left:left+patch_size] += patches[patch_idx]
            weight_map[top:top+patch_size, left:left+patch_size] += 1.0
            
            patch_idx += 1
    
    # Normaliser par le nombre de superpositions
    reconstructed /= weight_map[np.newaxis, :, :]
    
    return reconstructed


def denoise_full_image(model, img_clean, noise_type, noise_params, device, patch_size=32, stride=16):
    """
    Débruite une image complète en la découpant en patches, puis reconstruit
    
    Args:
        model: Modèle U-Net
        img_clean: Image propre [C, H, W] en [0,1]
        noise_type: Type de bruit
        noise_params: Paramètres du bruit
        device: Device PyTorch
        patch_size: Taille des patches (32)
        stride: Stride pour le découpage (16)
    
    Returns:
        img_noisy, img_denoised: Images bruitée et débruitée [C, H, W] en [0,1]
    """
    _, h, w = img_clean.shape
    
    # Convertir en uint8 pour ajouter le bruit
    img_clean_uint8 = (img_clean.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    # Ajouter le bruit
    img_noisy_uint8 = add_noise_to_images(
        img_clean_uint8[np.newaxis, ...],
        noise_type=noise_type,
        **noise_params
    )[0]
    
    # Convertir en tensor [0,1]
    img_noisy = torch.FloatTensor(img_noisy_uint8).permute(2, 0, 1) / 255.0
    
    # Découper en patches
    n_patches_h = (h - patch_size) // stride + 1
    n_patches_w = (w - patch_size) // stride + 1
    
    patches_noisy = []
    patches_denoised = []
    
    model.eval()
    with torch.no_grad():
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                top = i * stride
                left = j * stride
                
                # Extraire le patch
                patch_noisy = img_noisy[:, top:top+patch_size, left:left+patch_size]
                patches_noisy.append(patch_noisy.cpu().numpy())
                
                # Débruiter le patch
                patch_noisy_batch = patch_noisy.unsqueeze(0).to(device)
                patch_denoised = model(patch_noisy_batch)
                patch_denoised = torch.clamp(patch_denoised, 0., 1.)
                patches_denoised.append(patch_denoised.squeeze(0).cpu().numpy())
    
    # Reconstruire l'image complète
    img_denoised = reconstruct_image_from_patches(patches_denoised, h, patch_size, stride)
    img_denoised = torch.FloatTensor(img_denoised)
    
    return img_noisy, img_denoised


def calculate_psnr(img1, img2, max_pixel_value=255.0):
    """Calcule le PSNR entre deux images (format uint8)"""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr


def calculate_mse(img1, img2):
    """Calcule le MSE entre deux images"""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    return mse


def load_unet_model(model_path, device):
    """Charge le modèle U-Net sauvegardé"""
    model = UNet(n_channels=3, n_classes=3, base_features=64)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def calculate_metrics_for_noise_type(model, test_dataset, noise_type, noise_params, device, 
                                     n_samples=100, patch_size=32, stride=16):
    """
    Calcule les métriques moyennes pour un type de bruit donné sur images complètes
    
    Args:
        model: Modèle U-Net
        test_dataset: Dataset STL-10 test (images 96x96)
        noise_type: Type de bruit
        noise_params: Paramètres du bruit
        device: Device PyTorch
        n_samples: Nombre d'images à évaluer
        patch_size: Taille des patches (32)
        stride: Stride pour le découpage (16)
    
    Returns:
        dict: Métriques calculées
    """
    # Limiter le nombre d'échantillons
    n_samples = min(n_samples, len(test_dataset))
    indices = np.random.choice(len(test_dataset), n_samples, replace=False)
    
    psnr_noisy_list = []
    psnr_denoised_list = []
    mse_noisy_list = []
    mse_denoised_list = []
    
    print(f"  Traitement de {n_samples} images...")
    
    for idx in indices:
        # Récupérer l'image propre [C, H, W] en [0,1]
        img_clean, _ = test_dataset[idx]
        
        # Débruiter l'image complète (découpe + reconstruction)
        img_noisy, img_denoised = denoise_full_image(
            model, img_clean, noise_type, noise_params, device, patch_size, stride
        )
        
        # Convertir en uint8 [0, 255] pour les métriques
        img_clean_uint8 = (img_clean.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_noisy_uint8 = (img_noisy.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_denoised_uint8 = (img_denoised.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # Calculer les métriques
        psnr_noisy = calculate_psnr(img_noisy_uint8, img_clean_uint8, max_pixel_value=255.0)
        mse_noisy = calculate_mse(img_noisy_uint8, img_clean_uint8)
        
        psnr_denoised = calculate_psnr(img_denoised_uint8, img_clean_uint8, max_pixel_value=255.0)
        mse_denoised = calculate_mse(img_denoised_uint8, img_clean_uint8)
        
        psnr_noisy_list.append(psnr_noisy)
        psnr_denoised_list.append(psnr_denoised)
        mse_noisy_list.append(mse_noisy)
        mse_denoised_list.append(mse_denoised)
    
    # Calculer les moyennes
    avg_psnr_noisy = np.mean(psnr_noisy_list)
    avg_psnr_denoised = np.mean(psnr_denoised_list)
    avg_mse_noisy = np.mean(mse_noisy_list)
    avg_mse_denoised = np.mean(mse_denoised_list)
    
    # Calculer les gains
    psnr_gain = avg_psnr_denoised - avg_psnr_noisy
    mse_reduction = avg_mse_noisy - avg_mse_denoised
    mse_reduction_percent = (mse_reduction / avg_mse_noisy) * 100
    
    return {
        'avg_psnr_noisy': avg_psnr_noisy,
        'avg_psnr_denoised': avg_psnr_denoised,
        'psnr_gain': psnr_gain,
        'avg_mse_noisy': avg_mse_noisy,
        'avg_mse_denoised': avg_mse_denoised,
        'mse_reduction': mse_reduction,
        'mse_reduction_percent': mse_reduction_percent,
        'n_samples': n_samples
    }


def plot_visual_results(model, test_dataset, noise_configs, device, n_samples=10, 
                       patch_size=32, stride=16):
    """
    Affiche des exemples visuels de débruitage sur images complètes
    VERSION AMÉLIORÉE : Plus d'images (10) et plus grandes
    Crée un graphique séparé pour chaque type de bruit
    """
    indices = np.random.choice(len(test_dataset), n_samples, replace=False)
    
    for noise_idx, noise_config in enumerate(noise_configs):
        # Créer une grande figure pour ce type de bruit
        fig = plt.figure(figsize=(24, 7.2))  # 3 lignes × 10 colonnes, images plus grandes
        fig.suptitle(f"U-Net STL-10 - {noise_config['name']} (Params: {noise_config['params']})", 
                     fontsize=16, fontweight='bold', y=0.98)
        
        for i, idx in enumerate(indices):
            # Récupérer l'image propre
            img_clean, _ = test_dataset[idx]
            
            # Débruiter
            img_noisy, img_denoised = denoise_full_image(
                model, img_clean, noise_config['type'], noise_config['params'],
                device, patch_size, stride
            )
            
            # Calculer les métriques pour cette image
            img_clean_uint8 = (img_clean.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_noisy_uint8 = (img_noisy.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_denoised_uint8 = (img_denoised.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            
            psnr_noisy = calculate_psnr(img_noisy_uint8, img_clean_uint8)
            psnr_denoised = calculate_psnr(img_denoised_uint8, img_clean_uint8)
            psnr_gain = psnr_denoised - psnr_noisy
            
            # Ligne 1 : Original
            ax = plt.subplot(3, n_samples, i + 1)
            ax.imshow(img_clean.permute(1, 2, 0).numpy())
            ax.axis('off')
            if i == 0:
                ax.text(-0.1, 0.5, 'Original', rotation=90, va='center', ha='right',
                       fontsize=14, fontweight='bold', transform=ax.transAxes)
            
            # Ligne 2 : Bruitée avec PSNR
            ax = plt.subplot(3, n_samples, i + 1 + n_samples)
            ax.imshow(img_noisy.permute(1, 2, 0).numpy())
            ax.axis('off')
            ax.set_title(f'{psnr_noisy:.1f} dB', fontsize=10, color='red')
            if i == 0:
                ax.text(-0.1, 0.5, 'Noisy', rotation=90, va='center', ha='right',
                       fontsize=14, fontweight='bold', transform=ax.transAxes)
            
            # Ligne 3 : Débruitée avec PSNR et gain
            ax = plt.subplot(3, n_samples, i + 1 + n_samples * 2)
            ax.imshow(img_denoised.permute(1, 2, 0).numpy())
            ax.axis('off')
            ax.set_title(f'{psnr_denoised:.1f} dB (+{psnr_gain:.1f})', 
                        fontsize=10, color='green', fontweight='bold')
            if i == 0:
                ax.text(-0.1, 0.5, 'Denoised', rotation=90, va='center', ha='right',
                       fontsize=14, fontweight='bold', transform=ax.transAxes)
        
        plt.subplots_adjust(left=0.05, right=0.98, top=0.94, bottom=0.02, 
                           wspace=0.05, hspace=0.15)
        
        # Sauvegarder
        output_path = f'./code/evaluation_unet_stl10_visual_{noise_config["name"].lower().replace(" & ", "_")}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualisation sauvegardée: {output_path}")
        
        plt.show()


def plot_visual_results_combined(model, test_dataset, noise_configs, device, n_samples=8, 
                                 patch_size=32, stride=16):
    """
    Version alternative : tous les types de bruit sur une seule grande figure
    8 images × 3 types de bruit × 3 lignes = figure très large
    """
    indices = np.random.choice(len(test_dataset), n_samples, replace=False)
    
    # Figure géante
    fig = plt.figure(figsize=(32, 9 * len(noise_configs)))
    
    for noise_idx, noise_config in enumerate(noise_configs):
        for i, idx in enumerate(indices):
            # Récupérer l'image propre
            img_clean, _ = test_dataset[idx]
            
            # Débruiter
            img_noisy, img_denoised = denoise_full_image(
                model, img_clean, noise_config['type'], noise_config['params'],
                device, patch_size, stride
            )
            
            # Calculer PSNR
            img_clean_uint8 = (img_clean.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_noisy_uint8 = (img_noisy.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_denoised_uint8 = (img_denoised.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            
            psnr_noisy = calculate_psnr(img_noisy_uint8, img_clean_uint8)
            psnr_denoised = calculate_psnr(img_denoised_uint8, img_clean_uint8)
            psnr_gain = psnr_denoised - psnr_noisy
            
            base_idx = noise_idx * 3 * n_samples
            
            # Original
            ax = plt.subplot(len(noise_configs) * 3, n_samples, base_idx + i + 1)
            ax.imshow(img_clean.permute(1, 2, 0).numpy())
            ax.axis('off')
            if i == 0:
                ax.text(-0.1, 0.5, f'{noise_config["name"]}\nOriginal', 
                       rotation=90, va='center', ha='right',
                       fontsize=13, fontweight='bold', transform=ax.transAxes)
            
            # Noisy
            ax = plt.subplot(len(noise_configs) * 3, n_samples, base_idx + i + 1 + n_samples)
            ax.imshow(img_noisy.permute(1, 2, 0).numpy())
            ax.axis('off')
            ax.set_title(f'{psnr_noisy:.1f} dB', fontsize=9, color='red')
            if i == 0:
                ax.text(-0.1, 0.5, 'Noisy', rotation=90, va='center', ha='right',
                       fontsize=13, fontweight='bold', transform=ax.transAxes)
            
            # Denoised
            ax = plt.subplot(len(noise_configs) * 3, n_samples, base_idx + i + 1 + n_samples * 2)
            ax.imshow(img_denoised.permute(1, 2, 0).numpy())
            ax.axis('off')
            ax.set_title(f'{psnr_denoised:.1f} dB (+{psnr_gain:.1f})', 
                        fontsize=9, color='green', fontweight='bold')
            if i == 0:
                ax.text(-0.1, 0.5, 'Denoised', rotation=90, va='center', ha='right',
                       fontsize=13, fontweight='bold', transform=ax.transAxes)
    
    plt.suptitle('U-Net STL-10 (96x96) - Comparaison des types de bruit', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    plt.subplots_adjust(left=0.03, right=0.99, top=0.99, bottom=0.01, 
                       wspace=0.03, hspace=0.1)
    
    # Sauvegarder
    output_path = './code/evaluation_unet_stl10_visual_combined.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualisation combinée sauvegardée: {output_path}")
    
    plt.show()


def plot_metrics(results):
    """
    Crée 2 graphiques : PSNR et MSE avec barres pour chaque type de bruit
    """
    noise_types = [r['name'] for r in results]
    
    psnr_noisy = [r['metrics']['avg_psnr_noisy'] for r in results]
    psnr_denoised = [r['metrics']['avg_psnr_denoised'] for r in results]
    
    mse_noisy = [r['metrics']['avg_mse_noisy'] for r in results]
    mse_denoised = [r['metrics']['avg_mse_denoised'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(noise_types))
    width = 0.35
    
    # PSNR
    bars1 = ax1.bar(x - width/2, psnr_noisy, width, label='Images bruitées', 
                     color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, psnr_denoised, width, label='Images débruitées', 
                     color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_xlabel('Type de bruit', fontsize=13, fontweight='bold')
    ax1.set_ylabel('PSNR (dB)', fontsize=13, fontweight='bold')
    ax1.set_title('U-Net STL-10 (96x96) - PSNR moyen', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(noise_types, fontsize=11)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
    
    # MSE
    bars3 = ax2.bar(x - width/2, mse_noisy, width, label='Images bruitées', 
                     color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars4 = ax2.bar(x + width/2, mse_denoised, width, label='Images débruitées', 
                     color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    for bar in bars4:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Type de bruit', fontsize=13, fontweight='bold')
    ax2.set_ylabel('MSE', fontsize=13, fontweight='bold')
    ax2.set_title('U-Net STL-10 (96x96) - MSE moyen', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(noise_types, fontsize=11)
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
    
    plt.tight_layout()
    
    output_path = './code/evaluation_unet_stl10.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Graphique sauvegardé: {output_path}")
    
    plt.show()


def main():
    print("=" * 80)
    print("ÉVALUATION DU MODÈLE U-NET SUR STL-10 (96x96)")
    print("Reconstruction d'images complètes à partir de patches 32x32 débruités")
    print("=" * 80)
    
    # Configuration
    data_dir = './code/dataset'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice utilisé: {device}")
    
    # Charger STL-10
    print("\nChargement de STL-10...")
    test_dataset = torchvision.datasets.STL10(
        data_dir, split='test', download=True,
        transform=transforms.ToTensor()
    )
    print(f"✓ {len(test_dataset)} images de test (96x96)")
    
    # Charger le modèle
    model_path = './code/unet_denoising_stl10_multinoise.pth'
    print(f"\nChargement du modèle: {model_path}")
    
    try:
        model = load_unet_model(model_path, device)
        print("✓ Modèle chargé avec succès!")
    except FileNotFoundError:
        print(f"❌ Fichier '{model_path}' introuvable")
        print("   Entraînez d'abord le modèle avec:")
        print("   python unet_train_multi_noise.py --dataset stl10")
        return
    
    # Configurations de bruit
    noise_configs = [
        {'name': 'Gaussien', 'type': 'gaussian', 'params': {'std': 25}},
        {'name': 'Salt & Pepper', 'type': 'salt_pepper', 'params': {'salt_prob': 0.02, 'pepper_prob': 0.02}},
        {'name': 'Mixte', 'type': 'mixed', 'params': {'gaussian_std': 20, 'salt_prob': 0.01, 'pepper_prob': 0.01}}
    ]
    
    # Évaluer pour chaque type de bruit
    results = []
    patch_size = 32
    stride = 16
    
    for config in noise_configs:
        print("\n" + "=" * 80)
        print(f"ÉVALUATION: {config['name']}")
        print("=" * 80)
        print(f"Paramètres: {config['params']}")
        print(f"Découpage: patches {patch_size}x{patch_size}, stride={stride}")
        
        metrics = calculate_metrics_for_noise_type(
            model=model,
            test_dataset=test_dataset,
            noise_type=config['type'],
            noise_params=config['params'],
            device=device,
            n_samples=100,
            patch_size=patch_size,
            stride=stride
        )
        
        results.append({'name': config['name'], 'metrics': metrics})
        
        print(f"\n--- RÉSULTATS ---")
        print(f"PSNR moyen (bruitées):   {metrics['avg_psnr_noisy']:.2f} dB")
        print(f"PSNR moyen (débruitées): {metrics['avg_psnr_denoised']:.2f} dB")
        print(f"→ GAIN PSNR:             +{metrics['psnr_gain']:.2f} dB")
        print(f"\nMSE moyen (bruitées):    {metrics['avg_mse_noisy']:.2f}")
        print(f"MSE moyen (débruitées):  {metrics['avg_mse_denoised']:.2f}")
        print(f"→ RÉDUCTION MSE:         -{metrics['mse_reduction']:.2f} ({metrics['mse_reduction_percent']:.1f}%)")
    
    # Résumé
    print("\n" + "=" * 80)
    print("RÉSUMÉ COMPARATIF")
    print("=" * 80)
    
    print("\n{:<20s} {:>15s} {:>15s} {:>15s}".format(
        "Type de bruit", "PSNR gain (dB)", "MSE réduction", "MSE réd. (%)"
    ))
    print("-" * 80)
    
    for result in results:
        m = result['metrics']
        print("{:<20s} {:>15.2f} {:>15.2f} {:>15.1f}".format(
            result['name'], m['psnr_gain'], m['mse_reduction'], m['mse_reduction_percent']
        ))
    
    print("-" * 80)
    
    # Graphiques
    print("\n" + "=" * 80)
    print("GÉNÉRATION DES GRAPHIQUES")
    print("=" * 80)
    
    plot_metrics(results)
    
    print("\n✓ Génération des visualisations (10 images par type de bruit)...")
    plot_visual_results(model, test_dataset, noise_configs, device, 
                       n_samples=10, patch_size=patch_size, stride=stride)
    
    print("\n✓ Génération de la visualisation combinée (8 images, tous les bruits)...")
    plot_visual_results_combined(model, test_dataset, noise_configs, device, 
                                 n_samples=8, patch_size=patch_size, stride=stride)
    
    print("\n" + "=" * 80)
    print("ÉVALUATION TERMINÉE ✓")
    print("=" * 80)
    print("\nFichiers générés:")
    print("  - evaluation_unet_stl10.png (métriques)")
    print("  - evaluation_unet_stl10_visual_gaussien.png")
    print("  - evaluation_unet_stl10_visual_salt_pepper.png")
    print("  - evaluation_unet_stl10_visual_mixte.png")
    print("  - evaluation_unet_stl10_visual_combined.png (tous les bruits)")


if __name__ == "__main__":
    main()