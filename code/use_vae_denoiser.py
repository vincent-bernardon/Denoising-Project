import torch
import numpy as np
import matplotlib.pyplot as plt

from load_cifar10 import CIFAR10Loader
from vae_model import Encoder, Decoder
from vae_train import load_model, calculate_mse, calculate_psnr
from utils import add_noise_to_images


def denoise_image(encoder, decoder, image, device=None):
    """
    Débruite une seule image
    
    Args:
        encoder (nn.Module): Encodeur du VAE
        decoder (nn.Module): Décodeur du VAE
        image (np.ndarray): Image bruitée (H, W, C) - uint8
        device (torch.device): Device
    
    Returns:
        np.ndarray: Image débruitée (H, W, C) - uint8
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder.eval()
    decoder.eval()
    
    # Gérer le cas d'une seule image
    single_image = False
    if image.ndim == 3:
        image = image[np.newaxis, ...]
        single_image = True
    
    # Convertir en tensor PyTorch
    image_tensor = torch.FloatTensor(image).permute(0, 3, 1, 2) / 255.0
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        # Encodage
        mu, logvar = encoder(image_tensor)
        
        # Utiliser mu (sans sampling) pour l'inférence
        z = mu
        
        # Décodage
        denoised_tensor = decoder(z)
    
    # Convertir en numpy
    denoised = (denoised_tensor.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    
    if single_image:
        return denoised[0]
    return denoised


def main():
    """
    Exemple d'utilisation du VAE débruiteur
    """
    print("=" * 60)
    print("EXEMPLE D'UTILISATION DU VAE DÉBRUITEUR")
    print("=" * 60)
    
    # 1. Charger les données CIFAR-10
    print("\n1) Chargement des données CIFAR-10...")
    loader = CIFAR10Loader()
    x_train, y_train, x_test, y_test = loader.load_all_data()
    print(f"   → {len(x_test)} images de test chargées")
    
    # 2. Charger le modèle pré-entraîné
    print("\n2) Chargement du modèle VAE pré-entraîné...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_dim = 128
    
    encoder = Encoder(latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim)
    
    try:
        encoder, decoder, history = load_model(encoder, decoder, filepath='./code/vae_denoiser.pth', device=device)
        print(f"   → Modèle chargé avec succès")
        print(f"   → Entraîné pendant {len(history['epochs'])} epochs")
    except FileNotFoundError:
        print("   ⚠ Erreur: Le fichier './code/vae_denoiser.pth' n'existe pas.")
        print("   → Veuillez d'abord entraîner le modèle en exécutant 'python cnn.py'")
        return
    
    # 3. Débruiter une seule image
    print("\n3) Débruitage d'une image individuelle...")
    
    # Sélectionner une image aléatoire
    test_idx = np.random.randint(0, len(x_test))
    test_image_clean = x_test[test_idx]
    test_label = loader.class_names[y_test[test_idx]]
    
    # Ajouter du bruit
    test_image_noisy = add_noise_to_images(test_image_clean[np.newaxis, ...], 
                                           noise_type='gaussian', 
                                           std=25)[0]
    
    # Débruiter
    test_image_denoised = denoise_image(encoder, decoder, test_image_noisy, device=device)
    
    # Calculer les métriques
    mse_before = calculate_mse(test_image_noisy, test_image_clean)
    mse_after = calculate_mse(test_image_denoised, test_image_clean)
    psnr_before = calculate_psnr(test_image_noisy, test_image_clean)
    psnr_after = calculate_psnr(test_image_denoised, test_image_clean)
    
    print(f"   → Image: {test_label}")
    print(f"   → MSE avant: {mse_before:.2f} | MSE après: {mse_after:.2f}")
    print(f"   → PSNR avant: {psnr_before:.2f} dB | PSNR après: {psnr_after:.2f} dB")
    
    # Visualiser
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(test_image_noisy)
    axes[0].set_title(f'Image bruitée\nMSE: {mse_before:.1f} | PSNR: {psnr_before:.1f} dB')
    axes[0].axis('off')
    
    axes[1].imshow(test_image_denoised)
    axes[1].set_title(f'Image débruitée (VAE)\nMSE: {mse_after:.1f} | PSNR: {psnr_after:.1f} dB')
    axes[1].axis('off')
    
    axes[2].imshow(test_image_clean)
    axes[2].set_title('Image propre (Ground Truth)')
    axes[2].axis('off')
    
    fig.suptitle(f'Débruitage d\'une image ({test_label})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 4. Débruiter un batch d'images
    print("\n4) Débruitage d'un batch de 8 images...")
    
    # Sélectionner 8 images aléatoires
    test_indices = np.random.choice(len(x_test), size=8, replace=False)
    test_images_clean = x_test[test_indices]
    
    # Ajouter du bruit (mixte cette fois)
    test_images_noisy = add_noise_to_images(test_images_clean, 
                                            noise_type='mixed',
                                            gaussian_std=20,
                                            salt_prob=0.01,
                                            pepper_prob=0.01)
    
    # Débruiter
    test_images_denoised = denoise_image(encoder, decoder, test_images_noisy, device=device)
    
    # Calculer les métriques moyennes
    mse_batch_before = calculate_mse(test_images_noisy, test_images_clean)
    mse_batch_after = calculate_mse(test_images_denoised, test_images_clean)
    psnr_batch_before = calculate_psnr(test_images_noisy, test_images_clean)
    psnr_batch_after = calculate_psnr(test_images_denoised, test_images_clean)
    
    print(f"   → MSE moyen avant: {mse_batch_before:.2f} | MSE moyen après: {mse_batch_after:.2f}")
    print(f"   → PSNR moyen avant: {psnr_batch_before:.2f} dB | PSNR moyen après: {psnr_batch_after:.2f} dB")
    
    # Visualiser le batch
    fig, axes = plt.subplots(8, 3, figsize=(12, 32))
    
    for i in range(8):
        # Image bruitée
        axes[i, 0].imshow(test_images_noisy[i])
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title('Image bruitée', fontsize=12, fontweight='bold')
        axes[i, 0].text(0.5, -0.1, loader.class_names[y_test[test_indices[i]]], 
                       ha='center', va='top', transform=axes[i, 0].transAxes)
        
        # Image débruitée
        axes[i, 1].imshow(test_images_denoised[i])
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].set_title('Image débruitée (VAE)', fontsize=12, fontweight='bold')
        
        # Image propre
        axes[i, 2].imshow(test_images_clean[i])
        axes[i, 2].axis('off')
        if i == 0:
            axes[i, 2].set_title('Image propre (GT)', fontsize=12, fontweight='bold')
    
    fig.suptitle('Débruitage d\'un batch d\'images (bruit mixte)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 60)
    print("EXEMPLE TERMINÉ ✔")
    print("=" * 60)


if __name__ == "__main__":
    main()
