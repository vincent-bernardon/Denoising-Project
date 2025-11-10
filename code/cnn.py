import torch
import torch.nn as nn
import torch.optim as optim
from load_cifar10 import CIFAR10Loader
from image_visualizer import ImageVisualizer
from utils import select_one_per_class, add_noise_to_images
from vae_model import Encoder, Decoder
from vae_train import train_vae, evaluate_vae, plot_training_history, save_model
import numpy as np


def load_data():
    """
    Charge le dataset CIFAR-10
    
    Returns:
        tuple: (loader, x_train, y_train, x_test, y_test)
    """
    print("=" * 60)
    print("Chargement du dataset CIFAR-10")
    print("=" * 60)
    
    loader = CIFAR10Loader()
    x_train, y_train, x_test, y_test = loader.load_all_data()
    
    loader.print_info()
    
    return loader, x_train, y_train, x_test, y_test


def run_visualization_demo(loader, x_train, y_train, x_test=None, y_test=None):
    """
    Regroupe toute la logique de démonstration/visualisation :
      - affiche une image par classe
      - démontre les types de bruit sur une image (frog)
      - compare clean vs noisy (gaussian & mixed) en utilisant la même sélection

    Cette fonction permet de garder `main()` propre pour l'intégration du VAE plus tard.
    """
    # 1. Sélectionner une image par classe
    print("\n" + "=" * 60)
    print("DEMO: Une image par classe")
    print("=" * 60)
    indices = select_one_per_class(y_train, n_classes=len(loader.class_names))
    ImageVisualizer.visualize_clean_images(x_train[indices], y_train[indices], loader.class_names, n_samples=len(indices))

    # 2. Démonstration des types de bruit sur une image 'frog' (label 6) si possible
    print("\n" + "=" * 60)
    print("DEMO: Types de bruit (ex. 'frog')")
    print("=" * 60)
    try:
        frog_idx = int(np.where(y_train == 6)[0][0])
    except Exception:
        frog_idx = indices[0]
    sample_image = x_train[frog_idx]
    sample_label = loader.class_names[int(y_train[frog_idx])]
    ImageVisualizer.demonstrate_noise_types(sample_image, sample_label)

    # 3. Comparaison clean vs noisy (Gaussien) en utilisant les mêmes indices
    print("\n" + "=" * 60)
    print("DEMO: Comparaison clean vs noisy (Gaussien)")
    print("=" * 60)
    noisy_images = add_noise_to_images(x_train[indices], noise_type='gaussian', std=25)
    ImageVisualizer.compare_clean_and_noisy(x_train[indices], noisy_images, y_train[indices], loader.class_names, noise_type='gaussian', n_samples=len(indices))

    # 4. Comparaison clean vs noisy (Mixte)
    print("\n" + "=" * 60)
    print("DEMO: Comparaison clean vs noisy (Mixte)")
    print("=" * 60)
    mixed_noisy = add_noise_to_images(x_train[indices], noise_type='mixed', gaussian_std=20, salt_prob=0.01, pepper_prob=0.01)
    ImageVisualizer.compare_clean_and_noisy(x_train[indices], mixed_noisy, y_train[indices], loader.class_names, noise_type='mixed', n_samples=len(indices))

    print("\nDEMO terminée ✔")


def main():
    """
    Fonction principale - charge les données, entraîne et évalue le VAE
    """
    # Charger les données
    loader, x_train, y_train, x_test, y_test = load_data()
    
    # ====================================================================
    # INITIALISATION DU VAE
    # ====================================================================
    print("\n" + "=" * 60)
    print("INITIALISATION DU VAE")
    print("=" * 60)
    
    latent_dim = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = Encoder(latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim)
    
    print(f"Encodeur initialisé (latent_dim={latent_dim})")
    print(f"Décodeur initialisé (latent_dim={latent_dim})")
    print(f"Device: {device}")
    
    # ====================================================================
    # ÉTAPE 7 : ENTRAÎNEMENT DU VAE
    # ====================================================================
    history = train_vae(
        encoder=encoder,
        decoder=decoder,
        x_train=x_train,
        epochs=80,                      # Plus d'epochs pour convergence complète
        batch_size=128,                 # Taille des batchs
        learning_rate=1e-3,             # Taux d'apprentissage
        noise_type='gaussian',          # Type de bruit pour l'entraînement
        noise_params={'std': 25},       # Paramètres du bruit
        beta=0.01,                      # Beta très faible (quasi-autoencodeur) pour + de netteté
        device=device,
        validation_split=0.1,           # 10% pour validation
        verbose=True
    )
    
    # Afficher l'historique d'entraînement
    plot_training_history(history)
    
    # Sauvegarder le modèle
    save_model(encoder, decoder, history, filepath='./code/vae_denoiser.pth')
    
    # ====================================================================
    # ÉTAPE 8 : ÉVALUATION DU VAE
    # ====================================================================
    
    # 1. Évaluation avec bruit Gaussien
    print("\n" + "=" * 60)
    print("ÉVALUATION 1/3 : Bruit Gaussien")
    print("=" * 60)
    metrics_gaussian = evaluate_vae(
        encoder=encoder,
        decoder=decoder,
        x_test=x_test,
        y_test=y_test,
        class_names=loader.class_names,
        noise_type='gaussian',
        noise_params={'std': 25},
        n_samples=5,
        device=device,
        verbose=True
    )
    
    # 2. Évaluation avec bruit Salt & Pepper
    print("\n" + "=" * 60)
    print("ÉVALUATION 2/3 : Bruit Salt & Pepper")
    print("=" * 60)
    metrics_salt_pepper = evaluate_vae(
        encoder=encoder,
        decoder=decoder,
        x_test=x_test,
        y_test=y_test,
        class_names=loader.class_names,
        noise_type='salt_pepper',
        noise_params={'salt_prob': 0.02, 'pepper_prob': 0.02},
        n_samples=5,
        device=device,
        verbose=True
    )
    
    # 3. Évaluation avec bruit Mixte
    print("\n" + "=" * 60)
    print("ÉVALUATION 3/3 : Bruit Mixte")
    print("=" * 60)
    metrics_mixed = evaluate_vae(
        encoder=encoder,
        decoder=decoder,
        x_test=x_test,
        y_test=y_test,
        class_names=loader.class_names,
        noise_type='mixed',
        noise_params={'gaussian_std': 20, 'salt_prob': 0.01, 'pepper_prob': 0.01},
        n_samples=5,
        device=device,
        verbose=True
    )
    
    # ====================================================================
    # RÉSUMÉ FINAL
    # ====================================================================
    print("\n" + "=" * 60)
    print("RÉSUMÉ DES PERFORMANCES")
    print("=" * 60)
    print(f"\n{'Type de bruit':<20s} {'MSE avant':<12s} {'MSE après':<12s} {'PSNR avant':<12s} {'PSNR après':<12s}")
    print("-" * 75)
    print(f"{'Gaussien':<20s} {metrics_gaussian['mse_noisy_vs_clean']:<12.2f} {metrics_gaussian['mse_denoised_vs_clean']:<12.2f} {metrics_gaussian['psnr_noisy_vs_clean']:<12.2f} {metrics_gaussian['psnr_denoised_vs_clean']:<12.2f}")
    print(f"{'Salt & Pepper':<20s} {metrics_salt_pepper['mse_noisy_vs_clean']:<12.2f} {metrics_salt_pepper['mse_denoised_vs_clean']:<12.2f} {metrics_salt_pepper['psnr_noisy_vs_clean']:<12.2f} {metrics_salt_pepper['psnr_denoised_vs_clean']:<12.2f}")
    print(f"{'Mixte':<20s} {metrics_mixed['mse_noisy_vs_clean']:<12.2f} {metrics_mixed['mse_denoised_vs_clean']:<12.2f} {metrics_mixed['psnr_noisy_vs_clean']:<12.2f} {metrics_mixed['psnr_denoised_vs_clean']:<12.2f}")
    print("-" * 75)
    
    return loader, x_train, y_train, x_test, y_test, encoder, decoder


if __name__ == "__main__":
    
    # Lancer l'entraînement et l'évaluation complète du VAE
    loader, x_train, y_train, x_test, y_test, encoder, decoder = main()
    
    print("\n" + "=" * 60)
    print("PROGRAMME TERMINÉ ✔")
    print("=" * 60)
    print("\nLe modèle VAE a été entraîné et évalué avec succès!")
    print("Le modèle a été sauvegardé dans './code/vae_denoiser.pth'")
    print("\nVous pouvez maintenant utiliser le débruiteur pour nettoyer vos images.")


