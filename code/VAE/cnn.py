import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from load_cifar10 import CIFAR10Loader
from load_stl10 import STL10Loader
from image_visualizer import ImageVisualizer
from utils import select_one_per_class, add_noise_to_images
from vae_model import Encoder, Decoder
from vae_train import train_vae, evaluate_vae, plot_training_history, save_model, load_model
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


def partition_dataset(x, y, train_ratio=0.8, seed=42):
    """
    Sépare le dataset en deux sous-ensembles (train / évaluation) sans chevauchement.

    Args:
        x (np.ndarray): Images complètes
        y (np.ndarray): Labels
        train_ratio (float): Proportion dédiée à l'entraînement
        seed (int): graine pour la reproductibilité

    Returns:
        tuple: (x_train_split, y_train_split, x_eval_split, y_eval_split)
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio doit être compris entre 0 et 1 exclu")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(x))
    rng.shuffle(indices)

    n_train = int(len(indices) * train_ratio)
    train_idx = indices[:n_train]
    eval_idx = indices[n_train:]

    x_train_split = x[train_idx]
    y_train_split = y[train_idx]
    x_eval_split = x[eval_idx]
    y_eval_split = y[eval_idx]

    return x_train_split, y_train_split, x_eval_split, y_eval_split


def plot_average_psnr_gain(metrics_by_noise):
    """Trace un graphique montrant le gain moyen (PSNR) pour chaque type de bruit."""
    labels = [label for label, _ in metrics_by_noise]
    gains = [metrics['psnr_improvement'] for _, metrics in metrics_by_noise]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, gains, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.ylabel('Gain PSNR moyen (dB)')
    plt.title('Dénombrement des dB gagnés en moyenne par type de bruit')

    for bar, gain in zip(bars, gains):
        y_pos = bar.get_height()
        offset = 0.05 if gain >= 0 else -0.1
        va = 'bottom' if gain >= 0 else 'top'
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos + offset,
            f"{gain:.2f} dB",
            ha='center',
            va=va,
            fontsize=10
        )

    plt.tight_layout()
    plt.show()


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

    # Créer un split explicite entraînement/évaluation au sein du CIFAR-10 train set
    train_ratio = 0.8
    x_train_vae, y_train_vae, x_eval_vae, y_eval_vae = partition_dataset(
        x_train,
        y_train,
        train_ratio=train_ratio,
        seed=42
    )

    print("\n" + "-" * 60)
    print("Split personnalisé CIFAR-10 pour le VAE")
    print("-" * 60)
    print(f"Taille totale (train officiel): {len(x_train)}")
    print(f"Portion entraînement VAE ({int(train_ratio*100)}%): {len(x_train_vae)} images")
    print(f"Portion évaluation VAE ({int((1-train_ratio)*100)}%): {len(x_eval_vae)} images")
    print("-" * 60)
    
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
    use_pretrained = True
    pretrained_candidates = [
        ('./code/vae_denoiser_beta0.pth', 'BETA=0'),
    ]
    checkpoint_to_load = None
    checkpoint_label = None
    for path, label in pretrained_candidates:
        if os.path.exists(path):
            checkpoint_to_load = path
            checkpoint_label = label
            break

    history = None

    if use_pretrained and checkpoint_to_load is not None:
        print("\n" + "=" * 60)
        print(f"CHARGEMENT DU MODÈLE {checkpoint_label}")
        print("=" * 60)
        encoder, decoder, history = load_model(
            encoder,
            decoder,
            filepath=checkpoint_to_load,
            device=device
        )
    else:
        print("\n" + "=" * 60)
        print("ENTRAÎNEMENT D'UN NOUVEAU MODÈLE")
        print("=" * 60)
        history = train_vae(
            encoder=encoder,
            decoder=decoder,
            x_train=x_train_vae,
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
        
        # Sauvegarder le modèle fraîchement entraîné
        save_model(encoder, decoder, history, filepath='./code/vae_denoiser.pth')
    
    # Afficher l'historique (qu'il provienne d'un entraînement ou d'un checkpoint)
    if history is not None:
        plot_training_history(history)
    
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
        x_test=x_eval_vae,
        y_test=y_eval_vae,
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
        x_test=x_eval_vae,
        y_test=y_eval_vae,
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
        x_test=x_eval_vae,
        y_test=y_eval_vae,
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

    plot_average_psnr_gain([
        ("Gaussien", metrics_gaussian),
        ("Salt & Pepper", metrics_salt_pepper),
        ("Mixte", metrics_mixed)
    ])
    
    return (
        loader,
        x_train_vae,
        y_train_vae,
        x_eval_vae,
        y_eval_vae,
        encoder,
        decoder
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline VAE débruiteur")
    parser.add_argument(
        '--dataset',
        choices=['cifar10', 'stl10'],
        default='cifar10',
        help="Dataset à utiliser pour l'entraînement et l'évaluation"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Lancer l'entraînement et l'évaluation complète du VAE
    loader, x_train, y_train, x_test, y_test, encoder, decoder = main()
    
    print("\n" + "=" * 60)
    print("PROGRAMME TERMINÉ ✔")
    print("=" * 60)
    print("\nLe modèle VAE a été entraîné et évalué avec succès!")
    print("Le modèle a été sauvegardé dans './code/vae_denoiser.pth'")
    print("\nVous pouvez maintenant utiliser le débruiteur pour nettoyer vos images.")


