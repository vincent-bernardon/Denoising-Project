import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from image_visualizer import ImageVisualizer
from load_stl10 import STL10Loader
from utils import select_one_per_class, add_noise_to_images
from vae_model_96 import Encoder96, Decoder96
from vae_train import train_vae, evaluate_vae, plot_training_history, save_model, load_model


def load_data():
    print("=" * 60)
    print("Chargement du dataset STL10")
    print("=" * 60)
    loader = STL10Loader()
    x_train, y_train = loader.load_train_data()
    x_test, y_test = loader.load_test_data()
    loader.print_info()
    return loader, x_train, y_train, x_test, y_test


def partition_dataset(x, y, train_ratio=0.8, seed=42):
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio doit être compris entre 0 et 1 exclu")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(x))
    rng.shuffle(indices)

    n_train = int(len(indices) * train_ratio)
    train_idx = indices[:n_train]
    eval_idx = indices[n_train:]
    return x[train_idx], y[train_idx], x[eval_idx], y[eval_idx]


def plot_average_psnr_gain(metrics_by_noise):
    labels = [label for label, _ in metrics_by_noise]
    gains = [metrics['psnr_improvement'] for _, metrics in metrics_by_noise]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, gains, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.ylabel('Gain PSNR moyen (dB)')
    plt.title('dB gagnés en moyenne (STL-10)')

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


def run_visualization_demo(loader, x_train, y_train):
    print("\n" + "=" * 60)
    print("DEMO STL10: Une image par classe")
    print("=" * 60)
    indices = select_one_per_class(y_train, n_classes=len(loader.class_names))
    ImageVisualizer.visualize_clean_images(
        x_train[indices],
        y_train[indices],
        loader.class_names,
        n_samples=len(indices)
    )


def main(args):
    loader, x_train, y_train, x_test, y_test = load_data()
    print("\nDataset sélectionné: STL10 (96x96)")
    print("Les images restent en 96x96 dans le modèle, sans redimensionnement vers 32x32.")

    x_train_vae, y_train_vae, x_eval_vae, y_eval_vae = partition_dataset(
        x_train,
        y_train,
        train_ratio=args.train_ratio,
        seed=42
    )

    print("\n" + "-" * 60)
    print("Split personnalisé STL-10 pour le VAE")
    print("-" * 60)
    print(f"Taille totale (train officiel): {len(x_train)}")
    print(f"Portion entraînement VAE ({int(args.train_ratio * 100)}%): {len(x_train_vae)} images")
    print(f"Portion évaluation VAE ({int((1-args.train_ratio) * 100)}%): {len(x_eval_vae)} images")
    print("-" * 60)

    latent_dim = args.latent_dim
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = Encoder96(latent_dim=latent_dim)
    decoder = Decoder96(latent_dim=latent_dim)

    print("\n" + "=" * 60)
    print("INITIALISATION DU VAE 96x96")
    print("=" * 60)
    print(f"latent_dim={latent_dim}")
    print(f"Device: {device}")

    checkpoint_prefix = './code/vae_denoiser_stl96'
    pretrained_candidates = [
        (f'{checkpoint_prefix}_beta0.pth', 'STL32 BETA=0'),
        (f'{checkpoint_prefix}.pth', 'STL32 dernier entraînement')
    ]
    checkpoint_to_load = None
    checkpoint_label = None
    for path, label in pretrained_candidates:
        if os.path.exists(path):
            checkpoint_to_load = path
            checkpoint_label = label
            break

    history = None
    use_pretrained = not args.no_pretrained

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
        print("ENTRAÎNEMENT D'UN NOUVEAU MODÈLE 96x96")
        print("=" * 60)
        history = train_vae(
            encoder=encoder,
            decoder=decoder,
            x_train=x_train_vae,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            noise_type='gaussian',
            noise_params={'std': 25},
            beta=args.beta,
            device=device,
            validation_split=0.1,
            verbose=True
        )
        save_path = f'{checkpoint_prefix}_beta0.pth' if args.beta <= 0.001 else f'{checkpoint_prefix}.pth'
        save_model(encoder, decoder, history, filepath=save_path)

    if history is not None:
        plot_training_history(history)

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

    print("\n" + "=" * 60)
    print("RÉSUMÉ DES PERFORMANCES STL-10")
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
    parser = argparse.ArgumentParser(description="Pipeline VAE STL-10 (96x96)")
    parser.add_argument('--epochs', type=int, default=80, help='Nombre d\'epochs pour l\'entraînement')
    parser.add_argument('--batch-size', type=int, default=128, help='Taille des batchs')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help="Taux d'apprentissage")
    parser.add_argument('--beta', type=float, default=0.01, help='Coefficient KL (beta-VAE)')
    parser.add_argument('--latent-dim', type=int, default=128, help='Dimension de l\'espace latent')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Proportion du train dédiée à l\'entraînement VAE')
    parser.add_argument('--no-pretrained', action='store_true', help='Forcer un ré-entrainement complet (ignore les checkpoints)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print("\n" + "=" * 60)
    print("PIPELINE STL-10 TERMINÉE ✔")
    print("=" * 60)