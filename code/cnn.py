import torch
import torch.nn as nn
import torch.optim as optim
from load_cifar10 import CIFAR10Loader
from noise_generator import NoiseGenerator
from image_visualizer import ImageVisualizer
from vae_model import Encoder
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


def select_one_per_class(labels, n_classes=10):
    """
    Retourne les indices de la première occurrence de chaque classe (0..n_classes-1).

    Args:
        labels: array-like des labels (ex: y_train)
        n_classes: nombre de classes à récupérer

    Returns:
        list d'indices (longueur <= n_classes)
    """
    indices = []
    seen = set()
    for i, lab in enumerate(labels):
        lab_int = int(lab)
        if lab_int not in seen:
            indices.append(i)
            seen.add(lab_int)
            if len(seen) >= n_classes:
                break
    return indices


def add_noise_to_images(images, noise_type='gaussian', **noise_params):
    """
    Ajoute du bruit aux images
    
    Args:
        images: Array d'images (N, 32, 32, 3)
        noise_type: Type de bruit ('gaussian', 'salt_pepper', 'mixed', etc.)
        **noise_params: Paramètres du bruit
        
    Returns:
        Images bruitées
    """
    print(f"\nAjout de bruit ({noise_type})...")
    
    if noise_type == 'gaussian':
        noisy_images = NoiseGenerator.add_gaussian_noise(images, **noise_params)
    elif noise_type == 'salt_pepper':
        noisy_images = NoiseGenerator.add_salt_and_pepper_noise(images, **noise_params)
    elif noise_type == 'speckle':
        noisy_images = NoiseGenerator.add_speckle_noise(images, **noise_params)
    elif noise_type == 'poisson':
        noisy_images = NoiseGenerator.add_poisson_noise(images)
    elif noise_type == 'uniform':
        noisy_images = NoiseGenerator.add_uniform_noise(images, **noise_params)
    elif noise_type == 'mixed':
        noisy_images = NoiseGenerator.add_mixed_noise(images, **noise_params)
    else:
        raise ValueError(f"Type de bruit inconnu: {noise_type}")
    
    print(f"✓ Bruit ajouté avec succès")
    return noisy_images


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


def test_vae_encoder(x_train, y_train, class_names):
    """
    Teste l'encodeur VAE avec des données réelles de CIFAR-10 bruitées
    
    Args:
        x_train: Images d'entraînement (N, 32, 32, 3)
        y_train: Labels d'entraînement
        class_names: Noms des classes
    """
    print("\n" + "=" * 60)
    print("Test de l'Encodeur VAE avec images bruitées")
    print("=" * 60)
    
    # Créer un encodeur
    latent_dim = 128
    encoder = Encoder(latent_dim=latent_dim)
    
    # Sélectionner quelques images réelles (une par classe)
    indices = select_one_per_class(y_train, n_classes=10)
    batch_images = x_train[indices]
    batch_labels = y_train[indices]
    
    print(f"\nNombre d'images sélectionnées: {len(indices)}")
    print(f"Classes: {[class_names[int(label)] for label in batch_labels]}")
    
    # Choisir un type de bruit aléatoire
    noise_types = ['gaussian', 'salt_pepper', 'speckle', 'poisson', 'uniform', 'mixed']
    noise_type = np.random.choice(noise_types)
    
    print(f"\nType de bruit appliqué: {noise_type}")
    
    # Ajouter du bruit aux images
    if noise_type == 'gaussian':
        noisy_images = NoiseGenerator.add_gaussian_noise(batch_images, std=25)
    elif noise_type == 'salt_pepper':
        noisy_images = NoiseGenerator.add_salt_and_pepper_noise(batch_images, salt_prob=0.02, pepper_prob=0.02)
    elif noise_type == 'speckle':
        noisy_images = NoiseGenerator.add_speckle_noise(batch_images, std=0.1)
    elif noise_type == 'poisson':
        noisy_images = NoiseGenerator.add_poisson_noise(batch_images)
    elif noise_type == 'uniform':
        noisy_images = NoiseGenerator.add_uniform_noise(batch_images, low=-30, high=30)
    else:  # mixed
        noisy_images = NoiseGenerator.add_mixed_noise(batch_images, gaussian_std=20, salt_prob=0.01, pepper_prob=0.01)
    
    # Convertir en tenseur PyTorch (N, H, W, C) -> (N, C, H, W)
    x = torch.from_numpy(noisy_images).permute(0, 3, 1, 2).float()
    
    print(f"\nInput shape: {x.shape}")
    print(f"Latent dimension: {latent_dim}")
    
    # Forward pass
    mu, logvar = encoder(x)
    
    print(f"\nOutput mu shape: {mu.shape}")
    print(f"Output logvar shape: {logvar.shape}")
    
    # Vérifier les statistiques
    print(f"\nStatistiques de mu:")
    print(f"  Mean: {mu.mean().item():.4f}")
    print(f"  Std: {mu.std().item():.4f}")
    print(f"  Min: {mu.min().item():.4f}")
    print(f"  Max: {mu.max().item():.4f}")
    
    print(f"\nStatistiques de logvar:")
    print(f"  Mean: {logvar.mean().item():.4f}")
    print(f"  Std: {logvar.std().item():.4f}")
    print(f"  Min: {logvar.min().item():.4f}")
    print(f"  Max: {logvar.max().item():.4f}")
    
    # Test de la reparamétrisation
    print("\n" + "-" * 60)
    print("Test de la reparamétrisation")
    print("-" * 60)
    
    z = encoder.reparameterize(mu, logvar)
    
    print(f"\nOutput z shape: {z.shape}")
    print(f"\nStatistiques de z (vecteur latent):")
    print(f"  Mean: {z.mean().item():.4f}")
    print(f"  Std: {z.std().item():.4f}")
    print(f"  Min: {z.min().item():.4f}")
    print(f"  Max: {z.max().item():.4f}")
    
    # Vérifier que z est différent à chaque appel (stochasticité)
    z2 = encoder.reparameterize(mu, logvar)
    diff = torch.abs(z - z2).mean().item()
    print(f"\nDifférence moyenne entre 2 échantillons: {diff:.4f}")
    print(f"✓ Reparamétrisation stochastique: {'Oui' if diff > 0.01 else 'Non (problème!)'}")
    
    # Compter les paramètres
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    
    print(f"\nNombre de paramètres:")
    print(f"  Total: {total_params:,}")
    print(f"  Entraînables: {trainable_params:,}")
    
    print("\n" + "=" * 60)
    print("✓ Test de l'encodeur réussi!")
    print("=" * 60)


def main():
    loader, x_train, y_train, x_test, y_test = load_data()
    return loader, x_train, y_train, x_test, y_test

if __name__ == "__main__":
    # main reste minimal — charger les données et tester l'encodeur VAE
    loader, x_train, y_train, x_test, y_test = main()
    test_vae_encoder(x_train, y_train, loader.class_names)
