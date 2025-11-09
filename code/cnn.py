import torch
import torch.nn as nn
import torch.optim as optim
from load_cifar10 import CIFAR10Loader
from noise_generator import NoiseGenerator
from image_visualizer import ImageVisualizer
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


def main():
    loader, x_train, y_train, x_test, y_test = load_data()
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


if __name__ == "__main__":
    # main reste minimal — charger les données et lancer la démo pour l'instant
    loader, x_train, y_train, x_test, y_test = main()
    run_visualization_demo(loader, x_train, y_train, x_test, y_test)
