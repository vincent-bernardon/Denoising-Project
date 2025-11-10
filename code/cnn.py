import torch
import torch.nn as nn
import torch.optim as optim
from load_cifar10 import CIFAR10Loader
from image_visualizer import ImageVisualizer
from utils import select_one_per_class, add_noise_to_images
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
    Fonction principale - charge les données
    """
    loader, x_train, y_train, x_test, y_test = load_data()
    return loader, x_train, y_train, x_test, y_test


if __name__ == "__main__":
    import sys
    
    # Charger les données
    loader, x_train, y_train, x_test, y_test = main()

    print("\n" + "=" * 60)
    print("Lancement du test de la loss function")
    print("=" * 60)
    from test_vae import test_loss_function
    test_loss_function(x_train, y_train, loader.class_names)

