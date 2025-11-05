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
    """
    Fonction principale
    """
    # 1. Charger les données
    loader, x_train, y_train, x_test, y_test = load_data()
    
    # 2. Afficher quelques images propres
    print("\n" + "=" * 60)
    print("ÉTAPE 1: Visualisation des images propres")
    print("=" * 60)
    ImageVisualizer.visualize_clean_images(x_train, y_train, loader.class_names, n_samples=10)
    
    # 3. Démonstration de tous les types de bruit sur une image
    print("\n" + "=" * 60)
    print("ÉTAPE 2: Démonstration des types de bruit")
    print("=" * 60)
    sample_image = x_train[0]
    sample_label = loader.class_names[y_train[0]]
    ImageVisualizer.demonstrate_noise_types(sample_image, sample_label)
    
    # 4. Ajouter du bruit gaussien à plusieurs images
    print("\n" + "=" * 60)
    print("ÉTAPE 3: Comparaison images propres vs bruitées (Gaussien)")
    print("=" * 60)
    noisy_images = add_noise_to_images(
        x_train[:5], 
        noise_type='gaussian', 
        std=25
    )
    ImageVisualizer.compare_clean_and_noisy(
        x_train[:5], noisy_images, y_train, loader.class_names, 
        noise_type='gaussian', n_samples=5
    )
    
    # 5. Tester avec un bruit mixte (plus réaliste)
    print("\n" + "=" * 60)
    print("ÉTAPE 4: Test avec bruit mixte")
    print("=" * 60)
    mixed_noisy = add_noise_to_images(
        x_train[5:10],
        noise_type='mixed',
        gaussian_std=20,
        salt_prob=0.01,
        pepper_prob=0.01
    )
    ImageVisualizer.compare_clean_and_noisy(
        x_train[5:10], mixed_noisy, y_train[5:10], loader.class_names, 
        noise_type='mixed', n_samples=5
    )
    
    print("\n" + "=" * 60)
    print("✓ Démonstration terminée avec succès!")
    print("=" * 60)


if __name__ == "__main__":
    main()
