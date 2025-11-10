import numpy as np
from noise_generator import NoiseGenerator


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


if __name__ == "__main__":
    print("Module utils.py - Fonctions utilitaires")
    print("\nFonctions disponibles:")
    print("  - select_one_per_class(labels, n_classes=10)")
    print("  - add_noise_to_images(images, noise_type='gaussian', **noise_params)")
