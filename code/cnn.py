import torch
import torch.nn as nn
import torch.optim as optim
from load_cifar10 import CIFAR10Loader
import matplotlib.pyplot as plt
import numpy as np


def main():    
    #charger un dataset
    loader = CIFAR10Loader()
    
    #charger/remplir les données
    print("\nChargement des données...")
    x_train, y_train, x_test, y_test = loader.load_all_data()
    
    # Afficher les informations
    loader.print_info()
    
    print("\nAffichage de quelques exemples d'images...")
    visualize_images(loader.x_train, loader.y_train, loader.class_names, n_samples=10)


def visualize_images(images, labels, class_names, n_samples=10):
    """
    Affiche une grille d'images
    
    Args:
        images: Array d'images (N, 32, 32, 3)
        labels: Array de labels (N,)
        class_names: Liste des noms de classes
        n_samples: Nombre d'images à afficher
    """
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    #affichage d'exmple d'images de tout les classes
    for i in range(n_samples):
        img = images[i]
        axes[i].imshow(img)
        axes[i].set_title(f'{class_names[labels[i]]}', fontsize=12, fontweight='bold')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    save_path = 'code/cifar10_samples.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Images sauvegardées dans '{save_path}'")
    
    plt.show()


if __name__ == "__main__":
    main()
