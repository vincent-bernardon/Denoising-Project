import matplotlib.pyplot as plt
import numpy as np


class ImageVisualizer:
    """
    Classe pour gérer toutes les visualisations d'images
    """
    
    @staticmethod
    def visualize_clean_images(images, labels, class_names, n_samples=10, save_path='code/cifar10_clean.png'):
        """
        Affiche une grille d'images propres
        
        Args:
            images: Array d'images (N, 32, 32, 3)
            labels: Array de labels (N,)
            class_names: Liste des noms de classes
            n_samples: Nombre d'images à afficher
            save_path: Chemin pour sauvegarder l'image
        """
        print("\nAffichage d'images propres...")
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()
        
        for i in range(n_samples):
            img = images[i]
            axes[i].imshow(img)
            axes[i].set_title(f'{class_names[labels[i]]}', fontsize=12, fontweight='bold')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.suptitle('Images Propres CIFAR-10', fontsize=16, fontweight='bold', y=1.02)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Images propres sauvegardées: '{save_path}'")
        plt.show()
    
    @staticmethod
    def compare_clean_and_noisy(clean_images, noisy_images, labels, class_names, 
                                noise_type='', n_samples=5):
        """
        Compare côte à côte les images propres et bruitées
        
        Args:
            clean_images: Images propres
            noisy_images: Images bruitées
            labels: Labels des images
            class_names: Noms des classes
            noise_type: Type de bruit appliqué (pour affichage)
            n_samples: Nombre d'échantillons à comparer
        """
        print("\nComparaison images propres vs bruitées...")
        
        # Dictionnaire pour les noms d'affichage
        noise_display_names = {
            'gaussian': 'Gaussien',
            'salt_pepper': 'Sel & Poivre',
            'speckle': 'Speckle',
            'poisson': 'Poisson',
            'uniform': 'Uniforme',
            'mixed': 'Mixte'
        }
        
        noise_label = noise_display_names.get(noise_type, noise_type.capitalize() if noise_type else 'Bruitée')
        
        fig, axes = plt.subplots(2, n_samples, figsize=(3*n_samples, 6))
        
        for i in range(n_samples):
            # Image propre
            axes[0, i].imshow(clean_images[i].astype(np.uint8))
            axes[0, i].set_title(f'Propre\n{class_names[labels[i]]}', 
                                fontsize=10, fontweight='bold')
            axes[0, i].axis('off')
            
            # Image bruitée
            noisy_img = noisy_images[i]
            if noisy_img.max() <= 1.0:
                noisy_img = (noisy_img * 255).astype(np.uint8)
            else:
                noisy_img = noisy_img.astype(np.uint8)
                
            axes[1, i].imshow(noisy_img)
            axes[1, i].set_title(f'{noise_label}\n{class_names[labels[i]]}', 
                                fontsize=10, fontweight='bold', color='red')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        # Nom de fichier basé sur le type de bruit
        filename = f'cifar10_comparison_{noise_type}.png' if noise_type else 'cifar10_comparison.png'
        save_path = f'code/{filename}'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Comparaison sauvegardée: '{save_path}'")
        plt.show()
    
    @staticmethod
    def demonstrate_noise_types(image, class_name, noise_configs=None, save_path='code/noise_types_demo.png'):
        """
        Démontre tous les types de bruit sur une seule image
        
        Args:
            image: Une seule image (32, 32, 3)
            class_name: Nom de la classe de l'image
            noise_configs: Liste de configurations de bruit (optionnel)
            save_path: Chemin pour sauvegarder l'image
        """
        from noise_generator import NoiseGenerator
        
        print(f"\nDémonstration des types de bruit sur une image '{class_name}'...")
        
        if noise_configs is None:
            noise_configs = [
                ('Original', None, {}),
                ('Gaussien', 'gaussian', {'std': 25}),
                ('Sel & Poivre', 'salt_pepper', {'salt_prob': 0.02, 'pepper_prob': 0.02}),
                ('Speckle', 'speckle', {'std': 0.15}),
                ('Poisson', 'poisson', {}),
                ('Mixte', 'mixed', {'gaussian_std': 15, 'salt_prob': 0.01, 'pepper_prob': 0.01})
            ]
        
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.ravel()
        
        for idx, (title, noise_type, params) in enumerate(noise_configs):
            if noise_type is None:
                # Image originale
                img_display = image
            else:
                # Ajouter le bruit
                if noise_type == 'gaussian':
                    noisy = NoiseGenerator.add_gaussian_noise(image.copy(), **params)
                elif noise_type == 'salt_pepper':
                    noisy = NoiseGenerator.add_salt_and_pepper_noise(image.copy(), **params)
                elif noise_type == 'speckle':
                    noisy = NoiseGenerator.add_speckle_noise(image.copy(), **params)
                elif noise_type == 'poisson':
                    noisy = NoiseGenerator.add_poisson_noise(image.copy())
                elif noise_type == 'mixed':
                    noisy = NoiseGenerator.add_mixed_noise(image.copy(), **params)
                
                img_display = noisy
            
            # Afficher
            axes[idx].imshow(img_display.astype(np.uint8))
            axes[idx].set_title(title, fontsize=12, fontweight='bold')
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.suptitle(f'Types de Bruit - {class_name}', fontsize=14, fontweight='bold', y=1.02)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Démonstration sauvegardée: '{save_path}'")
        plt.show()
    
    @staticmethod
    def compare_denoising_results(clean_images, noisy_images, denoised_images, 
                                  labels, class_names, n_samples=5, save_path='code/denoising_results.png'):
        """
        Compare images propres, bruitées et débruitées côte à côte
        
        Args:
            clean_images: Images propres originales
            noisy_images: Images bruitées
            denoised_images: Images débruitées par le modèle
            labels: Labels des images
            class_names: Noms des classes
            n_samples: Nombre d'échantillons à comparer
            save_path: Chemin pour sauvegarder l'image
        """
        print("\nComparaison: Propre / Bruitée / Débruitée...")
        
        fig, axes = plt.subplots(3, n_samples, figsize=(3*n_samples, 9))
        
        for i in range(n_samples):
            # Image propre
            axes[0, i].imshow(clean_images[i].astype(np.uint8))
            axes[0, i].set_title(f'Propre\n{class_names[labels[i]]}', 
                                fontsize=10, fontweight='bold', color='green')
            axes[0, i].axis('off')
            
            # Image bruitée
            noisy_img = noisy_images[i]
            if noisy_img.max() <= 1.0:
                noisy_img = (noisy_img * 255).astype(np.uint8)
            else:
                noisy_img = noisy_img.astype(np.uint8)
            
            axes[1, i].imshow(noisy_img)
            axes[1, i].set_title('Bruitée', fontsize=10, fontweight='bold', color='red')
            axes[1, i].axis('off')
            
            # Image débruitée
            denoised_img = denoised_images[i]
            if denoised_img.max() <= 1.0:
                denoised_img = (denoised_img * 255).astype(np.uint8)
            else:
                denoised_img = denoised_img.astype(np.uint8)
            
            axes[2, i].imshow(denoised_img)
            axes[2, i].set_title('Débruitée', fontsize=10, fontweight='bold', color='blue')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.suptitle('Résultats du Débruitage', fontsize=14, fontweight='bold', y=1.01)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Résultats sauvegardés: '{save_path}'")
        plt.show()
    
    


if __name__ == "__main__":
    print("Classe ImageVisualizer prête à l'emploi")
    print("\nMéthodes disponibles:")
    print("  - visualize_clean_images()")
    print("  - compare_clean_and_noisy()")
    print("  - demonstrate_noise_types()")
    print("  - compare_denoising_results()")
