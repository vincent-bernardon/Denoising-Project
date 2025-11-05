import torch
import numpy as np
import matplotlib.pyplot as plt


class NoiseGenerator:
    """
    Classe pour ajouter différents types de bruit aux images
    """
    
    @staticmethod
    def add_gaussian_noise(images, mean=0.0, std=0.1):
        """
        Ajoute du bruit gaussien (bruit blanc)
        
        Args:
            images (torch.Tensor ou np.ndarray): Images à bruiter
            mean (float): Moyenne du bruit gaussien
            std (float): Écart-type du bruit gaussien
            
        Returns:
            Images bruitées
        """
        is_numpy = isinstance(images, np.ndarray)
        
        if is_numpy:
            noise = np.random.normal(mean, std, images.shape)
            noisy_images = images + noise
            noisy_images = np.clip(noisy_images, 0, 255 if images.max() > 1 else 1)
        else:
            noise = torch.randn_like(images) * std + mean
            noisy_images = images + noise
            noisy_images = torch.clamp(noisy_images, 0, 255 if images.max() > 1 else 1)
        
        return noisy_images
    
    @staticmethod
    def add_salt_and_pepper_noise(images, salt_prob=0.01, pepper_prob=0.01):
        """
        Ajoute du bruit sel et poivre (pixels blancs et noirs aléatoires)
        
        Args:
            images (torch.Tensor ou np.ndarray): Images à bruiter
            salt_prob (float): Probabilité de pixel blanc
            pepper_prob (float): Probabilité de pixel noir
            
        Returns:
            Images bruitées
        """
        is_numpy = isinstance(images, np.ndarray)
        
        if is_numpy:
            noisy_images = images.copy()
            max_val = 255 if images.max() > 1 else 1
            
            # Déterminer la forme pour le masque
            if len(images.shape) == 4:  # Batch d'images (N, H, W, C)
                mask_shape = images.shape[:3]  # (N, H, W)
            elif len(images.shape) == 3:  # Image unique (H, W, C)
                mask_shape = images.shape[:2]  # (H, W)
            else:
                raise ValueError(f"Format d'image non supporté: {images.shape}")
            
            # Salt (blanc)
            salt_mask = np.random.random(mask_shape) < salt_prob
            # Étendre le masque pour tous les canaux
            if len(images.shape) == 4:
                salt_mask = salt_mask[..., np.newaxis]  # (N, H, W, 1)
                salt_mask = np.repeat(salt_mask, images.shape[-1], axis=-1)  # (N, H, W, C)
            else:
                salt_mask = salt_mask[..., np.newaxis]  # (H, W, 1)
                salt_mask = np.repeat(salt_mask, images.shape[-1], axis=-1)  # (H, W, C)
            
            noisy_images[salt_mask] = max_val
            
            # Pepper (noir)
            pepper_mask = np.random.random(mask_shape) < pepper_prob
            # Étendre le masque pour tous les canaux
            if len(images.shape) == 4:
                pepper_mask = pepper_mask[..., np.newaxis]
                pepper_mask = np.repeat(pepper_mask, images.shape[-1], axis=-1)
            else:
                pepper_mask = pepper_mask[..., np.newaxis]
                pepper_mask = np.repeat(pepper_mask, images.shape[-1], axis=-1)
            
            noisy_images[pepper_mask] = 0
        else:
            noisy_images = images.clone()
            max_val = 255 if images.max() > 1 else 1
            
            # Déterminer la forme pour le masque
            if len(images.shape) == 4:  # (N, C, H, W)
                mask_shape = (images.shape[0], images.shape[2], images.shape[3])
            elif len(images.shape) == 3:  # (C, H, W)
                mask_shape = images.shape[1:]
            else:
                raise ValueError(f"Format d'image non supporté: {images.shape}")
            
            # Salt
            salt_mask = torch.rand(mask_shape, device=images.device) < salt_prob
            if len(images.shape) == 4:
                salt_mask = salt_mask.unsqueeze(1).expand_as(images)  # (N, C, H, W)
            else:
                salt_mask = salt_mask.unsqueeze(0).expand_as(images)  # (C, H, W)
            noisy_images[salt_mask] = max_val
            
            # Pepper
            pepper_mask = torch.rand(mask_shape, device=images.device) < pepper_prob
            if len(images.shape) == 4:
                pepper_mask = pepper_mask.unsqueeze(1).expand_as(images)
            else:
                pepper_mask = pepper_mask.unsqueeze(0).expand_as(images)
            noisy_images[pepper_mask] = 0
        
        return noisy_images
    
    @staticmethod
    def add_speckle_noise(images, std=0.1):
        """
        Ajoute du bruit speckle (multiplicatif)
        
        Args:
            images (torch.Tensor ou np.ndarray): Images à bruiter
            std (float): Écart-type du bruit
            
        Returns:
            Images bruitées
        """
        is_numpy = isinstance(images, np.ndarray)
        
        if is_numpy:
            noise = np.random.randn(*images.shape) * std
            noisy_images = images + images * noise
            noisy_images = np.clip(noisy_images, 0, 255 if images.max() > 1 else 1)
        else:
            noise = torch.randn_like(images) * std
            noisy_images = images + images * noise
            noisy_images = torch.clamp(noisy_images, 0, 255 if images.max() > 1 else 1)
        
        return noisy_images
    
    @staticmethod
    def add_poisson_noise(images):
        """
        Ajoute du bruit de Poisson (dépendant de l'intensité)
        
        Args:
            images (torch.Tensor ou np.ndarray): Images à bruiter
            
        Returns:
            Images bruitées
        """
        is_numpy = isinstance(images, np.ndarray)
        
        if is_numpy:
            # Normaliser si nécessaire
            if images.max() <= 1:
                images_scaled = images * 255
            else:
                images_scaled = images
            
            noisy_images = np.random.poisson(images_scaled)
            
            if images.max() <= 1:
                noisy_images = noisy_images / 255.0
            
            noisy_images = np.clip(noisy_images, 0, 255 if images.max() > 1 else 1)
        else:
            # PyTorch n'a pas de distribution de Poisson directe, utiliser NumPy
            images_np = images.cpu().numpy()
            noisy_np = NoiseGenerator.add_poisson_noise(images_np)
            noisy_images = torch.from_numpy(noisy_np).to(images.device)
        
        return noisy_images
    
    @staticmethod
    def add_uniform_noise(images, low=-0.1, high=0.1):
        """
        Ajoute du bruit uniforme
        
        Args:
            images (torch.Tensor ou np.ndarray): Images à bruiter
            low (float): Valeur minimale du bruit
            high (float): Valeur maximale du bruit
            
        Returns:
            Images bruitées
        """
        is_numpy = isinstance(images, np.ndarray)
        
        if is_numpy:
            noise = np.random.uniform(low, high, images.shape)
            noisy_images = images + noise
            noisy_images = np.clip(noisy_images, 0, 255 if images.max() > 1 else 1)
        else:
            noise = torch.rand_like(images) * (high - low) + low
            noisy_images = images + noise
            noisy_images = torch.clamp(noisy_images, 0, 255 if images.max() > 1 else 1)
        
        return noisy_images
    
    @staticmethod
    def add_mixed_noise(images, gaussian_std=0.05, salt_prob=0.005, pepper_prob=0.005):
        """
        Combine plusieurs types de bruit (réaliste)
        
        Args:
            images: Images à bruiter
            gaussian_std: Écart-type du bruit gaussien
            salt_prob: Probabilité de sel
            pepper_prob: Probabilité de poivre
            
        Returns:
            Images bruitées avec bruit mixte
        """
        noisy = NoiseGenerator.add_gaussian_noise(images, std=gaussian_std)
        noisy = NoiseGenerator.add_salt_and_pepper_noise(noisy, salt_prob, pepper_prob)
        return noisy
    
    @staticmethod
    def visualize_noise_comparison(clean_image, noise_types=None):
        """
        Visualise l'effet de différents types de bruit
        
        Args:
            clean_image: Image propre (H, W, C) ou (C, H, W)
            noise_types: Liste des types de bruit à comparer
        """
        if noise_types is None:
            noise_types = ['gaussian', 'salt_pepper', 'speckle', 'poisson', 'uniform', 'mixed']
        
        # Vérifier et ajuster le format de l'image
        if isinstance(clean_image, torch.Tensor):
            if len(clean_image.shape) == 3 and clean_image.shape[0] == 3:
                # (C, H, W) -> (H, W, C)
                clean_image = clean_image.permute(1, 2, 0).cpu().numpy()
            else:
                clean_image = clean_image.cpu().numpy()
        
        n_types = len(noise_types) + 1  # +1 pour l'image originale
        fig, axes = plt.subplots(1, n_types, figsize=(4 * n_types, 4))
        
        # Image originale
        axes[0].imshow(clean_image.astype(np.uint8) if clean_image.max() > 1 else clean_image)
        axes[0].set_title('Original', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Images bruitées
        for idx, noise_type in enumerate(noise_types, 1):
            if noise_type == 'gaussian':
                noisy = NoiseGenerator.add_gaussian_noise(clean_image.copy(), std=0.1)
                title = 'Gaussian'
            elif noise_type == 'salt_pepper':
                noisy = NoiseGenerator.add_salt_and_pepper_noise(clean_image.copy(), 0.02, 0.02)
                title = 'Salt & Pepper'
            elif noise_type == 'speckle':
                noisy = NoiseGenerator.add_speckle_noise(clean_image.copy(), std=0.1)
                title = 'Speckle'
            elif noise_type == 'poisson':
                noisy = NoiseGenerator.add_poisson_noise(clean_image.copy())
                title = 'Poisson'
            elif noise_type == 'uniform':
                noisy = NoiseGenerator.add_uniform_noise(clean_image.copy(), -0.15, 0.15)
                title = 'Uniform'
            elif noise_type == 'mixed':
                noisy = NoiseGenerator.add_mixed_noise(clean_image.copy())
                title = 'Mixed'
            
            axes[idx].imshow(noisy.astype(np.uint8) if noisy.max() > 1 else noisy)
            axes[idx].set_title(title, fontsize=12, fontweight='bold')
            axes[idx].axis('off')
        
        plt.tight_layout()
        save_path = 'code/noise_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparaison sauvegardée dans '{save_path}'")
        plt.show()


# Transform personnalisé pour PyTorch DataLoader
class AddNoise:
    """
    Transform PyTorch pour ajouter du bruit pendant l'entraînement
    """
    def __init__(self, noise_type='gaussian', **kwargs):
        """
        Args:
            noise_type (str): Type de bruit ('gaussian', 'salt_pepper', 'speckle', etc.)
            **kwargs: Paramètres spécifiques au type de bruit
        """
        self.noise_type = noise_type
        self.params = kwargs
    
    def __call__(self, tensor):
        """
        Applique le bruit à un tensor PyTorch
        
        Args:
            tensor: Tensor d'image (C, H, W)
            
        Returns:
            Tensor bruité
        """
        if self.noise_type == 'gaussian':
            return NoiseGenerator.add_gaussian_noise(
                tensor, 
                mean=self.params.get('mean', 0.0),
                std=self.params.get('std', 0.1)
            )
        elif self.noise_type == 'salt_pepper':
            return NoiseGenerator.add_salt_and_pepper_noise(
                tensor,
                salt_prob=self.params.get('salt_prob', 0.01),
                pepper_prob=self.params.get('pepper_prob', 0.01)
            )
        elif self.noise_type == 'speckle':
            return NoiseGenerator.add_speckle_noise(
                tensor,
                std=self.params.get('std', 0.1)
            )
        elif self.noise_type == 'mixed':
            return NoiseGenerator.add_mixed_noise(tensor, **self.params)
        else:
            raise ValueError(f"Type de bruit inconnu: {self.noise_type}")


if __name__ == "__main__":
    print("Test du générateur de bruit\n")
    
    # Créer une image de test
    test_image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    
    # Visualiser les différents types de bruit
    NoiseGenerator.visualize_noise_comparison(test_image)
