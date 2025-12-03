"""
Utilitaires pour le traitement d'images par patches
Utilisé pour traiter des images de grande taille avec des modèles entraînés sur patches 32x32
"""
import torch
import numpy as np


def denoise_with_patches(model, img_tensor, device, patch_size=32, stride=16):
    """
    Débruite une image en la découpant en patches, puis recompose l'image complète.
    
    Utilisé pour traiter des images de grande taille (ex: STL-10 96x96) avec des modèles 
    U-Net entraînés sur des patches 32x32. Les patches se chevauchent (stride < patch_size)
    pour éviter les artefacts aux bords et améliorer la qualité.
    
    Args:
        model: Modèle U-Net chargé et en mode eval
        img_tensor: Image à débruiter (C, H, W) en format torch.Tensor [0, 1]
        device: Device (cuda/cpu)
        patch_size: Taille des patches (défaut: 32)
        stride: Décalage entre patches (défaut: 16, 50% de chevauchement)
    
    Returns:
        torch.Tensor: Image débruitée (C, H, W) [0, 1]
    
    Example:
        >>> model = UNet(n_channels=3, n_classes=3, base_features=64)
        >>> model.load_state_dict(torch.load('model.pth'))
        >>> model.eval()
        >>> img = torch.randn(3, 96, 96)  # Image 96x96
        >>> denoised = denoise_with_patches(model, img, device, patch_size=32, stride=16)
        >>> print(denoised.shape)  # torch.Size([3, 96, 96])
    """
    _, h, w = img_tensor.shape
    
    # Si l'image est déjà de la taille d'un patch, traitement direct
    if h == patch_size and w == patch_size:
        with torch.no_grad():
            img_batch = img_tensor.unsqueeze(0).to(device)
            denoised = model(img_batch).cpu().squeeze()
        return denoised
    
    # Vérifier que l'image est assez grande pour être découpée
    if h < patch_size or w < patch_size:
        raise ValueError(f"Image trop petite ({h}x{w}) pour des patches {patch_size}x{patch_size}")
    
    # Créer une image de sortie et une carte de poids pour la moyenne pondérée
    denoised_img = torch.zeros_like(img_tensor)
    weight_map = torch.zeros((h, w))
    
    # Extraire et débruiter chaque patch avec chevauchement
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            # Extraire le patch
            patch = img_tensor[:, i:i+patch_size, j:j+patch_size]
            
            # Débruiter le patch
            with torch.no_grad():
                patch_batch = patch.unsqueeze(0).to(device)
                denoised_patch = model(patch_batch).cpu().squeeze()
            
            # Ajouter au résultat (accumulation pour moyenne pondérée)
            denoised_img[:, i:i+patch_size, j:j+patch_size] += denoised_patch
            weight_map[i:i+patch_size, j:j+patch_size] += 1
    
    # Normaliser par le nombre de chevauchements (moyenne pondérée)
    weight_map = weight_map.unsqueeze(0)  # (1, H, W) pour broadcasting
    denoised_img = denoised_img / weight_map
    
    return denoised_img


def denoise_batch_with_patches(model, img_batch, device, patch_size=32, stride=16):
    """
    Débruite un batch d'images en utilisant le système de patches.
    
    Args:
        model: Modèle U-Net chargé et en mode eval
        img_batch: Batch d'images (N, C, H, W) en format torch.Tensor [0, 1]
        device: Device (cuda/cpu)
        patch_size: Taille des patches (défaut: 32)
        stride: Décalage entre patches (défaut: 16)
    
    Returns:
        torch.Tensor: Batch d'images débruitées (N, C, H, W) [0, 1]
    
    Example:
        >>> model = UNet(n_channels=3, n_classes=3, base_features=64)
        >>> model.load_state_dict(torch.load('model.pth'))
        >>> model.eval()
        >>> imgs = torch.randn(4, 3, 96, 96)  # Batch de 4 images 96x96
        >>> denoised = denoise_batch_with_patches(model, imgs, device)
        >>> print(denoised.shape)  # torch.Size([4, 3, 96, 96])
    """
    n, c, h, w = img_batch.shape
    denoised_batch = torch.zeros_like(img_batch)
    
    for i in range(n):
        denoised_batch[i] = denoise_with_patches(
            model, img_batch[i], device, patch_size, stride
        )
    
    return denoised_batch


def extract_patches(img_tensor, patch_size=32, stride=16):
    """
    Extrait tous les patches d'une image avec chevauchement.
    
    Args:
        img_tensor: Image (C, H, W) en format torch.Tensor
        patch_size: Taille des patches (défaut: 32)
        stride: Décalage entre patches (défaut: 16)
    
    Returns:
        torch.Tensor: Patches extraits (N_patches, C, patch_size, patch_size)
        list: Positions des patches [(i, j), ...]
    
    Example:
        >>> img = torch.randn(3, 96, 96)
        >>> patches, positions = extract_patches(img, patch_size=32, stride=16)
        >>> print(patches.shape)  # torch.Size([25, 3, 32, 32]) pour 96x96 avec stride=16
        >>> print(len(positions))  # 25
    """
    c, h, w = img_tensor.shape
    patches = []
    positions = []
    
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = img_tensor[:, i:i+patch_size, j:j+patch_size]
            patches.append(patch)
            positions.append((i, j))
    
    patches = torch.stack(patches)
    return patches, positions


def reconstruct_from_patches(patches, positions, img_shape, patch_size=32):
    """
    Reconstruit une image à partir de patches avec moyenne pondérée des chevauchements.
    
    Args:
        patches: Patches (N_patches, C, patch_size, patch_size)
        positions: Positions des patches [(i, j), ...]
        img_shape: Forme de l'image de sortie (C, H, W)
        patch_size: Taille des patches (défaut: 32)
    
    Returns:
        torch.Tensor: Image reconstruite (C, H, W)
    
    Example:
        >>> patches = torch.randn(25, 3, 32, 32)
        >>> positions = [(i, j) for i in range(0, 65, 16) for j in range(0, 65, 16)]
        >>> img = reconstruct_from_patches(patches, positions, (3, 96, 96), patch_size=32)
        >>> print(img.shape)  # torch.Size([3, 96, 96])
    """
    c, h, w = img_shape
    reconstructed_img = torch.zeros(img_shape)
    weight_map = torch.zeros((h, w))
    
    for patch, (i, j) in zip(patches, positions):
        reconstructed_img[:, i:i+patch_size, j:j+patch_size] += patch
        weight_map[i:i+patch_size, j:j+patch_size] += 1
    
    # Normaliser par le nombre de chevauchements
    weight_map = weight_map.unsqueeze(0)
    reconstructed_img = reconstructed_img / weight_map
    
    return reconstructed_img


def calculate_num_patches(img_height, img_width, patch_size=32, stride=16):
    """
    Calcule le nombre de patches qui seront extraits d'une image.
    
    Args:
        img_height: Hauteur de l'image
        img_width: Largeur de l'image
        patch_size: Taille des patches (défaut: 32)
        stride: Décalage entre patches (défaut: 16)
    
    Returns:
        tuple: (n_patches_h, n_patches_w, total_patches)
    
    Example:
        >>> n_h, n_w, total = calculate_num_patches(96, 96, patch_size=32, stride=16)
        >>> print(f"Patches: {n_h}x{n_w} = {total}")  # Patches: 5x5 = 25
    """
    n_patches_h = (img_height - patch_size) // stride + 1
    n_patches_w = (img_width - patch_size) // stride + 1
    total_patches = n_patches_h * n_patches_w
    
    return n_patches_h, n_patches_w, total_patches


def visualize_patch_grid(img_height, img_width, patch_size=32, stride=16):
    """
    Affiche des informations sur la grille de patches.
    
    Args:
        img_height: Hauteur de l'image
        img_width: Largeur de l'image
        patch_size: Taille des patches (défaut: 32)
        stride: Décalage entre patches (défaut: 16)
    
    Example:
        >>> visualize_patch_grid(96, 96, patch_size=32, stride=16)
        Image: 96x96
        Patch size: 32x32
        Stride: 16
        Overlap: 50.0%
        Number of patches: 5x5 = 25
        Coverage: Each pixel is covered by ~4.0 patches (average)
    """
    n_h, n_w, total = calculate_num_patches(img_height, img_width, patch_size, stride)
    overlap_percent = (1 - stride / patch_size) * 100
    
    # Calculer le nombre moyen de fois qu'un pixel est couvert
    weight_map = np.zeros((img_height, img_width))
    for i in range(0, img_height - patch_size + 1, stride):
        for j in range(0, img_width - patch_size + 1, stride):
            weight_map[i:i+patch_size, j:j+patch_size] += 1
    avg_coverage = weight_map.mean()
    
    print(f"Image: {img_height}x{img_width}")
    print(f"Patch size: {patch_size}x{patch_size}")
    print(f"Stride: {stride}")
    print(f"Overlap: {overlap_percent:.1f}%")
    print(f"Number of patches: {n_h}x{n_w} = {total}")
    print(f"Coverage: Each pixel is covered by ~{avg_coverage:.1f} patches (average)")
    print(f"Min coverage: {weight_map.min():.0f}, Max coverage: {weight_map.max():.0f}")


if __name__ == "__main__":
    print("=" * 80)
    print("PATCH UTILITIES - Tests et exemples")
    print("=" * 80)
    
def denoise_with_patches_edge(model, img_tensor, device, patch_size=32, stride=16):
    """
    Débruite une image en utilisant des patches, en gérant les bords pour éviter les artefacts si l'image n'est pas un multiple de patch_size.
    Args:
        model: Modèle U-Net chargé et en mode eval
        img_tensor: Image (C, H, W) torch.Tensor [0, 1]
        device: Device (cuda/cpu)
        patch_size: Taille des patches (défaut: 32)
        stride: Décalage entre patches (défaut: 16)
    Returns:
        torch.Tensor: Image débruitée (C, H, W) [0, 1]
    """
    import torch
    c, h, w = img_tensor.shape
    denoised_img = torch.zeros((c, h, w), dtype=img_tensor.dtype)
    weight_map = torch.zeros((h, w), dtype=img_tensor.dtype)

    # Calculer les positions des patches en gérant les bords
    i_list = list(range(0, h - patch_size + 1, stride))
    j_list = list(range(0, w - patch_size + 1, stride))
    if (h - patch_size) % stride != 0:
        i_list.append(h - patch_size)
    if (w - patch_size) % stride != 0:
        j_list.append(w - patch_size)

    positions = [(i, j) for i in i_list for j in j_list]

    patches = []
    for (i, j) in positions:
        patch = img_tensor[:, i:i+patch_size, j:j+patch_size]
        patches.append(patch)

    patches = torch.stack(patches)
    patches = patches.to(device)

    with torch.no_grad():
        denoised_patches = model(patches).cpu()

    # Reconstruction avec moyenne pondérée
    for idx, (i, j) in enumerate(positions):
        denoised_img[:, i:i+patch_size, j:j+patch_size] += denoised_patches[idx]
        weight_map[i:i+patch_size, j:j+patch_size] += 1

    weight_map = weight_map.unsqueeze(0)
    denoised_img = denoised_img / weight_map
    return denoised_img
    # Exemple 1: Information sur la grille de patches pour STL-10
    print("\n📊 Configuration pour STL-10 (96x96):")
    visualize_patch_grid(96, 96, patch_size=32, stride=16)
    
    # Exemple 2: Information pour CIFAR-10
    print("\n📊 Configuration pour CIFAR-10 (32x32):")
    visualize_patch_grid(32, 32, patch_size=32, stride=16)
    
    # Exemple 3: Test de débruitage avec patches
    print("\n🧪 Test de débruitage avec patches:")
    print("  Creating dummy model and image...")
    
    # Import nécessaire pour le test
    try:
        from unet_model import UNet
        
        device = torch.device('cpu')
        model = UNet(n_channels=3, n_classes=3, base_features=64)
        model.eval()
        
        # Image de test 96x96
        img_test = torch.rand(3, 96, 96)
        print(f"  Input shape: {img_test.shape}")
        
        # Débruitage avec patches
        denoised = denoise_with_patches(model, img_test, device, patch_size=32, stride=16)
        print(f"  Output shape: {denoised.shape}")
        print("  ✓ Test réussi!")
        
    except ImportError:
        print("  ⚠️  Module unet_model non trouvé, test ignoré")
    
    print("\n" + "=" * 80)
    print("Fonctions disponibles:")
    print("  - denoise_with_patches(model, img_tensor, device, patch_size, stride)")
    print("  - denoise_batch_with_patches(model, img_batch, device, patch_size, stride)")
    print("  - extract_patches(img_tensor, patch_size, stride)")
    print("  - reconstruct_from_patches(patches, positions, img_shape, patch_size)")
    print("  - calculate_num_patches(img_height, img_width, patch_size, stride)")
    print("  - visualize_patch_grid(img_height, img_width, patch_size, stride)")
    print("=" * 80)
