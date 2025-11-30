"""
Entraînement du U-Net avec les 3 types de bruit mélangés aléatoirement
(comme le VAE beta0 qui s'entraîne sur plusieurs types de bruit)
UN SEUL MODÈLE pour les 3 bruits
Support CIFAR-10 (32x32) et STL-10 (96x96 découpé en patches 32x32 avec superposition)
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

from unet_model import UNet
from utils import add_noise_to_images


class PatchDataset(Dataset):
    """
    Dataset qui découpe les images en patches 32x32 avec superposition
    Pour STL-10 (96x96) -> avec stride=16, on obtient 25 patches qui se superposent
    """
    def __init__(self, base_dataset, patch_size=32, stride=16):
        self.base_dataset = base_dataset
        self.patch_size = patch_size
        self.stride = stride
        
        # Calculer le nombre de patches par image
        sample_img, _ = base_dataset[0]
        _, h, w = sample_img.shape
        
        self.n_patches_h = (h - patch_size) // stride + 1
        self.n_patches_w = (w - patch_size) // stride + 1
        self.patches_per_image = self.n_patches_h * self.n_patches_w
        
        print(f"  → Image size: {h}x{w}")
        print(f"  → Patch size: {patch_size}x{patch_size}, stride: {stride}")
        print(f"  → Patches per image: {self.patches_per_image} ({self.n_patches_h}x{self.n_patches_w})")
        print(f"  → Overlap: {patch_size - stride} pixels")
        print(f"  → Total patches: {len(base_dataset) * self.patches_per_image}")
    
    def __len__(self):
        return len(self.base_dataset) * self.patches_per_image
    
    def __getitem__(self, idx):
        # Trouver l'image et le patch correspondants
        img_idx = idx // self.patches_per_image
        patch_idx = idx % self.patches_per_image
        
        # Récupérer l'image complète
        img_tensor, label = self.base_dataset[img_idx]
        
        # Calculer les coordonnées du patch
        patch_row = patch_idx // self.n_patches_w
        patch_col = patch_idx % self.n_patches_w
        
        top = patch_row * self.stride
        left = patch_col * self.stride
        
        # Extraire le patch
        patch = img_tensor[:, top:top+self.patch_size, left:left+self.patch_size]
        
        return patch, label


class MultiNoisyDataset(Dataset):
    """Dataset qui applique aléatoirement un des 3 types de bruit"""
    def __init__(self, base_dataset, noise_configs):
        self.base_dataset = base_dataset
        self.noise_configs = noise_configs
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Récupérer l'image originale (déjà en tensor [C, H, W] dans [0, 1])
        img_tensor, label = self.base_dataset[idx]
        
        # Convertir en numpy uint8 [H, W, C] dans [0, 255]
        img_clean = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # Choisir aléatoirement un type de bruit
        noise_config = np.random.choice(self.noise_configs)
        
        # Ajouter du bruit (retourne uint8 [H, W, C])
        img_noisy = add_noise_to_images(
            img_clean[np.newaxis, ...],  # Ajouter batch dimension
            noise_type=noise_config['type'],
            **noise_config['params']
        )[0]  # Enlever batch dimension
        
        # Convertir en tensor [C, H, W] dans [0, 1]
        img_noisy_tensor = torch.FloatTensor(img_noisy).permute(2, 0, 1) / 255.0
        img_clean_tensor = torch.FloatTensor(img_clean).permute(2, 0, 1) / 255.0
        
        return img_noisy_tensor, img_clean_tensor, label


def train_epoch(model, device, dataloader, optimizer):
    """Fonction d'entraînement pour une époque"""
    model.train()
    train_loss = []
    
    for img_noisy, img_clean, _ in dataloader:
        img_noisy = img_noisy.to(device)
        img_clean = img_clean.to(device)
        
        # Forward pass
        output = model(img_noisy)
        
        # Calculer la perte (MSE entre image débruitée et image originale)
        loss = F.mse_loss(output, img_clean)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss.append(loss.item())
    
    return np.mean(train_loss)


def test_epoch(model, device, dataloader):
    """Fonction de validation"""
    model.eval()
    val_loss = []
    
    with torch.no_grad():
        for img_noisy, img_clean, _ in dataloader:
            img_noisy = img_noisy.to(device)
            img_clean = img_clean.to(device)
            
            output = model(img_noisy)
            loss = F.mse_loss(output, img_clean)
            val_loss.append(loss.item())
    
    return np.mean(val_loss)


def plot_results_multi_noise(model, base_test_dataset, device, noise_configs, n=10):
    """
    Affiche les résultats pour les 3 types de bruit
    3 blocs de 3 lignes (original, noisy, denoised) × 10 classes
    """
    if hasattr(base_test_dataset, 'targets'):
        targets = np.array(base_test_dataset.targets)
    else:
        targets = np.array(base_test_dataset.labels)
    
    t_idx = {i: np.where(targets == i)[0][0] for i in range(n)}
    
    model.eval()
    
    # 3 blocs (un par type de bruit)
    for noise_idx, noise_config in enumerate(noise_configs):
        fig = plt.figure(figsize=(16, 4.5))
        fig.suptitle(f"{noise_config['name']} - Params: {noise_config['params']}", 
                     fontsize=14, fontweight='bold')
        
        for i in range(n):
            # Récupérer l'image originale
            img_tensor, label = base_test_dataset[t_idx[i]]
            img_clean = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            
            # Ajouter du bruit spécifique
            img_noisy = add_noise_to_images(
                img_clean[np.newaxis, ...],
                noise_type=noise_config['type'],
                **noise_config['params']
            )[0]
            
            # Convertir en tensors
            img_noisy_tensor = torch.FloatTensor(img_noisy).permute(2, 0, 1) / 255.0
            img_clean_tensor = torch.FloatTensor(img_clean).permute(2, 0, 1) / 255.0
            
            # Image originale
            ax = plt.subplot(3, n, i+1)
            plt.imshow(img_clean_tensor.permute(1, 2, 0).numpy())
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == n//2:
                ax.set_title('Original', fontsize=10)
            
            # Image bruitée
            ax = plt.subplot(3, n, i + 1 + n)
            plt.imshow(img_noisy_tensor.permute(1, 2, 0).numpy())
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == n//2:
                ax.set_title('Noisy', fontsize=10)
            
            # Image débruitée
            ax = plt.subplot(3, n, i + 1 + n + n)
            with torch.no_grad():
                img_noisy_batch = img_noisy_tensor.unsqueeze(0).to(device)
                rec_img = model(img_noisy_batch)
            plt.imshow(rec_img.cpu().squeeze().permute(1, 2, 0).numpy())
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == n//2:
                ax.set_title('Denoised', fontsize=10)
        
        plt.tight_layout()
        plt.show()


def load_dataset(dataset_name, data_dir, use_patches=True, patch_stride=16):
    """
    Charge le dataset spécifié
    
    Args:
        dataset_name: 'cifar10' ou 'stl10'
        data_dir: Répertoire des données
        use_patches: Si True, découpe STL-10 en patches 32x32
        patch_stride: Stride pour le découpage (16 = superposition de 50%)
    
    Returns:
        train_dataset, test_dataset, image_size, num_classes
    """
    if dataset_name == 'cifar10':
        print("\n✓ Chargement de CIFAR-10 (32x32)...")
        train_dataset = torchvision.datasets.CIFAR10(
            data_dir, train=True, download=True,
            transform=transforms.ToTensor()
        )
        test_dataset = torchvision.datasets.CIFAR10(
            data_dir, train=False, download=True,
            transform=transforms.ToTensor()
        )
        return train_dataset, test_dataset, 32, 10
    
    elif dataset_name == 'stl10':
        print("\n✓ Chargement de STL-10 (96x96)...")
        train_dataset = torchvision.datasets.STL10(
            data_dir, split='train', download=True,
            transform=transforms.ToTensor()
        )
        test_dataset = torchvision.datasets.STL10(
            data_dir, split='test', download=True,
            transform=transforms.ToTensor()
        )
        
        if use_patches:
            print(f"✓ Découpage en patches 32x32 avec superposition...")
            train_dataset = PatchDataset(train_dataset, patch_size=32, stride=patch_stride)
            test_dataset = PatchDataset(test_dataset, patch_size=32, stride=patch_stride)
            return train_dataset, test_dataset, 32, 10
        else:
            print("⚠️  Mode sans patches (redimensionnement à 64x64)")
            # Fallback: redimensionner à 64x64
            transform_stl = transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor()
            ])
            train_dataset = torchvision.datasets.STL10(
                data_dir, split='train', download=True,
                transform=transform_stl
            )
            test_dataset = torchvision.datasets.STL10(
                data_dir, split='test', download=True,
                transform=transform_stl
            )
            return train_dataset, test_dataset, 64, 10
    
    else:
        raise ValueError(f"Dataset non supporté: {dataset_name}. Utilisez 'cifar10' ou 'stl10'")


if __name__ == "__main__":
    # Parsing des arguments
    parser = argparse.ArgumentParser(description='Entraînement U-Net Multi-Noise')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                        choices=['cifar10', 'stl10'],
                        help='Dataset à utiliser (cifar10 ou stl10)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Nombre d\'époques (défaut: 30)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Taille du batch (défaut: 64)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (défaut: 0.001)')
    parser.add_argument('--no-patches', action='store_true',
                        help='Désactiver le découpage en patches pour STL-10 (redimensionne à 64x64 à la place)')
    parser.add_argument('--patch-stride', type=int, default=16,
                        help='Stride pour le découpage en patches (défaut: 16 = overlap de 50%%)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ENTRAÎNEMENT U-NET AVEC MULTI-BRUIT MÉLANGÉ")
    print("UN SEUL MODÈLE pour les 3 types de bruit")
    print("=" * 80)
    
    print(f"\n📊 Configuration:")
    print(f"  - Dataset:    {args.dataset.upper()}")
    print(f"  - Epochs:     {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - LR:         {args.lr}")
    if args.dataset == 'stl10':
        if args.no_patches:
            print(f"  - Patches:    Désactivé (resize 64x64)")
        else:
            overlap_percent = (32 - args.patch_stride) / 32 * 100
            print(f"  - Patches:    32x32 avec stride={args.patch_stride} (overlap {overlap_percent:.0f}%)")
    
    # Configuration
    data_dir = './code/dataset'
    
    # DÉFINIR LES 3 TYPES DE BRUIT (comme dans test_beta_zero.py)
    noise_configs = [
        {
            'name': 'Gaussien',
            'type': 'gaussian',
            'params': {'std': 25}
        },
        {
            'name': 'Salt & Pepper',
            'type': 'salt_pepper',
            'params': {'salt_prob': 0.02, 'pepper_prob': 0.02}
        },
        {
            'name': 'Mixte',
            'type': 'mixed',
            'params': {'gaussian_std': 20, 'salt_prob': 0.01, 'pepper_prob': 0.01}
        }
    ]
    
    print("\n✓ Types de bruit utilisés:")
    for config in noise_configs:
        print(f"  - {config['name']}: {config['params']}")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Chargement du dataset
    base_train_dataset, base_test_dataset, img_size, num_classes = load_dataset(
        args.dataset, data_dir, use_patches=not args.no_patches, patch_stride=args.patch_stride
    )
    
    # Créer les datasets avec bruit aléatoire
    print("\nApplication des bruits aléatoires...")
    train_dataset = MultiNoisyDataset(base_train_dataset, noise_configs)
    test_dataset = MultiNoisyDataset(base_test_dataset, noise_configs)
    
    # Split train/validation
    m = len(train_dataset)
    train_data, val_data = random_split(train_dataset, [int(m-m*0.2), int(m*0.2)])
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=2)
    
    print(f"✓ Train: {len(train_data)} patches")
    print(f"✓ Val:   {len(val_data)} patches")
    print(f"✓ Test:  {len(test_dataset)} patches")
    print(f"✓ Patch size: {img_size}x{img_size}")
    
    # Initialisation du modèle U-Net
    print("\nInitialisation du modèle U-Net...")
    model = UNet(n_channels=3, n_classes=3, base_features=64)
    model.to(device)
    
    # Compter les paramètres
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Nombre de paramètres: {total_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Entraînement
    print("\n" + "=" * 80)
    print("DÉBUT DE L'ENTRAÎNEMENT")
    print("=" * 80)
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    model_path = f'./code/unet_denoising_{args.dataset}_multinoise.pth'
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, device, train_loader, optimizer)
        val_loss = test_epoch(model, device, valid_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Sauvegarder si meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            marker = " ✓ (best)"
        else:
            marker = ""
        
        print(f'EPOCH {epoch + 1:2d}/{args.epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}{marker}')
    
    print("\n" + "=" * 80)
    print("ENTRAÎNEMENT TERMINÉ")
    print("=" * 80)
    print(f"✓ Modèle sauvegardé: {model_path}")
    print(f"✓ Meilleure val loss: {best_val_loss:.4f}")
    
    # Charger le meilleur modèle pour l'évaluation
    model.load_state_dict(torch.load(model_path))
    
    # Affichage des résultats pour les 3 types de bruit
    print("\nAffichage des résultats sur le test set...")
    print("(3 graphiques, un par type de bruit)")
    
    # Pour la visualisation, on utilise le dataset de base (sans patches)
    if args.dataset == 'stl10' and not args.no_patches:
        print("⚠️  Visualisation sur images complètes 96x96 (non découpées)")
        vis_test_dataset = torchvision.datasets.STL10(
            data_dir, split='test', download=False,
            transform=transforms.ToTensor()
        )
    else:
        vis_test_dataset = base_test_dataset.base_dataset if hasattr(base_test_dataset, 'base_dataset') else base_test_dataset
    
    plot_results_multi_noise(model, vis_test_dataset, device, noise_configs, n=num_classes)
    
    # Plot de l'historique
    print("\nAffichage de l'historique d'entraînement...")
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss', marker='o', markersize=3)
    plt.plot(history['val_loss'], label='Val Loss', marker='s', markersize=3)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.legend(fontsize=11)
    title_suffix = f" - Patches 32x32 (stride={args.patch_stride})" if args.dataset == 'stl10' and not args.no_patches else ""
    plt.title(f'U-Net Training History - Multi-Noise ({args.dataset.upper()}){title_suffix}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Sauvegarder le graphique
    history_path = f'./code/unet_history_{args.dataset}_multinoise.png'
    plt.savefig(history_path, dpi=300, bbox_inches='tight')
    print(f"✓ Historique sauvegardé: {history_path}")
    
    plt.show()
    
    print("\n" + "=" * 80)
    print("TOUT EST TERMINÉ ✓")
    print("=" * 80)
    print(f"\nPour évaluer ce modèle:")
    print(f"  python unet_evaluate.py --dataset {args.dataset}")