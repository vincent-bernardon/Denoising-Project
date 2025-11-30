"""
Entraînement U-Net + GAN pour améliorer le débruitage
Le discriminateur apprend à différencier les vraies images des images débruitées
Le générateur (U-Net) apprend à tromper le discriminateur
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
from tqdm import tqdm

from UNET.unet_model import UNet
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


class Discriminator(nn.Module):
    """
    Discriminateur pour distinguer vraies images vs images débruitées
    Sortie: probabilité que l'image soit réelle (non débruitée)
    """
    def __init__(self, in_channels=3, base_features=64):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            # 32x32 -> 16x16
            nn.Conv2d(in_channels, base_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(base_features, base_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 4x4
            nn.Conv2d(base_features * 2, base_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 4x4 -> 2x2
            nn.Conv2d(base_features * 4, base_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 2x2 -> 1x1
            nn.Conv2d(base_features * 8, 1, 2, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x).view(-1, 1)


def add_noise_randomly(images, noise_configs, noise_probability=0.5):
    """
    Ajoute du bruit aléatoirement à un batch d'images
    
    Args:
        images: Tensor [B, C, H, W] dans [0, 1]
        noise_configs: Liste des configurations de bruit
        noise_probability: Probabilité d'ajouter du bruit
    
    Returns:
        noisy_images: Images bruitées [B, C, H, W]
        is_noisy: Masque booléen [B] indiquant si l'image a été bruitée
    """
    batch_size = images.shape[0]
    noisy_images = []
    is_noisy = []
    
    for i in range(batch_size):
        # Décider si on ajoute du bruit
        add_noise = np.random.random() < noise_probability
        is_noisy.append(add_noise)
        
        if add_noise:
            # Convertir en numpy uint8
            img_np = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            # Choisir un type de bruit aléatoire
            noise_config = np.random.choice(noise_configs)
            
            # Ajouter le bruit
            img_noisy = add_noise_to_images(
                img_np[np.newaxis, ...],
                noise_type=noise_config['type'],
                **noise_config['params']
            )[0]
            
            # Reconvertir en tensor sur le même device que l'entrée
            img_tensor = torch.FloatTensor(img_noisy).permute(2, 0, 1) / 255.0
            noisy_images.append(img_tensor)
        else:
            noisy_images.append(images[i].cpu())
    
    noisy_stack = torch.stack(noisy_images)
    is_noisy_tensor = torch.tensor(is_noisy, dtype=torch.bool)
    
    return noisy_stack, is_noisy_tensor


def train_gan_epoch(generator, discriminator, device, dataloader, 
                   g_optimizer, d_optimizer, noise_configs, 
                   lambda_pixel=100, noise_probability=0.5, epoch=0):
    """
    Entraînement GAN pour une époque
    
    Args:
        lambda_pixel: Poids de la perte L1 (reconstruction)
        noise_probability: Probabilité d'ajouter du bruit (0.5 = 50% des images)
    """
    generator.train()
    discriminator.train()
    
    g_losses = []
    d_losses = []
    d_real_accs = []
    d_fake_accs = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1:2d} [Train]', leave=False, ncols=120)
    
    for batch_idx, (images, _) in enumerate(pbar):
        batch_size = images.size(0)
        images = images.to(device)
        
        # Labels avec label smoothing pour stabiliser l'entraînement
        real_labels = torch.ones(batch_size, 1).to(device) * 0.9   # 0.9 au lieu de 1.0
        fake_labels = torch.zeros(batch_size, 1).to(device) + 0.1  # 0.1 au lieu de 0.0
        
        # Ajouter du bruit aléatoirement
        noisy_images, is_noisy = add_noise_randomly(images, noise_configs, noise_probability)
        noisy_images = noisy_images.to(device)
        
        # ============= Entraîner le Discriminateur =============
        d_optimizer.zero_grad()
        
        # Perte sur vraies images (non bruitées)
        real_output = discriminator(images)
        d_real_loss = F.binary_cross_entropy(real_output, real_labels)
        
        # Générer images débruitées (seulement pour celles qui ont du bruit)
        with torch.no_grad():
            denoised_images = generator(noisy_images)
        
        # Pour le discriminateur, on veut qu'il détecte les images débruitées comme fausses
        fake_output = discriminator(denoised_images.detach())
        d_fake_loss = F.binary_cross_entropy(fake_output, fake_labels)
        
        # Total discriminateur
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        d_optimizer.step()
        
        # ============= Entraîner le Générateur (U-Net) =============
        g_optimizer.zero_grad()
        
        # Le générateur veut tromper le discriminateur
        denoised_images = generator(noisy_images)
        fake_output = discriminator(denoised_images)
        
        # Perte adversariale (le générateur veut que ses images soient classées comme réelles)
        g_adv_loss = F.binary_cross_entropy(fake_output, real_labels)
        
        # Perte de reconstruction L1 (pour préserver le contenu)
        g_pixel_loss = F.l1_loss(denoised_images, images)
        
        # Perte totale du générateur
        g_loss = g_adv_loss + lambda_pixel * g_pixel_loss
        g_loss.backward()
        g_optimizer.step()
        
        # Métriques
        d_real_acc = (real_output > 0.5).float().mean().item()
        d_fake_acc = (fake_output < 0.5).float().mean().item()
        
        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        d_real_accs.append(d_real_acc)
        d_fake_accs.append(d_fake_acc)
        
        # Mise à jour de la barre de progression
        pbar.set_postfix({
            'G': f'{g_loss.item():.3f}',
            'D': f'{d_loss.item():.3f}',
            'D_real': f'{d_real_acc:.2f}',
            'D_fake': f'{d_fake_acc:.2f}'
        })
    
    return {
        'g_loss': np.mean(g_losses),
        'd_loss': np.mean(d_losses),
        'd_real_acc': np.mean(d_real_accs),
        'd_fake_acc': np.mean(d_fake_accs)
    }


def validate_gan(generator, discriminator, device, dataloader, noise_configs, 
                noise_probability=0.5, epoch=0):
    """Validation du GAN"""
    generator.eval()
    discriminator.eval()
    
    g_losses = []
    d_losses = []
    d_real_accs = []
    d_fake_accs = []
    psnrs = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1:2d} [Val]', leave=False, ncols=120)
    
    with torch.no_grad():
        for images, _ in pbar:
            batch_size = images.size(0)
            images = images.to(device)
            
            # Labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # Ajouter du bruit
            noisy_images, _ = add_noise_randomly(images, noise_configs, noise_probability)
            noisy_images = noisy_images.to(device)
            
            # Débruiter
            denoised_images = generator(noisy_images)
            
            # Pertes
            real_output = discriminator(images)
            fake_output = discriminator(denoised_images)
            
            d_loss = (F.binary_cross_entropy(real_output, real_labels) + 
                     F.binary_cross_entropy(fake_output, fake_labels)) / 2
            
            g_adv_loss = F.binary_cross_entropy(fake_output, real_labels)
            g_pixel_loss = F.l1_loss(denoised_images, images)
            g_loss = g_adv_loss + 100 * g_pixel_loss
            
            # PSNR
            mse = F.mse_loss(denoised_images, images)
            psnr = 10 * torch.log10(1 / (mse + 1e-10))
            
            # Métriques
            d_real_acc = (real_output > 0.5).float().mean().item()
            d_fake_acc = (fake_output < 0.5).float().mean().item()
            
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            d_real_accs.append(d_real_acc)
            d_fake_accs.append(d_fake_acc)
            psnrs.append(psnr.item())
            
            pbar.set_postfix({
                'G': f'{g_loss.item():.3f}',
                'D': f'{d_loss.item():.3f}',
                'D_real': f'{d_real_acc:.2f}',
                'D_fake': f'{d_fake_acc:.2f}',
                'PSNR': f'{psnr.item():.1f}'
            })
    
    return {
        'g_loss': np.mean(g_losses),
        'd_loss': np.mean(d_losses),
        'd_real_acc': np.mean(d_real_accs),
        'd_fake_acc': np.mean(d_fake_accs),
        'psnr': np.mean(psnrs)
    }


def plot_denoising_results(generator, test_dataset, device, noise_configs, n_samples=10):
    """Affiche des exemples de débruitage"""
    generator.eval()
    
    fig, axes = plt.subplots(3, n_samples, figsize=(20, 6))
    fig.suptitle('U-Net + GAN Denoising Results', fontsize=16, fontweight='bold')
    
    for i in range(n_samples):
        # Récupérer une image
        img_tensor, _ = test_dataset[i]
        img_clean = img_tensor.unsqueeze(0).to(device)
        
        # Ajouter du bruit
        noise_config = noise_configs[i % len(noise_configs)]
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_noisy_np = add_noise_to_images(
            img_np[np.newaxis, ...],
            noise_type=noise_config['type'],
            **noise_config['params']
        )[0]
        img_noisy = torch.FloatTensor(img_noisy_np).permute(2, 0, 1).unsqueeze(0) / 255.0
        img_noisy = img_noisy.to(device)
        
        # Débruiter
        with torch.no_grad():
            img_denoised = generator(img_noisy)
        
        # Afficher
        axes[0, i].imshow(img_tensor.permute(1, 2, 0).numpy())
        axes[0, i].axis('off')
        if i == n_samples // 2:
            axes[0, i].set_title('Original', fontsize=12)
        
        axes[1, i].imshow(img_noisy.cpu().squeeze().permute(1, 2, 0).numpy())
        axes[1, i].axis('off')
        if i == n_samples // 2:
            axes[1, i].set_title('Noisy', fontsize=12)
        
        axes[2, i].imshow(img_denoised.cpu().squeeze().permute(1, 2, 0).numpy())
        axes[2, i].axis('off')
        if i == n_samples // 2:
            axes[2, i].set_title('Denoised (GAN)', fontsize=12)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='U-Net + GAN pour débruitage')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                        choices=['cifar10', 'stl10'],
                        help='Dataset à utiliser (défaut: cifar10)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Nombre d\'époques GAN (défaut: 50)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Taille du batch (défaut: 64)')
    parser.add_argument('--g-lr', type=float, default=0.0002,
                        help='Learning rate générateur (défaut: 0.0002)')
    parser.add_argument('--d-lr', type=float, default=0.0002,
                        help='Learning rate discriminateur (défaut: 0.0002)')
    parser.add_argument('--lambda-pixel', type=float, default=100,
                        help='Poids perte L1 reconstruction (défaut: 100)')
    parser.add_argument('--noise-prob', type=float, default=0.5,
                        help='Probabilité d\'ajouter du bruit (défaut: 0.5)')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Chemin vers modèle U-Net pré-entraîné (.pth)')
    parser.add_argument('--no-patches', action='store_true',
                        help='Désactiver le découpage en patches pour STL-10 (redimensionne à 32x32 à la place)')
    parser.add_argument('--patch-stride', type=int, default=16,
                        help='Stride pour le découpage en patches STL-10 (défaut: 16 = overlap de 50%%)')
    
    args = parser.parse_args()
    
    # Générer le nom du fichier de sortie basé sur le modèle pré-entraîné
    if args.pretrained:
        import os
        base_name = os.path.basename(args.pretrained)
        dir_name = os.path.dirname(args.pretrained)
        # Insérer 'gan_' avant le nom du fichier
        output_model_name = f"gan_{base_name}"
        output_model_path = os.path.join(dir_name, output_model_name)
    else:
        output_model_path = f'./code/unet_gan_{args.dataset}.pth'
    
    print("=" * 80)
    print("U-NET + GAN POUR AMÉLIORER LE DÉBRUITAGE")
    print("=" * 80)
    
    print(f"\n📊 Configuration:")
    print(f"  - Dataset:        {args.dataset.upper()}")
    print(f"  - Epochs:         {args.epochs}")
    print(f"  - Batch size:     {args.batch_size}")
    print(f"  - G LR:           {args.g_lr}")
    print(f"  - D LR:           {args.d_lr}")
    print(f"  - Lambda pixel:   {args.lambda_pixel}")
    print(f"  - Noise prob:     {args.noise_prob * 100:.0f}%")
    if args.dataset == 'stl10':
        if args.no_patches:
            print(f"  - Patches:        Désactivé (resize 32x32)")
        else:
            overlap_percent = (32 - args.patch_stride) / 32 * 100
            print(f"  - Patches:        32x32 avec stride={args.patch_stride} (overlap {overlap_percent:.0f}%)")
    print(f"  - Output model:   {output_model_path}")
    
    # Configuration des bruits
    noise_configs = [
        {'name': 'Gaussien', 'type': 'gaussian', 'params': {'std': 25}},
        {'name': 'Salt & Pepper', 'type': 'salt_pepper', 
         'params': {'salt_prob': 0.02, 'pepper_prob': 0.02}},
        {'name': 'Mixte', 'type': 'mixed', 
         'params': {'gaussian_std': 20, 'salt_prob': 0.01, 'pepper_prob': 0.01}}
    ]
    
    print("\n✓ Types de bruit utilisés:")
    for config in noise_configs:
        print(f"  - {config['name']}: {config['params']}")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Device: {device}")
    
    # Chargement dataset
    print("\n📦 Chargement du dataset...")
    data_dir = './dataset'
    if args.dataset == 'cifar10':
        print("✓ Chargement de CIFAR-10 (32x32)...")
        train_dataset = torchvision.datasets.CIFAR10(
            data_dir, train=True, download=True, transform=transforms.ToTensor()
        )
        test_dataset = torchvision.datasets.CIFAR10(
            data_dir, train=False, download=True, transform=transforms.ToTensor()
        )
    else:
        print("✓ Chargement de STL-10 (96x96)...")
        train_dataset = torchvision.datasets.STL10(
            data_dir, split='train', download=True, transform=transforms.ToTensor()
        )
        test_dataset = torchvision.datasets.STL10(
            data_dir, split='test', download=True, transform=transforms.ToTensor()
        )
        
        if not args.no_patches:
            # Découper en patches 32x32 avec superposition
            print(f"✓ Découpage en patches 32x32 avec superposition...")
            train_dataset = PatchDataset(train_dataset, patch_size=32, stride=args.patch_stride)
            test_dataset = PatchDataset(test_dataset, patch_size=32, stride=args.patch_stride)
        else:
            # Redimensionner à 32x32
            print("⚠️  Mode sans patches: redimensionnement à 32x32")
            transform_stl = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor()
            ])
            train_dataset = torchvision.datasets.STL10(
                data_dir, split='train', download=True, transform=transform_stl
            )
            test_dataset = torchvision.datasets.STL10(
                data_dir, split='test', download=True, transform=transform_stl
            )
    
    # Split train/val
    m = len(train_dataset)
    train_data, val_data = random_split(train_dataset, [int(m*0.8), int(m*0.2)])
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, 
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=2)
    
    unit = "patches" if args.dataset == 'stl10' and not args.no_patches else "images"
    print(f"✓ Train: {len(train_data):,} {unit}")
    print(f"✓ Val:   {len(val_data):,} {unit}")
    print(f"✓ Test:  {len(test_dataset):,} {unit}")
    
    # Initialisation des modèles
    print("\n" + "=" * 80)
    print("INITIALISATION DES MODÈLES")
    print("=" * 80)
    
    generator = UNet(n_channels=3, n_classes=3, base_features=64).to(device)
    discriminator = Discriminator(in_channels=3, base_features=64).to(device)
    
    # Charger U-Net pré-entraîné si spécifié
    if args.pretrained:
        print(f"\n✓ Chargement U-Net pré-entraîné: {args.pretrained}")
        generator.load_state_dict(torch.load(args.pretrained, map_location=device))
        print("  Le générateur part d'un modèle déjà entraîné !")
    else:
        print("\n⚠️  Pas de modèle pré-entraîné, démarrage from scratch")
        print("  Recommandation: utilisez --pretrained pour de meilleurs résultats")
    
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"\n📊 Paramètres:")
    print(f"  - Générateur (U-Net):   {g_params:,}")
    print(f"  - Discriminateur:       {d_params:,}")
    print(f"  - Total:                {g_params + d_params:,}")
    
    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=args.g_lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
    
    # Entraînement
    print("\n" + "=" * 80)
    print("DÉBUT DE L'ENTRAÎNEMENT GAN")
    print("=" * 80)
    print("\nLégende:")
    print("  - G: Perte générateur (adversariale + reconstruction)")
    print("  - D: Perte discriminateur")
    print("  - D_real: Précision discrimination vraies images (target: 0.7-0.8)")
    print("  - D_fake: Précision discrimination images débruitées (target: 0.5-0.6)")
    print("  - PSNR: Peak Signal-to-Noise Ratio (dB, plus haut = meilleur)")
    print("\n⚠️  Équilibre idéal: D_real ≈ 0.75, D_fake ≈ 0.55")
    print("   Si D_real et D_fake > 0.8 → Discriminateur trop fort")
    print("   Si D_real et D_fake < 0.4 → Générateur trop fort\n")
    
    import os
    import numpy as np
    import csv
    history = {
        'train_g_loss': [], 'train_d_loss': [],
        'train_d_real_acc': [], 'train_d_fake_acc': [],
        'val_g_loss': [], 'val_d_loss': [],
        'val_d_real_acc': [], 'val_d_fake_acc': [],
        'val_psnr': []
    }
    # Fichier CSV pour sauvegarde continue
    csv_file = f'./code/unet_gan_history_{args.dataset}.csv'
    # Écrire l'entête si le fichier n'existe pas
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch',
                'train_g_loss', 'train_d_loss', 'train_d_real_acc', 'train_d_fake_acc',
                'val_g_loss', 'val_d_loss', 'val_d_real_acc', 'val_d_fake_acc', 'val_psnr'
            ])
    # Fichier d'historique pour sauvegarde continue
    history_file = f'./code/unet_gan_history_{args.dataset}.npz'
    
    best_psnr = 0
    model_path = output_model_path

    import os
    from glob import glob
    checkpoint_dir = './gan_train'
    best_dir = './gan_best_train'
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    # Padding dynamique pour les noms de fichiers
    epoch_pad = max(3, len(str(args.epochs)))

    # Adaptation auto des hyperparamètres
    d_strong_epochs = 0
    for epoch in range(args.epochs):
        # Train
        train_metrics = train_gan_epoch(
            generator, discriminator, device, train_loader,
            g_optimizer, d_optimizer, noise_configs,
            lambda_pixel=args.lambda_pixel,
            noise_probability=args.noise_prob,
            epoch=epoch
        )

        # Validation
        val_metrics = validate_gan(
            generator, discriminator, device, val_loader,
            noise_configs, noise_probability=args.noise_prob,
            epoch=epoch
        )

        # Historique
        history['train_g_loss'].append(train_metrics['g_loss'])
        history['train_d_loss'].append(train_metrics['d_loss'])
        history['train_d_real_acc'].append(train_metrics['d_real_acc'])
        history['train_d_fake_acc'].append(train_metrics['d_fake_acc'])
        history['val_g_loss'].append(val_metrics['g_loss'])
        history['val_d_loss'].append(val_metrics['d_loss'])
        history['val_d_real_acc'].append(val_metrics['d_real_acc'])
        history['val_d_fake_acc'].append(val_metrics['d_fake_acc'])
        history['val_psnr'].append(val_metrics['psnr'])

        # Sauvegarde CSV à chaque époque (append)
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                train_metrics['g_loss'], train_metrics['d_loss'],
                train_metrics['d_real_acc'], train_metrics['d_fake_acc'],
                val_metrics['g_loss'], val_metrics['d_loss'],
                val_metrics['d_real_acc'], val_metrics['d_fake_acc'],
                val_metrics['psnr']
            ])

        # Sauvegarde continue de l'historique à chaque époque
        try:
            np.savez(history_file,
                train_g_loss=np.array(history['train_g_loss']),
                train_d_loss=np.array(history['train_d_loss']),
                train_d_real_acc=np.array(history['train_d_real_acc']),
                train_d_fake_acc=np.array(history['train_d_fake_acc']),
                val_g_loss=np.array(history['val_g_loss']),
                val_d_loss=np.array(history['val_d_loss']),
                val_d_real_acc=np.array(history['val_d_real_acc']),
                val_d_fake_acc=np.array(history['val_d_fake_acc']),
                val_psnr=np.array(history['val_psnr'])
            )
        except Exception as e:
            print(f"⚠️  Erreur sauvegarde historique: {e}")

        # Sauvegarde checkpoint à chaque époque avec padding dynamique
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch+1:0{epoch_pad}d}.pth')
        torch.save(generator.state_dict(), checkpoint_path)
        checkpoints = sorted(glob(os.path.join(checkpoint_dir, f'checkpoint_epoch*.pth')))
        if len(checkpoints) > 10:
            os.remove(checkpoints[0])

        # Sauvegarder meilleur modèle dans best_dir avec padding dynamique
        if val_metrics['psnr'] > best_psnr:
            best_psnr = val_metrics['psnr']
            torch.save(generator.state_dict(), model_path)
            marker = " ✓ (best)"
            best_path = os.path.join(best_dir, f'best_epoch{epoch+1:0{epoch_pad}d}_psnr{best_psnr:.2f}.pth')
            torch.save(generator.state_dict(), best_path)
            best_checkpoints = sorted(glob(os.path.join(best_dir, 'best_epoch*.pth')))
            if len(best_checkpoints) > 5:
                os.remove(best_checkpoints[0])
        else:
            marker = ""

        # Affichage détaillé avec D_real et D_fake
        print(f"EPOCH {epoch+1:2d}/{args.epochs} | "
              f"Train G:{train_metrics['g_loss']:6.3f} D:{train_metrics['d_loss']:6.3f} | "
              f"D_real:{train_metrics['d_real_acc']:.2f} D_fake:{train_metrics['d_fake_acc']:.2f} | "
              f"Val PSNR:{val_metrics['psnr']:5.2f}dB{marker}")

        # Adaptation auto des hyperparamètres si D trop fort
        if train_metrics['d_real_acc'] > 0.85 and train_metrics['d_fake_acc'] > 0.85:
            d_strong_epochs += 1
            print(f"  ⚠️  Discriminateur trop fort ! d-lr et lambda-pixel vont être adaptés si ça continue...")
        else:
            d_strong_epochs = 0
        if d_strong_epochs >= 5:
            # Divise d-lr par 2
            for param_group in d_optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / 2
            args.d_lr = args.d_lr / 2
            # Augmente lambda-pixel de 20%
            args.lambda_pixel = args.lambda_pixel * 1.2
            print(f"  🔄 Adaptation auto : d-lr -> {args.d_lr:.6f}, lambda-pixel -> {args.lambda_pixel:.1f}")
            d_strong_epochs = 0

        # Warning si générateur trop fort
        if train_metrics['d_real_acc'] < 0.4 and train_metrics['d_fake_acc'] < 0.4:
            print(f"  ⚠️  Générateur domine ! Considérez --g-lr plus faible")
    
    print("\n" + "=" * 80)
    print("ENTRAÎNEMENT TERMINÉ ✓")
    print("=" * 80)
    print(f"\n📈 Résultats:")
    print(f"  - Meilleur PSNR: {best_psnr:.2f} dB")
    print(f"  - Modèle sauvegardé: {model_path}")
    
    # Charger le meilleur modèle
    generator.load_state_dict(torch.load(model_path, map_location=device))
    
    # Affichage des résultats
    print("\n📊 Génération des visualisations...")
    
    # Plot historique
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Pertes générateur
    axes[0, 0].plot(history['train_g_loss'], label='Train', marker='o', markersize=3)
    axes[0, 0].plot(history['val_g_loss'], label='Val', marker='s', markersize=3)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Generator Loss')
    axes[0, 0].set_title('Generator Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Pertes discriminateur
    axes[0, 1].plot(history['train_d_loss'], label='Train', marker='o', markersize=3)
    axes[0, 1].plot(history['val_d_loss'], label='Val', marker='s', markersize=3)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Discriminator Loss')
    axes[0, 1].set_title('Discriminator Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Précision discriminateur (Train + Val)
    axes[1, 0].plot(history['train_d_real_acc'], label='Train D_real', marker='o', markersize=3, color='blue')
    axes[1, 0].plot(history['train_d_fake_acc'], label='Train D_fake', marker='s', markersize=3, color='orange')
    axes[1, 0].plot(history['val_d_real_acc'], label='Val D_real', marker='^', markersize=3, color='cyan', linestyle='--')
    axes[1, 0].plot(history['val_d_fake_acc'], label='Val D_fake', marker='v', markersize=3, color='red', linestyle='--')
    axes[1, 0].axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Random (0.5)')
    axes[1, 0].axhspan(0.7, 0.8, alpha=0.1, color='green', label='Target D_real')
    axes[1, 0].axhspan(0.5, 0.6, alpha=0.1, color='yellow', label='Target D_fake')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Discriminator Accuracy (Train & Val)')
    axes[1, 0].legend(loc='best', fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # PSNR
    axes[1, 1].plot(history['val_psnr'], label='Val PSNR', marker='o', markersize=3, color='green')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('PSNR (dB)')
    axes[1, 1].set_title('Peak Signal-to-Noise Ratio')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    history_path = f'./code/unet_gan_history_{args.dataset}.png'
    plt.savefig(history_path, dpi=300, bbox_inches='tight')
    print(f"✓ Historique sauvegardé: {history_path}")
    plt.show()
    
    # Plot exemples de débruitage
    fig_results = plot_denoising_results(generator, test_dataset, device, noise_configs, n_samples=10)
    results_path = f'./code/unet_gan_results_{args.dataset}.png'
    fig_results.savefig(results_path, dpi=300, bbox_inches='tight')
    print(f"✓ Résultats sauvegardés: {results_path}")
    plt.show()
    
    print("\n" + "=" * 80)
    print("TOUT EST TERMINÉ ✓")
    print("=" * 80)
    print(f"\n💡 Pour utiliser ce modèle:")
    print(f"   model = UNet(n_channels=3, n_classes=3, base_features=64)")
    print(f"   model.load_state_dict(torch.load('{model_path}'))")
    print(f"   denoised = model(noisy_image)")
