"""
Entraînement du U-Net avec les 3 types de bruit mélangés aléatoirement
(comme le VAE beta0 qui s'entraîne sur plusieurs types de bruit)
UN SEUL MODÈLE pour les 3 bruits
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

from unet_model import UNet
from utils import add_noise_to_images


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
    targets = np.array(base_test_dataset.targets)
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


if __name__ == "__main__":
    print("=" * 80)
    print("ENTRAÎNEMENT U-NET AVEC MULTI-BRUIT MÉLANGÉ")
    print("UN SEUL MODÈLE pour les 3 types de bruit")
    print("=" * 80)
    
    # Configuration
    data_dir = 'dataset'
    batch_size = 64
    lr = 0.001
    num_epochs = 30
    
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
    
    # Chargement des données de base
    print("\nChargement de CIFAR-10...")
    base_train_dataset = torchvision.datasets.CIFAR10(
        data_dir, train=True, download=True,
        transform=transforms.ToTensor()
    )
    base_test_dataset = torchvision.datasets.CIFAR10(
        data_dir, train=False, download=True,
        transform=transforms.ToTensor()
    )
    
    # Créer les datasets avec bruit aléatoire
    print("Application des bruits aléatoires...")
    train_dataset = MultiNoisyDataset(base_train_dataset, noise_configs)
    test_dataset = MultiNoisyDataset(base_test_dataset, noise_configs)
    
    # Split train/validation
    m = len(train_dataset)
    train_data, val_data = random_split(train_dataset, [int(m-m*0.2), int(m*0.2)])
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(val_data, batch_size=batch_size, num_workers=2)
    
    print(f"✓ Train: {len(train_data)} images")
    print(f"✓ Val:   {len(val_data)} images")
    print(f"✓ Test:  {len(test_dataset)} images")
    
    # Initialisation du modèle U-Net
    print("\nInitialisation du modèle U-Net...")
    model = UNet(n_channels=3, n_classes=3, base_features=64)
    model.to(device)
    
    # Compter les paramètres
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Nombre de paramètres: {total_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Entraînement
    print("\n" + "=" * 80)
    print("DÉBUT DE L'ENTRAÎNEMENT")
    print("=" * 80)
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    model_path = './code/unet_denoising_multinoise.pth'
    
    for epoch in range(num_epochs):
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
        
        print(f'EPOCH {epoch + 1:2d}/{num_epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}{marker}')
    
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
    plot_results_multi_noise(model, base_test_dataset, device, noise_configs)
    
    # Plot de l'historique
    print("\nAffichage de l'historique d'entraînement...")
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss', marker='o', markersize=3)
    plt.plot(history['val_loss'], label='Val Loss', marker='s', markersize=3)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.title('U-Net Training History - Multi-Noise', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Sauvegarder le graphique
    history_path = './code/unet_history_multinoise.png'
    plt.savefig(history_path, dpi=300, bbox_inches='tight')
    print(f"✓ Historique sauvegardé: {history_path}")
    
    plt.show()
    
    print("\n" + "=" * 80)
    print("TOUT EST TERMINÉ ✓")
    print("=" * 80)
    print(f"\nPour évaluer ce modèle, modifiez unet_evaluate.py:")
    print(f"  model_path = './code/unet_denoising_multinoise.pth'")