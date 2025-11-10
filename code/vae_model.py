import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encodeur du VAE - Compresse l'image bruitée en une distribution latente
    
    Architecture:
        Input: (batch, 3, 32, 32) - Images CIFAR-10
        Output: mu (batch, latent_dim), logvar (batch, latent_dim)
        
    Couches convolutionnelles avec réduction progressive:
        32x32x3 → 16x16x32 → 8x8x64 → 4x4x128 → 2x2x256
    """
    
    def __init__(self, latent_dim=128):
        """
        Initialise l'encodeur
        
        Args:
            latent_dim (int): Dimension de l'espace latent (défaut: 128)
        """
        super(Encoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Bloc 1: 32x32x3 → 16x16x32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Bloc 2: 16x16x32 → 8x8x64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Bloc 3: 8x8x64 → 4x4x128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Bloc 4: 4x4x128 → 2x2x256
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Taille finale après convolutions: 2x2x256 = 1024 features
        self.flatten = nn.Flatten()
        
        # Couches fully-connected pour mu et logvar
        self.fc_mu = nn.Linear(2 * 2 * 256, latent_dim)
        self.fc_logvar = nn.Linear(2 * 2 * 256, latent_dim)
        
    def forward(self, x):
        """
        Forward pass de l'encodeur
        
        Args:
            x (torch.Tensor): Images d'entrée (batch, 3, 32, 32)
            
        Returns:
            mu (torch.Tensor): Moyenne de la distribution latente (batch, latent_dim)
            logvar (torch.Tensor): Log-variance de la distribution (batch, latent_dim)
        """
        # Bloc 1
        x = self.conv1(x)           # (batch, 32, 16, 16)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)
        
        # Bloc 2
        x = self.conv2(x)           # (batch, 64, 8, 8)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)
        
        # Bloc 3
        x = self.conv3(x)           # (batch, 128, 4, 4)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.2)
        
        # Bloc 4
        x = self.conv4(x)           # (batch, 256, 2, 2)
        x = self.bn4(x)
        x = F.leaky_relu(x, 0.2)
        
        # Aplatir
        x = self.flatten(x)         # (batch, 1024)
        
        # Calculer mu et logvar
        mu = self.fc_mu(x)          # (batch, latent_dim)
        logvar = self.fc_logvar(x)  # (batch, latent_dim)
        
        return mu, logvar
