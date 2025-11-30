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
        
        # Bloc 1: 32x32x3 → 16x16x64 (+ de channels pour + de capacité)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Bloc 2: 16x16x64 → 8x8x128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Bloc 3: 8x8x128 → 4x4x256
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Bloc 4: 4x4x256 → 2x2x512
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Taille finale après convolutions: 2x2x512 = 2048 features
        self.flatten = nn.Flatten()
        
        # Couches fully-connected pour mu et logvar
        self.fc_mu = nn.Linear(2 * 2 * 512, latent_dim)
        self.fc_logvar = nn.Linear(2 * 2 * 512, latent_dim)
        
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
        x = self.conv1(x)           # (batch, 64, 16, 16)
        x = self.bn1(x)
        x = F.relu(x)               # ReLU standard (+ stable que LeakyReLU)
        
        # Bloc 2
        x = self.conv2(x)           # (batch, 128, 8, 8)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Bloc 3
        x = self.conv3(x)           # (batch, 256, 4, 4)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Bloc 4
        x = self.conv4(x)           # (batch, 512, 2, 2)
        x = self.bn4(x)
        x = F.relu(x)
        
        # Aplatir
        x = self.flatten(x)         # (batch, 2048)
        
        # Calculer mu et logvar
        mu = self.fc_mu(x)          # (batch, latent_dim)
        logvar = self.fc_logvar(x)  # (batch, latent_dim)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick pour échantillonner z de manière différentiable
        
        Args:
            mu (torch.Tensor): Moyenne de la distribution latente (batch, latent_dim)
            logvar (torch.Tensor): Log-variance de la distribution (batch, latent_dim)
            
        Returns:
            z (torch.Tensor): Vecteur latent échantillonné (batch, latent_dim)
            
        Note:
            z = mu + sigma * epsilon
            où sigma = exp(log(sigma²) / 2) = exp(logvar / 2)
            et epsilon ~ N(0, 1)
        """
        std = torch.exp(0.5 * logvar)  # sigma = exp(logvar / 2)
        eps = torch.randn_like(std)     # epsilon ~ N(0, 1)
        z = mu + eps * std              # z = mu + sigma * epsilon
        return z


class Decoder(nn.Module):
    """
    Décodeur du VAE - Reconstruit l'image propre à partir du vecteur latent
    
    Architecture:
        Input: (batch, latent_dim) - Vecteur latent z
        Output: (batch, 3, 32, 32) - Image reconstruite
        
    Couches de déconvolution avec agrandissement progressif:
        latent_dim → 2x2x256 → 4x4x128 → 8x8x64 → 16x16x32 → 32x32x3
    """
    
    def __init__(self, latent_dim=128):
        """
        Initialise le décodeur
        
        Args:
            latent_dim (int): Dimension de l'espace latent (défaut: 128)
        """
        super(Decoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Couche fully-connected pour passer de latent_dim à 2x2x512
        self.fc = nn.Linear(latent_dim, 2 * 2 * 512)
        
        # Bloc 1: 2x2x512 → 4x4x256
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        
        # Bloc 2: 4x4x256 → 8x8x128
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Bloc 3: 8x8x128 → 16x16x64
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Bloc 4: 16x16x64 → 32x32x32
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        
        # Couche finale: 32x32x32 → 32x32x3 avec convolution 1x1 pour plus de précision
        self.final_conv = nn.Conv2d(32, 3, kernel_size=1)
        
    def forward(self, z):
        """
        Forward pass du décodeur
        
        Args:
            z (torch.Tensor): Vecteur latent (batch, latent_dim)
            
        Returns:
            x_recon (torch.Tensor): Image reconstruite (batch, 3, 32, 32)
        """
        # Fully connected: latent_dim → 2048
        x = self.fc(z)                      # (batch, 2048)
        x = x.view(-1, 512, 2, 2)           # (batch, 512, 2, 2)
        
        # Bloc 1: 2x2x512 → 4x4x256
        x = self.deconv1(x)                 # (batch, 256, 4, 4)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Bloc 2: 4x4x256 → 8x8x128
        x = self.deconv2(x)                 # (batch, 128, 8, 8)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Bloc 3: 8x8x128 → 16x16x64
        x = self.deconv3(x)                 # (batch, 64, 16, 16)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Bloc 4: 16x16x64 → 32x32x32
        x = self.deconv4(x)                 # (batch, 32, 32, 32)
        x = self.bn4(x)
        x = F.relu(x)
        
        # Couche finale: 32x32x32 → 32x32x3
        x = self.final_conv(x)              # (batch, 3, 32, 32)
        x_recon = torch.sigmoid(x)          # Valeurs entre [0, 1]
        
        return x_recon


def vae_loss(x_recon, x_clean, mu, logvar, beta=1.0):
    """
    Fonction de perte VAE améliorée pour le débruitage d'images
    Combine MSE + perte perceptuelle (L1) pour réduire le flou
    
    Args:
        x_recon (torch.Tensor): Image reconstruite par le décodeur (batch, C, H, W)
        x_clean (torch.Tensor): Image propre originale (batch, C, H, W)
        mu (torch.Tensor): Moyenne de la distribution latente (batch, latent_dim)
        logvar (torch.Tensor): Log-variance de la distribution latente (batch, latent_dim)
        beta (float): Facteur de pondération de la KL divergence (défaut: 1.0)
    
    Returns:
        total_loss (torch.Tensor): Perte totale (scalaire)
        recon_loss (torch.Tensor): Perte de reconstruction (scalaire)
        kl_loss (torch.Tensor): KL divergence (scalaire)
        
    Notes:
        - MSE Loss: Mesure pixel-par-pixel (tend à produire du flou)
        - L1 Loss: Préserve mieux les détails et contours
        - Combinaison 50/50 pour équilibrer précision et netteté
        - Beta réduit favorise la reconstruction au détriment de la régularisation
    """
    # Reconstruction Loss combinée: MSE + L1 pour réduire le flou
    mse_loss = F.mse_loss(x_recon, x_clean, reduction='sum')
    l1_loss = F.l1_loss(x_recon, x_clean, reduction='sum')
    
    # Combiner MSE et L1 (50/50) pour meilleure qualité visuelle
    recon_loss = 0.5 * mse_loss + 0.5 * l1_loss
    
    # KL Divergence (régularisation de l'espace latent)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Perte totale
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss
