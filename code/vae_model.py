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
        
        # Couche fully-connected pour passer de latent_dim à 2x2x256
        self.fc = nn.Linear(latent_dim, 2 * 2 * 256)
        
        # Bloc 1: 2x2x256 → 4x4x128
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        
        # Bloc 2: 4x4x128 → 8x8x64
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Bloc 3: 8x8x64 → 16x16x32
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        
        # Bloc 4: 16x16x32 → 32x32x3
        self.deconv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
        
    def forward(self, z):
        """
        Forward pass du décodeur
        
        Args:
            z (torch.Tensor): Vecteur latent (batch, latent_dim)
            
        Returns:
            x_recon (torch.Tensor): Image reconstruite (batch, 3, 32, 32)
        """
        # Fully connected: latent_dim → 1024
        x = self.fc(z)                      # (batch, 1024)
        x = x.view(-1, 256, 2, 2)           # (batch, 256, 2, 2)
        
        # Bloc 1: 2x2x256 → 4x4x128
        x = self.deconv1(x)                 # (batch, 128, 4, 4)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)
        
        # Bloc 2: 4x4x128 → 8x8x64
        x = self.deconv2(x)                 # (batch, 64, 8, 8)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)
        
        # Bloc 3: 8x8x64 → 16x16x32
        x = self.deconv3(x)                 # (batch, 32, 16, 16)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.2)
        
        # Bloc 4: 16x16x32 → 32x32x3
        x = self.deconv4(x)                 # (batch, 3, 32, 32)
        x_recon = torch.sigmoid(x)          # Valeurs entre [0, 1]
        
        return x_recon


def vae_loss(x_recon, x_clean, mu, logvar, beta=1.0):
    """
    Fonction de perte VAE pour le débruitage d'images
    
    Args:
        x_recon (torch.Tensor): Image reconstruite par le décodeur (batch, C, H, W)
        x_clean (torch.Tensor): Image propre originale (batch, C, H, W)
        mu (torch.Tensor): Moyenne de la distribution latente (batch, latent_dim)
        logvar (torch.Tensor): Log-variance de la distribution latente (batch, latent_dim)
        beta (float): Facteur de pondération de la KL divergence (défaut: 1.0)
                     Utiliser beta < 1.0 au début de l'entraînement (KL annealing)
    
    Returns:
        total_loss (torch.Tensor): Perte totale (scalaire)
        recon_loss (torch.Tensor): Perte de reconstruction (scalaire)
        kl_loss (torch.Tensor): KL divergence (scalaire)
        
    Notes:
        - Reconstruction Loss (MSE): Mesure la différence entre l'image reconstruite 
          et l'image PROPRE (pas l'image bruitée). Plus cette valeur est faible,
          mieux le VAE débruite.
          
        - KL Divergence: Force la distribution latente q(z|x) à rester proche de 
          la prior N(0,1). Cela régularise l'espace latent et permet une meilleure
          généralisation.
          
          Formule: KL(q(z|x) || p(z)) = -0.5 * sum(1 + log(σ²) - μ² - σ²)
          où σ² = exp(logvar)
          
        - Beta (β): Coefficient qui contrôle l'importance de la KL divergence.
          * β = 0: Pas de régularisation (risque d'overfitting)
          * β = 1: VAE standard
          * β > 1: β-VAE (meilleure disentanglement)
          * β croissant: KL annealing (commence petit, monte vers 1)
    """
    # Reconstruction Loss (MSE)
    # Comparer l'image reconstruite avec l'image PROPRE (pas bruitée)
    recon_loss = F.mse_loss(x_recon, x_clean, reduction='sum')
    
    # KL Divergence
    # KL(q(z|x) || p(z)) où q(z|x) = N(mu, exp(logvar)) et p(z) = N(0, 1)
    # Formule: -0.5 * sum(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Perte totale
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss
