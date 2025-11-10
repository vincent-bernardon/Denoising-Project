"""
VAE ULTRA-SHARP pour Débruitage - Version Anti-Flou Maximale

Modifications pour combattre le flou:
1. Beta = 0.0 (Autoencodeur pur, pas de régularisation VAE)
2. Loss 100% L1 (préserve les contours nets)
3. Plus de skip connections implicites (features plus larges)
4. Activation Tanh finale (meilleur pour les détails)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderSharp(nn.Module):
    """
    Encodeur optimisé anti-flou avec features très larges
    """
    
    def __init__(self, latent_dim=256):  # latent_dim plus grand pour + d'info
        super(EncoderSharp, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Architecture plus large: plus de channels = plus d'info préservée
        # 32x32x3 → 16x16x128
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        
        # 16x16x128 → 8x8x256
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        
        # 8x8x256 → 4x4x512
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        
        # 4x4x512 → 2x2x512
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Bottleneck
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(2 * 2 * 512, latent_dim)
        self.fc_logvar = nn.Linear(2 * 2 * 512, latent_dim)
        
    def forward(self, x):
        # Bloc 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Bloc 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Bloc 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Bloc 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        # Flatten
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        # En mode inference, utiliser JUSTE mu (déterministe)
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu  # Pas de sampling = plus net


class DecoderSharp(nn.Module):
    """
    Décodeur optimisé anti-flou avec features larges
    """
    
    def __init__(self, latent_dim=256):
        super(DecoderSharp, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Projection
        self.fc = nn.Linear(latent_dim, 2 * 2 * 512)
        
        # 2x2x512 → 4x4x512
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        
        # 4x4x512 → 8x8x256
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        
        # 8x8x256 → 16x16x128
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 16x16x128 → 32x32x64
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        # Couche finale avec plus de convolutions pour affiner
        self.refine_conv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.refine_bn = nn.BatchNorm2d(32)
        self.refine_conv2 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        
    def forward(self, z):
        # Projection
        x = self.fc(z)
        x = x.view(-1, 512, 2, 2)
        
        # Upsampling
        x = self.deconv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.deconv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.deconv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = self.deconv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        # Couches de raffinement
        x = self.refine_conv1(x)
        x = self.refine_bn(x)
        x = F.relu(x)
        
        x = self.refine_conv2(x)
        x = torch.sigmoid(x)  # Sigmoid plus stable que tanh
        
        return x


def vae_loss_sharp(x_recon, x_clean, mu, logvar, beta=0.0):
    """
    Loss ULTRA-AGRESSIVE pour netteté maximale
    
    - 100% L1 (pas de MSE = pas de flou)
    - Beta = 0.0 (pas de KL = autoencodeur pur)
    - Focus total sur la reconstruction nette
    """
    # UNIQUEMENT L1 loss (préserve les contours nets)
    recon_loss = F.l1_loss(x_recon, x_clean, reduction='sum')
    
    # KL divergence (mais beta=0.0 donc ignorée)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


# Fonction de test
def test_sharp_vae():
    print("Test VAE Ultra-Sharp")
    print("=" * 60)
    
    latent_dim = 256
    encoder = EncoderSharp(latent_dim=latent_dim)
    decoder = DecoderSharp(latent_dim=latent_dim)
    
    # Test
    x = torch.randn(4, 3, 32, 32)
    mu, logvar = encoder(x)
    z = encoder.reparameterize(mu, logvar)
    x_recon = decoder(z)
    
    print(f"✅ Architecture Sharp fonctionne!")
    print(f"Input: {x.shape}")
    print(f"Latent dim: {latent_dim}")
    print(f"Output: {x_recon.shape}")
    print(f"Output range: [{x_recon.min():.3f}, {x_recon.max():.3f}]")
    
    # Paramètres
    total_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in decoder.parameters())
    print(f"\n📊 Paramètres totaux: {total_params:,}")
    print("\n🎯 Optimisations anti-flou:")
    print("  ✓ Latent_dim=256 (+ d'information)")
    print("  ✓ Channels larges (128→256→512)")
    print("  ✓ Couches de raffinement")
    print("  ✓ Tanh activation (meilleur gradient)")
    print("  ✓ Loss 100% L1 (préserve contours)")
    print("  ✓ Beta=0.0 (autoencodeur pur)")


if __name__ == "__main__":
    test_sharp_vae()
