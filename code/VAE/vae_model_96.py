import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder96(nn.Module):
    """Encodeur adapté aux images 96x96 (STL-10)."""

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)   # 96 -> 48
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 48 -> 24
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # 24 -> 12
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)  # 12 -> 6
        self.bn4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)  # 6 -> 3
        self.bn5 = nn.BatchNorm2d(512)

        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(512 * 3 * 3, latent_dim)
        self.fc_logvar = nn.Linear(512 * 3 * 3, latent_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class Decoder96(nn.Module):
    """Décodeur pour reconstruire des images 96x96."""

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, 512 * 3 * 3)

        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)  # 3 -> 6
        self.bn1 = nn.BatchNorm2d(512)

        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # 6 -> 12
        self.bn2 = nn.BatchNorm2d(256)

        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 12 -> 24
        self.bn3 = nn.BatchNorm2d(128)

        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # 24 -> 48
        self.bn4 = nn.BatchNorm2d(64)

        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)    # 48 -> 96
        self.bn5 = nn.BatchNorm2d(32)

        self.final_conv = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, z):
        x = self.fc(z).view(-1, 512, 3, 3)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.relu(self.bn4(self.deconv4(x)))
        x = F.relu(self.bn5(self.deconv5(x)))
        x = self.final_conv(x)
        return torch.sigmoid(x)