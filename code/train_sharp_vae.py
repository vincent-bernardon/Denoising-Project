"""
Script pour entraîner le VAE ULTRA-SHARP et comparer avec l'ancien
"""
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm

from load_cifar10 import CIFAR10Loader
from vae_model_sharp import EncoderSharp, DecoderSharp, vae_loss_sharp
from vae_train import evaluate_vae, plot_training_history, save_model, calculate_mse, calculate_psnr
from utils import add_noise_to_images


def train_vae_sharp(encoder, decoder, x_train, 
                    epochs=30,              # RÉDUIT à 30 epochs (4 min x 30 = 2h)
                    batch_size=128, 
                    learning_rate=5e-4,     # RÉDUIT pour éviter NaN (2e-3 → 5e-4)
                    noise_type='gaussian',
                    noise_params=None,
                    device=None,
                    validation_split=0.1,
                    verbose=True):
    """
    Entraîne le VAE Sharp avec loss L1 pure et beta=0
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if noise_params is None:
        noise_params = {'std': 25}
    
    if verbose:
        print("\n🚀 ENTRAÎNEMENT VAE ULTRA-SHARP")
        print("=" * 60)
        print(f"Device: {device}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Noise: {noise_type} {noise_params}")
        print(f"Loss: 100% L1 (anti-flou)")
        print(f"Beta: 0.0 (autoencodeur pur)")
        print("=" * 60)
    
    # Modèles sur device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    # Optimiseur
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=learning_rate
    )
    
    # Split train/val
    n_samples = len(x_train)
    n_val = int(n_samples * validation_split)
    n_train = n_samples - n_val
    
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    x_train_split = x_train[train_indices]
    x_val_split = x_train[val_indices]
    
    # Tensors
    x_train_tensor = torch.FloatTensor(x_train_split).permute(0, 3, 1, 2) / 255.0
    x_val_tensor = torch.FloatTensor(x_val_split).permute(0, 3, 1, 2) / 255.0
    
    # DataLoader
    train_dataset = TensorDataset(x_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if verbose:
        print(f"\nTrain: {len(x_train_split)} | Val: {len(x_val_split)}")
        print(f"Batches/epoch: {len(train_loader)}\n")
    
    # Historique
    history = {
        'train_loss': [],
        'train_recon_loss': [],
        'train_kl_loss': [],
        'val_loss': [],
        'epochs': []
    }
    
    # ENTRAÎNEMENT
    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        
        epoch_total_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") if verbose else train_loader
        
        for batch_idx, (x_clean,) in enumerate(pbar):
            # Ajouter du bruit
            x_clean_np = (x_clean.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
            x_noisy_np = add_noise_to_images(x_clean_np, noise_type=noise_type, verbose=False, **noise_params)
            x_noisy = torch.FloatTensor(x_noisy_np).permute(0, 3, 1, 2) / 255.0
            
            x_clean = x_clean.to(device)
            x_noisy = x_noisy.to(device)
            
            # Forward
            mu, logvar = encoder(x_noisy)
            z = encoder.reparameterize(mu, logvar)
            x_recon = decoder(z)
            
            # Loss SHARP (100% L1, beta=0.0)
            total_loss, recon_loss, kl_loss = vae_loss_sharp(
                x_recon, x_clean, mu, logvar, beta=0.0
            )
            
            # Normaliser
            batch_size_actual = x_clean.size(0)
            total_loss = total_loss / batch_size_actual
            recon_loss = recon_loss / batch_size_actual
            kl_loss = kl_loss / batch_size_actual
            
            # Backprop
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Stats
            epoch_total_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            
            if verbose and isinstance(pbar, tqdm):
                pbar.set_postfix({
                    'loss': f'{total_loss.item():.4f}',
                    'recon': f'{recon_loss.item():.4f}'
                })
        
        # Moyennes
        epoch_total_loss /= len(train_loader)
        epoch_recon_loss /= len(train_loader)
        epoch_kl_loss /= len(train_loader)
        
        # Validation
        encoder.eval()
        decoder.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            x_val_np = (x_val_tensor.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
            x_val_noisy_np = add_noise_to_images(x_val_np, noise_type=noise_type, verbose=False, **noise_params)
            x_val_noisy = torch.FloatTensor(x_val_noisy_np).permute(0, 3, 1, 2) / 255.0
            
            x_val_clean = x_val_tensor.to(device)
            x_val_noisy = x_val_noisy.to(device)
            
            mu_val, logvar_val = encoder(x_val_noisy)
            z_val = encoder.reparameterize(mu_val, logvar_val)
            x_val_recon = decoder(z_val)
            
            val_total_loss, _, _ = vae_loss_sharp(x_val_recon, x_val_clean, mu_val, logvar_val, beta=0.0)
            val_loss = val_total_loss.item() / x_val_clean.size(0)
        
        # Historique
        history['train_loss'].append(epoch_total_loss)
        history['train_recon_loss'].append(epoch_recon_loss)
        history['train_kl_loss'].append(epoch_kl_loss)
        history['val_loss'].append(val_loss)
        history['epochs'].append(epoch + 1)
        
        if verbose:
            print(f"\nEpoch {epoch+1}/{epochs} - Train Loss: {epoch_total_loss:.4f} - Val Loss: {val_loss:.4f}")
    
    if verbose:
        print("\n" + "=" * 60)
        print("✅ ENTRAÎNEMENT TERMINÉ")
        print("=" * 60)
    
    return history


def main():
    print("=" * 60)
    print("ENTRAÎNEMENT VAE ULTRA-SHARP (Anti-Flou)")
    print("=" * 60)
    
    # Charger données
    loader = CIFAR10Loader()
    x_train, y_train, x_test, y_test = loader.load_all_data()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Modèle Sharp
    latent_dim = 256  # Plus grand pour plus d'info
    encoder = EncoderSharp(latent_dim=latent_dim)
    decoder = DecoderSharp(latent_dim=latent_dim)
    
    print(f"✅ VAE Sharp initialisé (latent_dim={latent_dim})")
    total_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in decoder.parameters())
    print(f"📊 Paramètres: {total_params:,}\n")
    
    # ENTRAÎNEMENT avec paramètres ultra-agressifs
    history = train_vae_sharp(
        encoder=encoder,
        decoder=decoder,
        x_train=x_train,
        epochs=30,                      # Réduit à 30 epochs (~2h au lieu de 7h)
        batch_size=128,
        learning_rate=5e-4,             # Learning rate réduit pour éviter NaN
        noise_type='gaussian',
        noise_params={'std': 25},
        device=device,
        validation_split=0.1,
        verbose=True
    )
    
    # Afficher historique
    plot_training_history(history)
    
    # Sauvegarder
    save_model(encoder, decoder, history, filepath='./code/vae_denoiser_sharp.pth')
    
    # ÉVALUATION
    print("\n" + "=" * 60)
    print("ÉVALUATION VAE SHARP")
    print("=" * 60)
    
    # Gaussien
    print("\n🔬 Test 1/3: Bruit Gaussien")
    metrics_gaussian = evaluate_vae(
        encoder=encoder,
        decoder=decoder,
        x_test=x_test,
        y_test=y_test,
        class_names=loader.class_names,
        noise_type='gaussian',
        noise_params={'std': 25},
        n_samples=5,
        device=device,
        verbose=True
    )
    
    # Salt & Pepper
    print("\n🔬 Test 2/3: Bruit Salt & Pepper")
    metrics_salt = evaluate_vae(
        encoder=encoder,
        decoder=decoder,
        x_test=x_test,
        y_test=y_test,
        class_names=loader.class_names,
        noise_type='salt_pepper',
        noise_params={'salt_prob': 0.02, 'pepper_prob': 0.02},
        n_samples=5,
        device=device,
        verbose=True
    )
    
    # Mixte
    print("\n🔬 Test 3/3: Bruit Mixte")
    metrics_mixed = evaluate_vae(
        encoder=encoder,
        decoder=decoder,
        x_test=x_test,
        y_test=y_test,
        class_names=loader.class_names,
        noise_type='mixed',
        noise_params={'gaussian_std': 20, 'salt_prob': 0.01, 'pepper_prob': 0.01},
        n_samples=5,
        device=device,
        verbose=True
    )
    
    # RÉSUMÉ
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ FINAL - VAE ULTRA-SHARP")
    print("=" * 60)
    print(f"\n{'Type de bruit':<20s} {'MSE avant':<12s} {'MSE après':<12s} {'PSNR avant':<12s} {'PSNR après':<12s} {'Amélioration':<12s}")
    print("-" * 95)
    print(f"{'Gaussien':<20s} {metrics_gaussian['mse_noisy_vs_clean']:<12.2f} {metrics_gaussian['mse_denoised_vs_clean']:<12.2f} {metrics_gaussian['psnr_noisy_vs_clean']:<12.2f} {metrics_gaussian['psnr_denoised_vs_clean']:<12.2f} {metrics_gaussian['psnr_improvement']:+<12.2f} dB")
    print(f"{'Salt & Pepper':<20s} {metrics_salt['mse_noisy_vs_clean']:<12.2f} {metrics_salt['mse_denoised_vs_clean']:<12.2f} {metrics_salt['psnr_noisy_vs_clean']:<12.2f} {metrics_salt['psnr_denoised_vs_clean']:<12.2f} {metrics_salt['psnr_improvement']:+<12.2f} dB")
    print(f"{'Mixte':<20s} {metrics_mixed['mse_noisy_vs_clean']:<12.2f} {metrics_mixed['mse_denoised_vs_clean']:<12.2f} {metrics_mixed['psnr_noisy_vs_clean']:<12.2f} {metrics_mixed['psnr_denoised_vs_clean']:<12.2f} {metrics_mixed['psnr_improvement']:+<12.2f} dB")
    print("-" * 95)
    
    print("\n✅ TERMINÉ!")
    print("\nSi toujours flou avec ce modèle, c'est la limite des images 32x32 CIFAR-10")
    print("Pour de vraies images haute résolution, le modèle fonctionnera BEAUCOUP mieux!")
    
    return encoder, decoder, history


if __name__ == "__main__":
    encoder, decoder, history = main()
