"""
Module d'entraînement et d'évaluation du VAE pour le débruitage d'images

Ce module implémente :
- L'entraînement du VAE avec ajout de bruit
- L'évaluation visuelle et quantitative (MSE, PSNR)
- La sauvegarde/chargement du modèle
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from vae_model import Encoder, Decoder, vae_loss
from utils import add_noise_to_images
from image_visualizer import ImageVisualizer


def calculate_psnr(img1, img2, max_pixel_value=255.0):
    """
    Calcule le PSNR (Peak Signal-to-Noise Ratio) entre deux images
    
    Args:
        img1 (np.ndarray): Première image (H, W, C) ou (N, H, W, C)
        img2 (np.ndarray): Deuxième image (même shape que img1)
        max_pixel_value (float): Valeur maximale des pixels (255 pour uint8)
    
    Returns:
        float: PSNR en dB (plus élevé = meilleur)
    """
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr


def calculate_mse(img1, img2):
    """
    Calcule le MSE (Mean Squared Error) entre deux images
    
    Args:
        img1 (np.ndarray): Première image (H, W, C) ou (N, H, W, C)
        img2 (np.ndarray): Deuxième image (même shape que img1)
    
    Returns:
        float: MSE (plus faible = meilleur)
    """
    return np.mean((img1.astype(float) - img2.astype(float)) ** 2)


def train_vae(encoder, decoder, x_train, 
              epochs=30, 
              batch_size=128, 
              learning_rate=1e-3,
              noise_type='gaussian',
              noise_params=None,
              beta=1.0,
              device=None,
              validation_split=0.1,
              verbose=True):
    """
    ÉTAPE 7 : ENTRAÎNEMENT DU VAE
    
    Entraîne le VAE pour le débruitage d'images
    
    Args:
        encoder (nn.Module): Encodeur du VAE
        decoder (nn.Module): Décodeur du VAE
        x_train (np.ndarray): Images d'entraînement propres (N, H, W, C)
        epochs (int): Nombre d'epochs
        batch_size (int): Taille des batchs
        learning_rate (float): Taux d'apprentissage
        noise_type (str): Type de bruit ('gaussian', 'salt_pepper', 'mixed')
        noise_params (dict): Paramètres du bruit
        beta (float): Coefficient pour la KL divergence
        device (torch.device): Device (CPU ou CUDA)
        validation_split (float): Proportion de données pour validation
        verbose (bool): Afficher les détails
    
    Returns:
        dict: Historique d'entraînement (losses)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if noise_params is None:
        if noise_type == 'gaussian':
            noise_params = {'std': 25}
        elif noise_type == 'salt_pepper':
            noise_params = {'salt_prob': 0.02, 'pepper_prob': 0.02}
        elif noise_type == 'mixed':
            noise_params = {'gaussian_std': 20, 'salt_prob': 0.01, 'pepper_prob': 0.01}
    
    if verbose:
        print("\n" + "=" * 60)
        print("ÉTAPE 7 : ENTRAÎNEMENT DU VAE DÉBRUITEUR")
        print("=" * 60)
        print(f"Device: {device}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Type de bruit: {noise_type}")
        print(f"Paramètres du bruit: {noise_params}")
        print(f"Beta (KL): {beta}")
        print("=" * 60)
    
    # Déplacer les modèles sur le device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    # Optimiseur
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=learning_rate
    )
    
    # Split train/validation
    n_samples = len(x_train)
    n_val = int(n_samples * validation_split)
    n_train = n_samples - n_val
    
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    x_train_split = x_train[train_indices]
    x_val_split = x_train[val_indices]
    
    # Préparer les données
    # Convertir de (N, H, W, C) à (N, C, H, W) et normaliser à [0, 1]
    x_train_tensor = torch.FloatTensor(x_train_split).permute(0, 3, 1, 2) / 255.0
    x_val_tensor = torch.FloatTensor(x_val_split).permute(0, 3, 1, 2) / 255.0
    
    # DataLoader
    train_dataset = TensorDataset(x_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if verbose:
        print(f"\nDonnées d'entraînement: {len(x_train_split)}")
        print(f"Données de validation: {len(x_val_split)}")
        print(f"Nombre de batchs par epoch: {len(train_loader)}\n")
    
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
        
        # Barre de progression
        if verbose:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        else:
            pbar = train_loader
        
        for batch_idx, (x_clean,) in enumerate(pbar):
            # a) Ajouter du bruit aux images propres
            x_clean_np = (x_clean.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
            x_noisy_np = add_noise_to_images(x_clean_np, noise_type=noise_type, verbose=False, **noise_params)
            x_noisy = torch.FloatTensor(x_noisy_np).permute(0, 3, 1, 2) / 255.0
            
            # Déplacer sur le device
            x_clean = x_clean.to(device)
            x_noisy = x_noisy.to(device)
            
            # b) Passer l'image bruitée dans le VAE
            # Encodage
            mu, logvar = encoder(x_noisy)
            
            # Reparameterization trick
            z = encoder.reparameterize(mu, logvar)
            
            # Décodage
            x_recon = decoder(z)
            
            # c) Calculer la perte (reconstruction + KL)
            total_loss, recon_loss, kl_loss = vae_loss(
                x_recon, x_clean, mu, logvar, beta=beta
            )
            
            # Normaliser par la taille du batch
            batch_size_actual = x_clean.size(0)
            total_loss = total_loss / batch_size_actual
            recon_loss = recon_loss / batch_size_actual
            kl_loss = kl_loss / batch_size_actual
            
            # d) Backpropagation et mise à jour des poids
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Accumuler les pertes
            epoch_total_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            
            # Mettre à jour la barre de progression
            if verbose and isinstance(pbar, tqdm):
                pbar.set_postfix({
                    'loss': f'{total_loss.item():.4f}',
                    'recon': f'{recon_loss.item():.4f}',
                    'kl': f'{kl_loss.item():.4f}'
                })
        
        # Moyenne des pertes sur l'epoch
        epoch_total_loss /= len(train_loader)
        epoch_recon_loss /= len(train_loader)
        epoch_kl_loss /= len(train_loader)
        
        # Validation
        encoder.eval()
        decoder.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            # Ajouter du bruit aux données de validation
            x_val_np = (x_val_tensor.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
            x_val_noisy_np = add_noise_to_images(x_val_np, noise_type=noise_type, verbose=False, **noise_params)
            x_val_noisy = torch.FloatTensor(x_val_noisy_np).permute(0, 3, 1, 2) / 255.0
            
            x_val_clean = x_val_tensor.to(device)
            x_val_noisy = x_val_noisy.to(device)
            
            # Forward pass
            mu_val, logvar_val = encoder(x_val_noisy)
            z_val = encoder.reparameterize(mu_val, logvar_val)
            x_val_recon = decoder(z_val)
            
            # Calculer la perte
            val_total_loss, _, _ = vae_loss(x_val_recon, x_val_clean, mu_val, logvar_val, beta=beta)
            val_loss = val_total_loss.item() / x_val_clean.size(0)
        
        # Sauvegarder l'historique
        history['train_loss'].append(epoch_total_loss)
        history['train_recon_loss'].append(epoch_recon_loss)
        history['train_kl_loss'].append(epoch_kl_loss)
        history['val_loss'].append(val_loss)
        history['epochs'].append(epoch + 1)
        
        if verbose:
            print(f"\nEpoch {epoch+1}/{epochs} - Train Loss: {epoch_total_loss:.4f} - Val Loss: {val_loss:.4f}")
    
    if verbose:
        print("\n" + "=" * 60)
        print("ENTRAÎNEMENT TERMINÉ ✔")
        print("=" * 60)
    
    return history


def evaluate_vae(encoder, decoder, x_test, y_test=None, class_names=None,
                 noise_type='gaussian',
                 noise_params=None,
                 n_samples=5,
                 device=None,
                 verbose=True):
    """
    ÉTAPE 8 : ÉVALUATION DU VAE
    
    Évalue le VAE visuellement et quantitativement
    
    Args:
        encoder (nn.Module): Encodeur du VAE
        decoder (nn.Module): Décodeur du VAE
        x_test (np.ndarray): Images de test propres (N, H, W, C)
        y_test (np.ndarray): Labels (optionnel)
        class_names (list): Noms des classes
        noise_type (str): Type de bruit à tester
        noise_params (dict): Paramètres du bruit
        n_samples (int): Nombre d'images à visualiser
        device (torch.device): Device
        verbose (bool): Afficher les détails
    
    Returns:
        dict: Métriques (MSE et PSNR moyens)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if noise_params is None:
        if noise_type == 'gaussian':
            noise_params = {'std': 25}
        elif noise_type == 'salt_pepper':
            noise_params = {'salt_prob': 0.02, 'pepper_prob': 0.02}
        elif noise_type == 'mixed':
            noise_params = {'gaussian_std': 20, 'salt_prob': 0.01, 'pepper_prob': 0.01}
    
    if verbose:
        print("\n" + "=" * 60)
        print("ÉTAPE 8 : ÉVALUATION DU VAE DÉBRUITEUR")
        print("=" * 60)
        print(f"Type de bruit: {noise_type}")
        print(f"Paramètres: {noise_params}")
        print("=" * 60)
    
    encoder.eval()
    decoder.eval()
    
    # Sélectionner n_samples images aléatoires
    indices = np.random.choice(len(x_test), size=min(n_samples, len(x_test)), replace=False)
    x_samples = x_test[indices]
    
    # Ajouter du bruit
    x_noisy = add_noise_to_images(x_samples, noise_type=noise_type, verbose=False, **noise_params)
    
    # Débruiter
    x_noisy_tensor = torch.FloatTensor(x_noisy).permute(0, 3, 1, 2) / 255.0
    x_noisy_tensor = x_noisy_tensor.to(device)
    
    with torch.no_grad():
        # Encodage
        mu, logvar = encoder(x_noisy_tensor)
        
        # Utiliser mu pour l'inférence (sans sampling)
        z = mu
        
        # Décodage
        x_recon_tensor = decoder(z)
    
    # Convertir en numpy et clipper pour être sûr
    x_denoised = (x_recon_tensor.permute(0, 2, 3, 1).cpu().numpy() * 255.0)
    x_denoised = np.clip(x_denoised, 0, 255).astype(np.uint8)
    
    # Calculer les métriques
    mse_noisy_vs_clean = calculate_mse(x_noisy, x_samples)
    mse_denoised_vs_clean = calculate_mse(x_denoised, x_samples)
    
    psnr_noisy_vs_clean = calculate_psnr(x_noisy, x_samples)
    psnr_denoised_vs_clean = calculate_psnr(x_denoised, x_samples)
    
    if verbose:
        print(f"\nMÉTRIQUES ({n_samples} images):")
        print("-" * 60)
        print(f"{'Comparaison':<30s} {'MSE':<15s} {'PSNR (dB)':<15s}")
        print("-" * 60)
        print(f"{'Image bruitée vs propre':<30s} {mse_noisy_vs_clean:<15.2f} {psnr_noisy_vs_clean:<15.2f}")
        print(f"{'Image débruitée vs propre':<30s} {mse_denoised_vs_clean:<15.2f} {psnr_denoised_vs_clean:<15.2f}")
        print("-" * 60)
        print(f"Amélioration MSE: {mse_noisy_vs_clean - mse_denoised_vs_clean:.2f} (plus faible = mieux)")
        print(f"Amélioration PSNR: {psnr_denoised_vs_clean - psnr_noisy_vs_clean:.2f} dB (plus élevé = mieux)")
        print("-" * 60)
    
    # VISUALISATION avec ImageVisualizer
    ImageVisualizer.visualize_vae_results(
        clean_images=x_samples,
        noisy_images=x_noisy,
        denoised_images=x_denoised,
        labels=y_test,
        class_names=class_names,
        indices=indices,
        noise_type=noise_type,
        calculate_mse_func=calculate_mse,
        calculate_psnr_func=calculate_psnr,
        n_samples=n_samples
    )
    
    if verbose:
        print("\n" + "=" * 60)
        print("ÉVALUATION TERMINÉE ✔")
        print("=" * 60)
    
    # Retourner les métriques
    metrics = {
        'mse_noisy_vs_clean': mse_noisy_vs_clean,
        'mse_denoised_vs_clean': mse_denoised_vs_clean,
        'psnr_noisy_vs_clean': psnr_noisy_vs_clean,
        'psnr_denoised_vs_clean': psnr_denoised_vs_clean,
        'mse_improvement': mse_noisy_vs_clean - mse_denoised_vs_clean,
        'psnr_improvement': psnr_denoised_vs_clean - psnr_noisy_vs_clean
    }
    
    return metrics


def plot_training_history(history):
    """
    Affiche l'évolution des pertes pendant l'entraînement
    
    Args:
        history (dict): Historique d'entraînement
    """
    if not history['epochs']:
        print("Aucun historique disponible.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.ravel()

    has_val = bool(history.get('val_loss')) and len(history['val_loss']) == len(history['train_loss'])

    # Perte totale (train vs val)
    axes[0].plot(history['epochs'], history['train_loss'], 'b-', linewidth=2, label='Train')
    if has_val:
        axes[0].plot(history['epochs'], history['val_loss'], 'r--', linewidth=2, label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Perte totale')
    axes[0].set_title('Évolution de la perte totale')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Diagnostic de surapprentissage
    if has_val:
        gap = np.array(history['val_loss']) - np.array(history['train_loss'])
        axes[1].plot(history['epochs'], gap, color='crimson', linewidth=2)
        axes[1].fill_between(history['epochs'], gap, 0, color='crimson', alpha=0.2)
        axes[1].axhline(0, color='black', linewidth=1, linestyle='--')
        axes[1].set_title('Écart validation - train')
        axes[1].set_ylabel('Δ Loss (val - train)')
        axes[1].set_xlabel('Epoch')
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "Pas de validation pour diagnostic",
                     ha='center', va='center', fontsize=12, color='gray')
        axes[1].set_axis_off()

    # Perte de reconstruction
    axes[2].plot(history['epochs'], history['train_recon_loss'], 'g-', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Perte de reconstruction')
    axes[2].set_title('Évolution de la reconstruction loss')
    axes[2].grid(True, alpha=0.3)

    # KL divergence
    axes[3].plot(history['epochs'], history['train_kl_loss'], 'orange', linewidth=2)
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('KL divergence')
    axes[3].set_title('Évolution de la KL divergence')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def save_model(encoder, decoder, history, filepath='./code/vae_denoiser.pth'):
    """
    Sauvegarde le modèle entraîné
    
    Args:
        encoder (nn.Module): Encodeur
        decoder (nn.Module): Décodeur
        history (dict): Historique d'entraînement
        filepath (str): Chemin du fichier
    """
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'history': history
    }, filepath)
    print(f"Modèle sauvegardé: {filepath}")


def load_model(encoder, decoder, filepath='./code/vae_denoiser.pth', device=None):
    """
    Charge un modèle sauvegardé
    
    Args:
        encoder (nn.Module): Encodeur (structure vide)
        decoder (nn.Module): Décodeur (structure vide)
        filepath (str): Chemin du fichier
        device (torch.device): Device
    
    Returns:
        tuple: (encoder, decoder, history)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(filepath, map_location=device)
    
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    history = checkpoint['history']
    
    print(f"Modèle chargé depuis: {filepath}")
    
    return encoder, decoder, history
