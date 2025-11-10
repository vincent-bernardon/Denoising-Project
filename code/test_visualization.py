import torch
import numpy as np
from load_cifar10 import CIFAR10Loader
from vae_model import Encoder, Decoder
from vae_train import load_model, evaluate_vae

def test_visualization():
    """Teste uniquement la visualisation avec le modèle sauvegardé"""
    
    print("=" * 60)
    print("TEST DE VISUALISATION")
    print("=" * 60)
    
    # Charger les données
    print("\nChargement des données CIFAR-10...")
    loader = CIFAR10Loader()
    x_train, y_train, x_test, y_test = loader.load_all_data()
    
    # Charger le modèle
    print("\nChargement du modèle VAE...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_dim = 128
    
    encoder = Encoder(latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim)
    
    try:
        encoder, decoder, history = load_model(encoder, decoder, './code/vae_denoiser_beta0.pth', device)
        print("✅ Modèle chargé avec succès")
    except FileNotFoundError:
        print("❌ Fichier './code/vae_denoiser.pth' introuvable")
        print("   Veuillez d'abord entraîner le modèle avec: python cnn.py")
        return
    
    # Test 1: Bruit Gaussien
    print("\n" + "=" * 60)
    print("TEST 1/3 : Bruit Gaussien (std=25)")
    print("=" * 60)
    
    evaluate_vae(
        encoder=encoder,
        decoder=decoder,
        x_test=x_test,
        y_test=y_test,
        class_names=loader.class_names,
        noise_type='gaussian',
        noise_params={'std': 25},
        n_samples=3,  # Moins d'images pour test rapide
        device=device,
        verbose=True
    )
    
    # Test 2: Bruit Salt & Pepper
    print("\n" + "=" * 60)
    print("TEST 2/3 : Bruit Salt & Pepper")
    print("=" * 60)
    
    evaluate_vae(
        encoder=encoder,
        decoder=decoder,
        x_test=x_test,
        y_test=y_test,
        class_names=loader.class_names,
        noise_type='salt_pepper',
        noise_params={'salt_prob': 0.02, 'pepper_prob': 0.02},
        n_samples=3,
        device=device,
        verbose=True
    )
    
    # Test 3: Bruit Mixte
    print("\n" + "=" * 60)
    print("TEST 3/3 : Bruit Mixte")
    print("=" * 60)
    
    evaluate_vae(
        encoder=encoder,
        decoder=decoder,
        x_test=x_test,
        y_test=y_test,
        class_names=loader.class_names,
        noise_type='mixed',
        noise_params={'gaussian_std': 20, 'salt_prob': 0.01, 'pepper_prob': 0.01},
        n_samples=3,
        device=device,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("TEST TERMINÉ ✔")
    print("=" * 60)
    print("\nSi les warnings 'Clipping input data' ont disparu, c'est bon ! ✅")


if __name__ == "__main__":
    test_visualization()
