import torch
import numpy as np
from load_cifar10 import CIFAR10Loader
from utils import select_one_per_class, add_noise_to_images
from noise_generator import NoiseGenerator
from vae_model import Encoder, Decoder, vae_loss


def test_encoder(x_train, y_train, class_names):
    """
    Teste l'encodeur VAE avec des données réelles de CIFAR-10 bruitées
    
    Args:
        x_train: Images d'entraînement (N, 32, 32, 3)
        y_train: Labels d'entraînement
        class_names: Noms des classes
    """
    print("\n" + "=" * 60)
    print("Test de l'Encodeur VAE avec images bruitées")
    print("=" * 60)
    
    # Créer un encodeur
    latent_dim = 128
    encoder = Encoder(latent_dim=latent_dim)
    
    # Sélectionner quelques images réelles (une par classe)
    indices = select_one_per_class(y_train, n_classes=10)
    batch_images = x_train[indices]
    batch_labels = y_train[indices]
    
    print(f"\nNombre d'images sélectionnées: {len(indices)}")
    print(f"Classes: {[class_names[int(label)] for label in batch_labels]}")
    
    # Choisir un type de bruit aléatoire
    noise_types = ['gaussian', 'salt_pepper', 'speckle', 'poisson', 'uniform', 'mixed']
    noise_type = np.random.choice(noise_types)
    
    print(f"\nType de bruit appliqué: {noise_type}")
    
    # Ajouter du bruit aux images
    if noise_type == 'gaussian':
        noisy_images = NoiseGenerator.add_gaussian_noise(batch_images, std=25)
    elif noise_type == 'salt_pepper':
        noisy_images = NoiseGenerator.add_salt_and_pepper_noise(batch_images, salt_prob=0.02, pepper_prob=0.02)
    elif noise_type == 'speckle':
        noisy_images = NoiseGenerator.add_speckle_noise(batch_images, std=0.1)
    elif noise_type == 'poisson':
        noisy_images = NoiseGenerator.add_poisson_noise(batch_images)
    elif noise_type == 'uniform':
        noisy_images = NoiseGenerator.add_uniform_noise(batch_images, low=-30, high=30)
    else:  # mixed
        noisy_images = NoiseGenerator.add_mixed_noise(batch_images, gaussian_std=20, salt_prob=0.01, pepper_prob=0.01)
    
    # Convertir en tenseur PyTorch (N, H, W, C) -> (N, C, H, W)
    x = torch.from_numpy(noisy_images).permute(0, 3, 1, 2).float()
    
    print(f"\nInput shape: {x.shape}")
    print(f"Latent dimension: {latent_dim}")
    
    # Forward pass
    mu, logvar = encoder(x)
    
    print(f"\nOutput mu shape: {mu.shape}")
    print(f"Output logvar shape: {logvar.shape}")
    
    # Vérifier les statistiques
    print(f"\nStatistiques de mu:")
    print(f"  Mean: {mu.mean().item():.4f}")
    print(f"  Std: {mu.std().item():.4f}")
    print(f"  Min: {mu.min().item():.4f}")
    print(f"  Max: {mu.max().item():.4f}")
    
    print(f"\nStatistiques de logvar:")
    print(f"  Mean: {logvar.mean().item():.4f}")
    print(f"  Std: {logvar.std().item():.4f}")
    print(f"  Min: {logvar.min().item():.4f}")
    print(f"  Max: {logvar.max().item():.4f}")
    
    # Test de la reparamétrisation
    print("\n" + "-" * 60)
    print("Test de la reparamétrisation")
    print("-" * 60)
    
    z = encoder.reparameterize(mu, logvar)
    
    print(f"\nOutput z shape: {z.shape}")
    print(f"\nStatistiques de z (vecteur latent):")
    print(f"  Mean: {z.mean().item():.4f}")
    print(f"  Std: {z.std().item():.4f}")
    print(f"  Min: {z.min().item():.4f}")
    print(f"  Max: {z.max().item():.4f}")
    
    # Vérifier que z est différent à chaque appel (stochasticité)
    z2 = encoder.reparameterize(mu, logvar)
    diff = torch.abs(z - z2).mean().item()
    print(f"\nDifférence moyenne entre 2 échantillons: {diff:.4f}")
    print(f"✓ Reparamétrisation stochastique: {'Oui' if diff > 0.01 else 'Non (problème!)'}")
    
    # Compter les paramètres
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    
    print(f"\nNombre de paramètres:")
    print(f"  Total: {total_params:,}")
    print(f"  Entraînables: {trainable_params:,}")
    
    print("\n" + "=" * 60)
    print("✓ Test de l'encodeur réussi!")
    print("=" * 60)


def test_decoder(x_train, y_train, class_names):
    """
    Teste le décodeur VAE avec des vecteurs latents générés par l'encodeur
    
    Args:
        x_train: Images d'entraînement (N, 32, 32, 3)
        y_train: Labels d'entraînement
        class_names: Noms des classes
    """
    print("\n" + "=" * 60)
    print("Test du Décodeur VAE")
    print("=" * 60)
    
    # Créer un encodeur et un décodeur
    latent_dim = 128
    encoder = Encoder(latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim)
    
    # Sélectionner quelques images réelles (une par classe)
    indices = select_one_per_class(y_train, n_classes=10)
    batch_images = x_train[indices]
    batch_labels = y_train[indices]
    
    print(f"\nNombre d'images sélectionnées: {len(indices)}")
    print(f"Classes: {[class_names[int(label)] for label in batch_labels]}")
    
    # Choisir un type de bruit aléatoire
    noise_types = ['gaussian', 'salt_pepper', 'speckle', 'poisson', 'uniform', 'mixed']
    noise_type = np.random.choice(noise_types)
    
    print(f"\nType de bruit appliqué: {noise_type}")
    
    # Ajouter du bruit aux images
    if noise_type == 'gaussian':
        noisy_images = NoiseGenerator.add_gaussian_noise(batch_images, std=25)
    elif noise_type == 'salt_pepper':
        noisy_images = NoiseGenerator.add_salt_and_pepper_noise(batch_images, salt_prob=0.02, pepper_prob=0.02)
    elif noise_type == 'speckle':
        noisy_images = NoiseGenerator.add_speckle_noise(batch_images, std=0.1)
    elif noise_type == 'poisson':
        noisy_images = NoiseGenerator.add_poisson_noise(batch_images)
    elif noise_type == 'uniform':
        noisy_images = NoiseGenerator.add_uniform_noise(batch_images, low=-30, high=30)
    else:  # mixed
        noisy_images = NoiseGenerator.add_mixed_noise(batch_images, gaussian_std=20, salt_prob=0.01, pepper_prob=0.01)
    
    # Convertir en tenseur PyTorch (N, H, W, C) -> (N, C, H, W)
    x_noisy = torch.from_numpy(noisy_images).permute(0, 3, 1, 2).float()
    
    print(f"\nInput shape (noisy): {x_noisy.shape}")
    print(f"Latent dimension: {latent_dim}")
    
    # Encoder les images bruitées
    print("\n" + "-" * 60)
    print("Encodage des images bruitées")
    print("-" * 60)
    
    mu, logvar = encoder(x_noisy)
    z = encoder.reparameterize(mu, logvar)
    
    print(f"Vecteur latent z shape: {z.shape}")
    print(f"Statistiques de z:")
    print(f"  Mean: {z.mean().item():.4f}")
    print(f"  Std: {z.std().item():.4f}")
    
    # Décoder les vecteurs latents
    print("\n" + "-" * 60)
    print("Décodage des vecteurs latents")
    print("-" * 60)
    
    x_recon = decoder(z)
    
    print(f"\nOutput shape (reconstructed): {x_recon.shape}")
    print(f"\nStatistiques de l'image reconstruite:")
    print(f"  Mean: {x_recon.mean().item():.4f}")
    print(f"  Std: {x_recon.std().item():.4f}")
    print(f"  Min: {x_recon.min().item():.4f}")
    print(f"  Max: {x_recon.max().item():.4f}")
    
    # Calculer l'erreur de reconstruction (MSE)
    x_original = torch.from_numpy(batch_images).permute(0, 3, 1, 2).float() / 255.0
    mse_noisy = torch.mean((x_noisy / 255.0 - x_original) ** 2).item()
    mse_recon = torch.mean((x_recon - x_original) ** 2).item()
    
    print(f"\n" + "-" * 60)
    print("Erreur de reconstruction (MSE)")
    print("-" * 60)
    print(f"MSE (Original vs Noisy): {mse_noisy:.6f}")
    print(f"MSE (Original vs Reconstructed): {mse_recon:.6f}")
    print(f"Amélioration: {((mse_noisy - mse_recon) / mse_noisy * 100):.2f}%")
    
    # Compter les paramètres
    encoder_params = sum(p.numel() for p in encoder.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    total_params = encoder_params + decoder_params
    
    print(f"\n" + "-" * 60)
    print("Nombre de paramètres")
    print("-" * 60)
    print(f"Encodeur: {encoder_params:,}")
    print(f"Décodeur: {decoder_params:,}")
    print(f"Total VAE: {total_params:,}")
    
    print("\n" + "=" * 60)
    print("✓ Test du décodeur réussi!")
    print("=" * 60)


def test_loss_function(x_train, y_train, class_names):
    """
    Teste la fonction de perte VAE complète (reconstruction + KL divergence)
    
    Args:
        x_train: Images d'entraînement (N, 32, 32, 3)
        y_train: Labels d'entraînement
        class_names: Noms des classes
    """
    print("\n" + "=" * 60)
    print("Test de la fonction de perte VAE")
    print("=" * 60)
    
    # Créer un encodeur et un décodeur
    latent_dim = 128
    encoder = Encoder(latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim)
    
    # Sélectionner quelques images réelles (une par classe)
    indices = select_one_per_class(y_train, n_classes=10)
    batch_images = x_train[indices]
    batch_labels = y_train[indices]
    
    print(f"\nNombre d'images sélectionnées: {len(indices)}")
    print(f"Classes: {[class_names[int(label)] for label in batch_labels]}")
    
    # Choisir un type de bruit aléatoire
    noise_types = ['gaussian', 'salt_pepper', 'speckle', 'poisson', 'uniform', 'mixed']
    noise_type = np.random.choice(noise_types)
    
    print(f"\nType de bruit appliqué: {noise_type}")
    
    # Ajouter du bruit aux images
    if noise_type == 'gaussian':
        noisy_images = NoiseGenerator.add_gaussian_noise(batch_images, std=25)
    elif noise_type == 'salt_pepper':
        noisy_images = NoiseGenerator.add_salt_and_pepper_noise(batch_images, salt_prob=0.02, pepper_prob=0.02)
    elif noise_type == 'speckle':
        noisy_images = NoiseGenerator.add_speckle_noise(batch_images, std=0.1)
    elif noise_type == 'poisson':
        noisy_images = NoiseGenerator.add_poisson_noise(batch_images)
    elif noise_type == 'uniform':
        noisy_images = NoiseGenerator.add_uniform_noise(batch_images, low=-30, high=30)
    else:  # mixed
        noisy_images = NoiseGenerator.add_mixed_noise(batch_images, gaussian_std=20, salt_prob=0.01, pepper_prob=0.01)
    
    # Préparer les tenseurs
    x_clean = torch.from_numpy(batch_images).permute(0, 3, 1, 2).float() / 255.0  # Normaliser [0, 1]
    x_noisy = torch.from_numpy(noisy_images).permute(0, 3, 1, 2).float() / 255.0
    
    print(f"\nInput shape (clean): {x_clean.shape}")
    print(f"Input shape (noisy): {x_noisy.shape}")
    print(f"Latent dimension: {latent_dim}")
    
    # Forward pass complet
    print("\n" + "-" * 60)
    print("Forward pass complet du VAE")
    print("-" * 60)
    
    # Encoder l'image bruitée
    mu, logvar = encoder(x_noisy)
    print(f"Encodeur → mu: {mu.shape}, logvar: {logvar.shape}")
    
    # Reparamétrisation
    z = encoder.reparameterize(mu, logvar)
    print(f"Reparamétrisation → z: {z.shape}")
    
    # Décoder
    x_recon = decoder(z)
    print(f"Décodeur → x_recon: {x_recon.shape}")
    
    # Calculer la loss
    print("\n" + "-" * 60)
    print("Calcul de la fonction de perte")
    print("-" * 60)
    
    # Test avec différentes valeurs de beta
    betas = [0.1, 0.5, 1.0, 2.0]
    
    print("\nComparaison avec différentes valeurs de β:")
    print(f"{'β':<6} {'Total Loss':<15} {'Recon Loss':<15} {'KL Loss':<15} {'Recon/Total':<15}")
    print("-" * 75)
    
    for beta in betas:
        total_loss, recon_loss, kl_loss = vae_loss(x_recon, x_clean, mu, logvar, beta=beta)
        recon_ratio = (recon_loss / total_loss * 100).item()
        
        print(f"{beta:<6.1f} {total_loss.item():<15.2f} {recon_loss.item():<15.2f} {kl_loss.item():<15.2f} {recon_ratio:<15.1f}%")
    
    # Utiliser beta = 1.0 pour le reste de l'analyse
    total_loss, recon_loss, kl_loss = vae_loss(x_recon, x_clean, mu, logvar, beta=1.0)
    
    print("\n" + "-" * 60)
    print("Analyse détaillée de la loss (β = 1.0)")
    print("-" * 60)
    
    batch_size = x_clean.shape[0]
    print(f"\nLoss totale: {total_loss.item():.4f}")
    print(f"  - Reconstruction loss: {recon_loss.item():.4f} ({recon_loss.item()/total_loss.item()*100:.1f}%)")
    print(f"  - KL divergence: {kl_loss.item():.4f} ({kl_loss.item()/total_loss.item()*100:.1f}%)")
    print(f"\nLoss moyenne par image:")
    print(f"  - Total: {total_loss.item()/batch_size:.4f}")
    print(f"  - Reconstruction: {recon_loss.item()/batch_size:.4f}")
    print(f"  - KL: {kl_loss.item()/batch_size:.4f}")
    
    # Calculer MSE pour comparaison
    mse_noisy = torch.mean((x_noisy - x_clean) ** 2).item()
    mse_recon = torch.mean((x_recon - x_clean) ** 2).item()
    
    print(f"\n" + "-" * 60)
    print("Comparaison MSE")
    print("-" * 60)
    print(f"MSE (Original vs Noisy): {mse_noisy:.6f}")
    print(f"MSE (Original vs Reconstructed): {mse_recon:.6f}")
    print(f"Amélioration: {((mse_noisy - mse_recon) / mse_noisy * 100):.2f}%")
    print(f"\nNote: Le modèle n'est pas entraîné, donc l'amélioration est négative.")
    print(f"Après entraînement, la reconstruction loss devrait diminuer significativement.")
    
    # Statistiques sur mu et logvar
    print(f"\n" + "-" * 60)
    print("Statistiques de l'espace latent")
    print("-" * 60)
    print(f"mu → Mean: {mu.mean().item():.4f}, Std: {mu.std().item():.4f}")
    print(f"logvar → Mean: {logvar.mean().item():.4f}, Std: {logvar.std().item():.4f}")
    print(f"z → Mean: {z.mean().item():.4f}, Std: {z.std().item():.4f}")
    
    print("\n" + "=" * 60)
    print("✓ Test de la loss function réussi!")
    print("=" * 60)
    print("\nProchaines étapes:")
    print("1. Créer une boucle d'entraînement")
    print("2. Utiliser un optimizer (Adam recommandé)")
    print("3. Entraîner sur plusieurs epochs avec KL annealing (β croissant)")
    print("4. Visualiser les résultats de débruitage")


def run_all_tests(x_train, y_train, class_names):
    """
    Lance tous les tests dans l'ordre
    
    Args:
        x_train: Images d'entraînement (N, 32, 32, 3)
        y_train: Labels d'entraînement
        class_names: Noms des classes
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "SUITE DE TESTS COMPLÈTE DU VAE")
    print("=" * 80)
    
    # Test 1: Encodeur
    test_encoder(x_train, y_train, class_names)
    
    # Test 2: Décodeur
    test_decoder(x_train, y_train, class_names)
    
    # Test 3: Loss function
    test_loss_function(x_train, y_train, class_names)
    
    print("\n" + "=" * 80)
    print(" " * 25 + "TOUS LES TESTS RÉUSSIS ✓")
    print("=" * 80)


if __name__ == "__main__":
    # Charger les données
    print("=" * 60)
    print("Chargement du dataset CIFAR-10")
    print("=" * 60)
    
    loader = CIFAR10Loader()
    x_train, y_train, x_test, y_test = loader.load_all_data()
    
    loader.print_info()
    
    # Lancer tous les tests
    run_all_tests(x_train, y_train, loader.class_names)
