import torch
from vae_model import Encoder, Decoder
from vae_train import train_vae, evaluate_vae, plot_training_history, save_model
from load_cifar10 import CIFAR10Loader


def config_baseline():
    """Configuration de base """
    return {
        'epochs': 30,
        'batch_size': 128,
        'learning_rate': 1e-3,
        'beta': 1.0,
        'noise_type': 'gaussian',
        'noise_params': {'std': 25}
    }


def config_improved():
    """Configuration améliorée - Beta réduit + plus d'epochs"""
    return {
        'epochs': 50,
        'batch_size': 128,
        'learning_rate': 1e-3,
        'beta': 0.5,  # Réduit pour favoriser la reconstruction
        'noise_type': 'gaussian',
        'noise_params': {'std': 25}
    }


def config_aggressive():
    """Configuration agressive - Beta très faible + beaucoup d'epochs"""
    return {
        'epochs': 100,
        'batch_size': 128,
        'learning_rate': 1e-3,
        'beta': 0.1,  # Très faible pour se concentrer sur la reconstruction
        'noise_type': 'gaussian',
        'noise_params': {'std': 25}
    }


def config_light_noise():
    """Configuration avec bruit plus léger"""
    return {
        'epochs': 50,
        'batch_size': 128,
        'learning_rate': 1e-3,
        'beta': 0.5,
        'noise_type': 'gaussian',
        'noise_params': {'std': 15}  # Bruit plus faible
    }


def config_large_batch():
    """Configuration avec batch plus large"""
    return {
        'epochs': 50,
        'batch_size': 256,  # Batch plus large
        'learning_rate': 2e-3,  # Learning rate augmenté
        'beta': 0.5,
        'noise_type': 'gaussian',
        'noise_params': {'std': 25}
    }


def train_with_config(config_name='improved'):
    """
    Entraîne le VAE avec une configuration spécifique
    
    Args:
        config_name (str): 'baseline', 'improved', 'aggressive', 'light_noise', 'large_batch'
    """
    print("=" * 60)
    print(f"ENTRAÎNEMENT AVEC CONFIGURATION: {config_name.upper()}")
    print("=" * 60)
    
    # Charger les données
    print("\nChargement des données CIFAR-10...")
    loader = CIFAR10Loader()
    x_train, y_train, x_test, y_test = loader.load_all_data()
    
    # Sélectionner la configuration
    configs = {
        'baseline': config_baseline(),
        'improved': config_improved(),
        'aggressive': config_aggressive(),
        'light_noise': config_light_noise(),
        'large_batch': config_large_batch()
    }
    
    if config_name not in configs:
        print(f"Configuration inconnue: {config_name}")
        print(f"Configurations disponibles: {list(configs.keys())}")
        return
    
    config = configs[config_name]
    
    # Afficher la configuration
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Initialiser le VAE
    print("\nInitialisation du VAE...")
    latent_dim = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = Encoder(latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim)
    
    # Entraîner
    history = train_vae(
        encoder=encoder,
        decoder=decoder,
        x_train=x_train,
        device=device,
        validation_split=0.1,
        verbose=True,
        **config
    )
    
    # Sauvegarder
    model_path = f'vae_denoiser_{config_name}.pth'
    save_model(encoder, decoder, history, filepath=model_path)
    
    # Afficher l'historique
    plot_training_history(history)
    
    # Évaluer
    print("\n" + "=" * 60)
    print("ÉVALUATION")
    print("=" * 60)
    
    metrics = evaluate_vae(
        encoder=encoder,
        decoder=decoder,
        x_test=x_test,
        y_test=y_test,
        class_names=loader.class_names,
        noise_type=config['noise_type'],
        noise_params=config['noise_params'],
        n_samples=5,
        device=device,
        verbose=True
    )
    
    # Résumé
    print("\n" + "=" * 60)
    print("RÉSUMÉ DES PERFORMANCES")
    print("=" * 60)
    print(f"Configuration: {config_name}")
    print(f"MSE avant: {metrics['mse_noisy_vs_clean']:.2f}")
    print(f"MSE après: {metrics['mse_denoised_vs_clean']:.2f}")
    print(f"Amélioration MSE: {metrics['mse_improvement']:.2f}")
    print(f"PSNR avant: {metrics['psnr_noisy_vs_clean']:.2f} dB")
    print(f"PSNR après: {metrics['psnr_denoised_vs_clean']:.2f} dB")
    print(f"Amélioration PSNR: {metrics['psnr_improvement']:.2f} dB")
    print("=" * 60)
    
    if metrics['mse_improvement'] > 0:
        print("✅ Le modèle AMÉLIORE les images!")
    else:
        print("❌ Le modèle EMPIRE les images - essayez une autre configuration")
    
    return encoder, decoder, history, metrics


if __name__ == "__main__":
    import sys
    
    # Permettre de spécifier la configuration en argument
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
    else:
        # Par défaut, utiliser la configuration améliorée
        config_name = 'improved'
    
    print(f"\nUtilisation de la configuration: {config_name}")
    print("\nConfigurations disponibles:")
    print("  - baseline: Configuration de base (epochs=30, beta=1.0)")
    print("  - improved: Configuration améliorée (epochs=50, beta=0.5)")
    print("  - aggressive: Configuration agressive (epochs=100, beta=0.1)")
    print("  - light_noise: Bruit plus léger (std=15 au lieu de 25)")
    print("  - large_batch: Batch plus large (256 au lieu de 128)")
    print("\nUtilisation: python vae_configs.py [config_name]\n")
    
    encoder, decoder, history, metrics = train_with_config(config_name)
