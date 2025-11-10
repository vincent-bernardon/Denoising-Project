"""
Test rapide: VAE actuel avec BETA=0 (autoencodeur pur)
Utilise le modèle déjà entraîné mais en mode autoencodeur
"""
import torch
from load_cifar10 import CIFAR10Loader
from vae_model import Encoder, Decoder
from vae_train import train_vae, evaluate_vae, plot_training_history, save_model

def main():
    print("=" * 60)
    print("TEST RAPIDE: VAE avec BETA=0 (Autoencodeur pur)")
    print("=" * 60)
    
    # Charger données
    loader = CIFAR10Loader()
    x_train, y_train, x_test, y_test = loader.load_all_data()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Modèle actuel (6M paramètres)
    latent_dim = 128
    encoder = Encoder(latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim)
    
    print(f"✅ VAE initialisé (latent_dim={latent_dim})")
    print(f"📊 Changement: Beta 0.01 → 0.0 (autoencodeur pur)")
    print(f"⏱️  Temps estimé: 30 epochs × 50 sec = 25 minutes\n")
    
    # ENTRAÎNEMENT BETA=0
    history = train_vae(
        encoder=encoder,
        decoder=decoder,
        x_train=x_train,
        epochs=30,                      # Seulement 30 epochs
        batch_size=128,
        learning_rate=1e-3,             # Learning rate standard
        noise_type='gaussian',
        noise_params={'std': 25},
        beta=0.0,                       # BETA = 0 (autoencodeur pur)
        device=device,
        validation_split=0.1,
        verbose=True
    )
    
    # Historique
    plot_training_history(history)
    
    # Sauvegarder
    save_model(encoder, decoder, history, filepath='./code/vae_denoiser_beta0.pth')
    
    # ÉVALUATION
    print("\n" + "=" * 60)
    print("ÉVALUATION - Beta=0 vs Beta=0.01")
    print("=" * 60)
    
    # Gaussien
    print("\n🔬 Test 1/3: Bruit Gaussien")
    metrics_g = evaluate_vae(
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
    metrics_s = evaluate_vae(
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
    metrics_m = evaluate_vae(
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
    
    # COMPARAISON
    print("\n" + "=" * 60)
    print("📊 COMPARAISON Beta=0.01 vs Beta=0")
    print("=" * 60)
    
    print("\n📋 Résultats Beta=0.01 (ancien):")
    print("  Gaussien:      -0.03 dB  ❌ (empire)")
    print("  Salt & Pepper: +1.48 dB  ✅")
    print("  Mixte:         +2.70 dB  ✅")
    
    print("\n📋 Résultats Beta=0 (nouveau):")
    print(f"  Gaussien:      {metrics_g['psnr_improvement']:+.2f} dB  {'✅' if metrics_g['psnr_improvement'] > 0 else '❌'}")
    print(f"  Salt & Pepper: {metrics_s['psnr_improvement']:+.2f} dB  {'✅' if metrics_s['psnr_improvement'] > 0 else '❌'}")
    print(f"  Mixte:         {metrics_m['psnr_improvement']:+.2f} dB  {'✅' if metrics_m['psnr_improvement'] > 0 else '❌'}")
    
    # Conclusion
    print("\n" + "=" * 60)
    print("💡 CONCLUSION")
    print("=" * 60)
    
    if metrics_g['psnr_improvement'] > 0.5:
        print("\n✅ SUCCÈS! Beta=0 améliore le débruitage gaussien")
        print("   → Le problème était la régularisation KL excessive")
    elif metrics_g['psnr_improvement'] > 0:
        print("\n⚠️  AMÉLIORATION LÉGÈRE. Beta=0 aide un peu.")
        print("   → Le flou est partiellement dû à CIFAR-10 (32×32)")
    else:
        print("\n❌ PAS D'AMÉLIORATION. Le flou persiste.")
        print("   → C'est la LIMITE FONDAMENTALE de CIFAR-10 (32×32 trop petit)")
        print("   → Le VAE ne peut pas inventer des détails qui n'existent pas")
        print("\n💡 Solutions:")
        print("   1. Utiliser des images plus grandes (256×256+)")
        print("   2. Utiliser un U-Net avec skip connections")
        print("   3. Accepter que 32×32 = détails limités")
    
    return encoder, decoder, history


if __name__ == "__main__":
    encoder, decoder, history = main()
