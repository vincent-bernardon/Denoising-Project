"""
Comparaison globale : VAE vs U-Net
Affiche les graphiques PSNR et MSE pour comparer les performances
"""
import torch
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from VAE.vae_model import Encoder, Decoder
from VAE.vae_model_96 import Encoder96, Decoder96
from VAE.vae_train import load_model, calculate_psnr, calculate_mse
from UNET.unet_model import UNet
from utils import add_noise_to_images
from patch_utils import denoise_with_patches
from patch_utils import denoise_with_patches


def load_vae_model(model_path, device, latent_dim=128, is_stl10=False):
    """Charge un modèle VAE avec la bonne architecture selon le dataset"""
    if is_stl10:
        encoder = Encoder96(latent_dim=latent_dim)
        decoder = Decoder96(latent_dim=latent_dim)
    else:
        encoder = Encoder(latent_dim=latent_dim)
        decoder = Decoder(latent_dim=latent_dim)
    
    encoder, decoder, _ = load_model(encoder, decoder, model_path, device)
    return encoder, decoder


def load_unet_model(model_path, device):
    """Charge un modèle U-Net"""
    model = UNet(n_channels=3, n_classes=3, base_features=64)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def find_model_paths(dataset_name):
    """
    Trouve automatiquement les chemins des modèles en fonction du dataset
    
    Args:
        dataset_name: 'cifar10' ou 'stl10'
    
    Returns:
        dict: Chemins des modèles VAE et U-Net
    """
    code_dir = Path('./code')
    
    # Chercher les modèles VAE
    if dataset_name == 'cifar10':
        # Pour CIFAR-10, utiliser vae_denoiser_beta0.pth
        vae_path = code_dir / 'vae_denoiser_beta0.pth'
    else:
        # Pour STL-10, chercher vae_denoiser_stl96.pth
        vae_path = code_dir / 'vae_denoiser_stl96.pth'
    
    # Chercher les modèles U-Net (avec ou sans GAN)
    unet_patterns = [
        f'unet_denoising_{dataset_name}_multinoise.pth',
        f'gan_unet_denoising_{dataset_name}_multinoise.pth'
    ]
    
    unet_paths = []
    for pattern in unet_patterns:
        path = code_dir / pattern
        if path.exists():
            unet_paths.append(path)
    
    return {
        'vae': vae_path if vae_path.exists() else None,
        'unet': unet_paths[0] if unet_paths else None,
        'unet_gan': unet_paths[1] if len(unet_paths) > 1 else None
    }


def denoise_with_vae(encoder, decoder, x_noisy_tensor, device):
    """Débruite avec le VAE"""
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        x_noisy_tensor = x_noisy_tensor.to(device)
        mu, logvar = encoder(x_noisy_tensor)
        z = mu  # Utiliser mu directement
        x_recon = decoder(z)
    
    return x_recon


def denoise_with_unet(model, x_noisy_tensor, device):
    """Débruite avec le U-Net"""
    model.eval()
    
    with torch.no_grad():
        x_noisy_tensor = x_noisy_tensor.to(device)
        x_recon = model(x_noisy_tensor)
    
    return x_recon


def evaluate_models(vae_encoder, vae_decoder, unet_model, x_data, 
                    noise_type, noise_params, device, n_samples=1000, is_stl10=False):
    """
    Évalue les deux modèles sur le même ensemble d'images bruitées
    
    Returns:
        dict: Métriques pour chaque modèle
    """
    # Limiter le nombre d'échantillons
    if len(x_data) > n_samples:
        indices = np.random.choice(len(x_data), n_samples, replace=False)
        x_data = x_data[indices]
    
    # Convertir en uint8 [0, 255]
    x_samples = (x_data * 255).astype(np.uint8)
    
    # Générer les images bruitées
    x_noisy = add_noise_to_images(x_samples, noise_type=noise_type, **noise_params)
    
    # Convertir en tensor [0, 1]
    x_noisy_tensor = torch.FloatTensor(x_noisy).permute(0, 3, 1, 2) / 255.0
    
    # Débruiter avec VAE (utilise la taille native: 96x96 pour STL-10, 32x32 pour CIFAR-10)
    x_vae_recon = denoise_with_vae(vae_encoder, vae_decoder, x_noisy_tensor, device)
    x_vae_denoised = (x_vae_recon.permute(0, 2, 3, 1).cpu().numpy() * 255.0)
    x_vae_denoised = np.clip(x_vae_denoised, 0, 255).astype(np.uint8)
    
    # Débruiter avec U-Net en utilisant le système de patches pour STL-10 96x96
    x_unet_denoised_list = []
    for i in range(x_noisy_tensor.shape[0]):
        img_denoised = denoise_with_patches(
            unet_model, x_noisy_tensor[i], device, 
            patch_size=32, stride=16
        )
        x_unet_denoised_list.append(img_denoised)
    
    x_unet_recon = torch.stack(x_unet_denoised_list)
    x_unet_denoised = (x_unet_recon.permute(0, 2, 3, 1).cpu().numpy() * 255.0)
    x_unet_denoised = np.clip(x_unet_denoised, 0, 255).astype(np.uint8)
    
    # Calculer les métriques
    psnr_noisy = []
    psnr_vae = []
    psnr_unet = []
    mse_noisy = []
    mse_vae = []
    mse_unet = []
    
    for i in range(len(x_samples)):
        # PSNR
        psnr_noisy.append(calculate_psnr(x_noisy[i], x_samples[i], max_pixel_value=255.0))
        psnr_vae.append(calculate_psnr(x_vae_denoised[i], x_samples[i], max_pixel_value=255.0))
        psnr_unet.append(calculate_psnr(x_unet_denoised[i], x_samples[i], max_pixel_value=255.0))
        
        # MSE
        mse_noisy.append(calculate_mse(x_noisy[i], x_samples[i]))
        mse_vae.append(calculate_mse(x_vae_denoised[i], x_samples[i]))
        mse_unet.append(calculate_mse(x_unet_denoised[i], x_samples[i]))
    
    return {
        'psnr_noisy': np.mean(psnr_noisy),
        'psnr_vae': np.mean(psnr_vae),
        'psnr_unet': np.mean(psnr_unet),
        'mse_noisy': np.mean(mse_noisy),
        'mse_vae': np.mean(mse_vae),
        'mse_unet': np.mean(mse_unet),
        'n_samples': len(x_samples)
    }


def plot_comparison(results, dataset_name, save_path=None):
    """
    Crée les graphiques de comparaison PSNR et MSE
    
    Args:
        results: Liste de dictionnaires avec les résultats par type de bruit
        dataset_name: Nom du dataset
        save_path: Chemin de sauvegarde (optionnel)
    """
    noise_types = [r['name'] for r in results]
    
    # Extraire les données
    psnr_noisy = [r['metrics']['psnr_noisy'] for r in results]
    psnr_vae = [r['metrics']['psnr_vae'] for r in results]
    psnr_unet = [r['metrics']['psnr_unet'] for r in results]
    
    mse_noisy = [r['metrics']['mse_noisy'] for r in results]
    mse_vae = [r['metrics']['mse_vae'] for r in results]
    mse_unet = [r['metrics']['mse_unet'] for r in results]
    
    # Créer la figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f'Comparaison VAE vs U-Net - {dataset_name.upper()}', 
                 fontsize=16, fontweight='bold')
    
    # Position des barres
    x = np.arange(len(noise_types))
    width = 0.25
    
    # ========== GRAPHIQUE 1 : PSNR ==========
    bars1 = ax1.bar(x - width, psnr_noisy, width, label='Images bruitées', 
                    color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x, psnr_vae, width, label='VAE débruité', 
                    color='#3498db', alpha=0.85, edgecolor='black', linewidth=1.5)
    bars3 = ax1.bar(x + width, psnr_unet, width, label='U-Net débruité', 
                    color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=1.5)
    
    # Ajouter les valeurs sur les barres
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Configuration PSNR
    ax1.set_xlabel('Type de bruit', fontsize=13, fontweight='bold')
    ax1.set_ylabel('PSNR (dB)', fontsize=13, fontweight='bold')
    ax1.set_title('PSNR moyen : Comparaison des méthodes', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(noise_types, fontsize=11)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
    
    all_psnr = psnr_noisy + psnr_vae + psnr_unet
    ax1.set_ylim(bottom=min(all_psnr) * 0.92, top=max(all_psnr) * 1.08)
    
    # ========== GRAPHIQUE 2 : MSE ==========
    bars4 = ax2.bar(x - width, mse_noisy, width, label='Images bruitées', 
                    color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.5)
    bars5 = ax2.bar(x, mse_vae, width, label='VAE débruité', 
                    color='#3498db', alpha=0.85, edgecolor='black', linewidth=1.5)
    bars6 = ax2.bar(x + width, mse_unet, width, label='U-Net débruité', 
                    color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=1.5)
    
    # Ajouter les valeurs sur les barres
    for bars in [bars4, bars5, bars6]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Configuration MSE
    ax2.set_xlabel('Type de bruit', fontsize=13, fontweight='bold')
    ax2.set_ylabel('MSE', fontsize=13, fontweight='bold')
    ax2.set_title('MSE moyen : Comparaison des méthodes', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(noise_types, fontsize=11)
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
    
    all_mse = mse_noisy + mse_vae + mse_unet
    ax2.set_ylim(bottom=0, top=max(all_mse) * 1.15)
    
    plt.tight_layout()
    
    # Sauvegarder
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Graphique sauvegardé: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Comparaison VAE vs U-Net')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar10', 'stl10'],
                        help='Dataset à utiliser')
    parser.add_argument('--n-samples', type=int, default=100,
                        help='Nombre d\'images à évaluer (défaut: 100)')
    parser.add_argument('--vae-model', type=str, default=None,
                        help='Chemin vers le modèle VAE (auto-détecté si non spécifié)')
    parser.add_argument('--unet-model', type=str, default=None,
                        help='Chemin vers le modèle U-Net (auto-détecté si non spécifié)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("COMPARAISON VAE vs U-Net")
    print("=" * 80)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Device: {device}")
    
    # Trouver les modèles automatiquement
    print(f"\n📂 Recherche des modèles pour {args.dataset.upper()}...")
    model_paths = find_model_paths(args.dataset)
    
    # Utiliser les chemins fournis ou auto-détectés
    vae_path = args.vae_model if args.vae_model else model_paths['vae']
    unet_path = args.unet_model if args.unet_model else (model_paths['unet_gan'] or model_paths['unet'])
    
    print(f"\nModèles sélectionnés:")
    print(f"  - VAE:   {vae_path}")
    print(f"  - U-Net: {unet_path}")
    
    # Vérifier l'existence des modèles
    if not vae_path or not Path(vae_path).exists():
        print(f"\n❌ Erreur: Modèle VAE non trouvé")
        print(f"   Attendu: ./code/vae_denoiser_beta0.pth (cifar10)")
        print(f"            ./code/vae_denoiser_stl96.pth (stl10)")
        return
    
    if not unet_path or not Path(unet_path).exists():
        print(f"\n❌ Erreur: Modèle U-Net non trouvé")
        print(f"   Attendu: ./code/unet_denoising_{args.dataset}_multinoise.pth")
        print(f"         ou ./code/gan_unet_denoising_{args.dataset}_multinoise.pth")
        return
    
    # Charger les modèles
    print(f"\n📦 Chargement des modèles...")
    vae_encoder, vae_decoder = load_vae_model(vae_path, device, is_stl10=(args.dataset == 'stl10'))
    print("  ✓ VAE chargé")
    
    unet_model = load_unet_model(unet_path, device)
    print("  ✓ U-Net chargé")
    
    # Charger le dataset
    print(f"\n📦 Chargement de {args.dataset.upper()}...")
    
    if args.dataset == 'cifar10':
        data_dir = './code/code'
        test_dataset = torchvision.datasets.CIFAR10(
            data_dir, train=False, download=False, transform=transforms.ToTensor()
        )
    else:
        data_dir = './code/code/stl10'
        test_dataset = torchvision.datasets.STL10(
            data_dir, split='test', download=False, transform=transforms.ToTensor()
        )
    
    # Convertir en numpy
    x_test = []
    for i in range(len(test_dataset)):
        img, _ = test_dataset[i]
        # Pour STL-10 avec VAE: garder 96x96, mais redimensionner à 32x32 pour U-Net
        # On va traiter séparément dans evaluate_models
        x_test.append(img.permute(1, 2, 0).numpy())
    
    x_test = np.array(x_test)
    print(f"  ✓ {len(x_test)} images chargées ({x_test.shape[1]}x{x_test.shape[2]})")
    
    # Configuration des bruits
    noise_configs = [
        {'name': 'Gaussien', 'type': 'gaussian', 'params': {'std': 25}},
        {'name': 'Salt & Pepper', 'type': 'salt_pepper', 
         'params': {'salt_prob': 0.02, 'pepper_prob': 0.02}},
        {'name': 'Mixte', 'type': 'mixed', 
         'params': {'gaussian_std': 20, 'salt_prob': 0.01, 'pepper_prob': 0.01}}
    ]
    
    # Évaluer pour chaque type de bruit
    results = []
    
    print(f"\n{'='*80}")
    print("ÉVALUATION DES MODÈLES")
    print(f"{'='*80}")
    
    for config in noise_configs:
        print(f"\n⏳ {config['name']}... ", end='', flush=True)
        
        metrics = evaluate_models(
            vae_encoder, vae_decoder, unet_model,
            x_test, config['type'], config['params'],
            device, n_samples=args.n_samples, is_stl10=(args.dataset == 'stl10')
        )
        
        results.append({
            'name': config['name'],
            'metrics': metrics
        })
        
        print("✓")
        print(f"   PSNR: Noisy={metrics['psnr_noisy']:.2f} | "
              f"VAE={metrics['psnr_vae']:.2f} | U-Net={metrics['psnr_unet']:.2f}")
        print(f"   MSE:  Noisy={metrics['mse_noisy']:.4f} | "
              f"VAE={metrics['mse_vae']:.4f} | U-Net={metrics['mse_unet']:.4f}")
    
    # Afficher le tableau comparatif
    print(f"\n{'='*80}")
    print("TABLEAU COMPARATIF")
    print(f"{'='*80}")
    
    print(f"\n{'Type de bruit':<20s} {'Méthode':<12s} {'PSNR (dB)':<12s} {'MSE':<12s} {'Gain PSNR':<12s}")
    print("-" * 80)
    
    for result in results:
        m = result['metrics']
        print(f"{result['name']:<20s} {'Bruitée':<12s} {m['psnr_noisy']:>10.2f}   {m['mse_noisy']:>10.4f}   {'-':<12s}")
        
        gain_vae = m['psnr_vae'] - m['psnr_noisy']
        print(f"{'':20s} {'VAE':<12s} {m['psnr_vae']:>10.2f}   {m['mse_vae']:>10.4f}   {gain_vae:>+10.2f}")
        
        gain_unet = m['psnr_unet'] - m['psnr_noisy']
        winner = '🏆' if m['psnr_unet'] > m['psnr_vae'] else ''
        print(f"{'':20s} {'U-Net':<12s} {m['psnr_unet']:>10.2f}   {m['mse_unet']:>10.4f}   {gain_unet:>+10.2f} {winner}")
        print()
    
    # Calculer les moyennes globales
    avg_psnr_vae = np.mean([r['metrics']['psnr_vae'] for r in results])
    avg_psnr_unet = np.mean([r['metrics']['psnr_unet'] for r in results])
    avg_mse_vae = np.mean([r['metrics']['mse_vae'] for r in results])
    avg_mse_unet = np.mean([r['metrics']['mse_unet'] for r in results])
    
    print("-" * 80)
    print(f"\n🏆 Performances moyennes sur tous les types de bruit:")
    print(f"   VAE:   PSNR = {avg_psnr_vae:.2f} dB | MSE = {avg_mse_vae:.4f}")
    print(f"   U-Net: PSNR = {avg_psnr_unet:.2f} dB | MSE = {avg_mse_unet:.4f}")
    
    if avg_psnr_unet > avg_psnr_vae:
        diff = avg_psnr_unet - avg_psnr_vae
        print(f"\n   ✅ U-Net est meilleur de {diff:.2f} dB en moyenne")
    else:
        diff = avg_psnr_vae - avg_psnr_unet
        print(f"\n   ✅ VAE est meilleur de {diff:.2f} dB en moyenne")
    
    # Créer les graphiques
    print(f"\n{'='*80}")
    print("GÉNÉRATION DES GRAPHIQUES")
    print(f"{'='*80}")
    
    save_path = f'./code/comparison_vae_unet_{args.dataset}.png'
    plot_comparison(results, args.dataset, save_path=save_path)
    
    print(f"\n{'='*80}")
    print("COMPARAISON TERMINÉE ✓")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
