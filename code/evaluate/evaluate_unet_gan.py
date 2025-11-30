"""
Comparaison globale : U-Net vs U-Net+GAN
Affiche les graphiques PSNR et MSE pour comparer les performances
(Actuellement disponible uniquement pour STL-10)
"""
import torch
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from UNET.unet_model import UNet
from utils import add_noise_to_images
from patch_utils import denoise_with_patches


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
        dict: Chemins des modèles U-Net et U-Net+GAN
    """
    code_dir = Path('./code')
    
    # Chercher les modèles U-Net
    unet_path = code_dir / f'unet_denoising_{dataset_name}_multinoise.pth'
    gan_path = code_dir / f'gan_unet_denoising_{dataset_name}_multinoise.pth'
    
    return {
        'unet': unet_path if unet_path.exists() else None,
        'unet_gan': gan_path if gan_path.exists() else None
    }


def calculate_psnr(img1, img2, max_pixel_value=255.0):
    """Calcule le PSNR entre deux images"""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_pixel_value / np.sqrt(mse))


def calculate_mse(img1, img2):
    """Calcule le MSE entre deux images"""
    return np.mean((img1.astype(float) - img2.astype(float)) ** 2)


def evaluate_models(unet_model, gan_model, x_data, 
                    noise_type, noise_params, device, n_samples=100, vif_func=None, dists_model=None):
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
    
    # Débruiter avec U-Net classique
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
    
    # Débruiter avec U-Net+GAN
    x_gan_denoised_list = []
    for i in range(x_noisy_tensor.shape[0]):
        img_denoised = denoise_with_patches(
            gan_model, x_noisy_tensor[i], device, 
            patch_size=32, stride=16
        )
        x_gan_denoised_list.append(img_denoised)
    
    x_gan_recon = torch.stack(x_gan_denoised_list)
    x_gan_denoised = (x_gan_recon.permute(0, 2, 3, 1).cpu().numpy() * 255.0)
    x_gan_denoised = np.clip(x_gan_denoised, 0, 255).astype(np.uint8)
    
    # Calculer les métriques
    psnr_noisy = []
    psnr_unet = []
    psnr_gan = []
    mse_noisy = []
    mse_unet = []
    mse_gan = []
    
    for i in range(len(x_samples)):
        # PSNR
        psnr_noisy.append(calculate_psnr(x_noisy[i], x_samples[i], max_pixel_value=255.0))
        psnr_unet.append(calculate_psnr(x_unet_denoised[i], x_samples[i], max_pixel_value=255.0))
        psnr_gan.append(calculate_psnr(x_gan_denoised[i], x_samples[i], max_pixel_value=255.0))
        
        # MSE
        mse_noisy.append(calculate_mse(x_noisy[i], x_samples[i]))
        mse_unet.append(calculate_mse(x_unet_denoised[i], x_samples[i]))
        mse_gan.append(calculate_mse(x_gan_denoised[i], x_samples[i]))
    
    # VIF
    vif_noisy = []
    vif_unet = []
    vif_gan = []
    if vif_func is not None:
        # sewar vifp expects 2D grayscale images in uint8 or float
        def to_gray_uint8(img_arr):
            # img_arr: HxWx3 uint8
            if img_arr.ndim == 3 and img_arr.shape[2] == 3:
                gray = (0.299 * img_arr[:,:,0] + 0.587 * img_arr[:,:,1] + 0.114 * img_arr[:,:,2])
            else:
                gray = img_arr[:,:,0]
            return gray.astype(np.uint8)
        for i in range(len(x_samples)):
            gt = to_gray_uint8(x_samples[i])
            noisy = to_gray_uint8(x_noisy[i])
            unet = to_gray_uint8(x_unet_denoised[i])
            gan = to_gray_uint8(x_gan_denoised[i])
            try:
                vif_noisy.append(float(vif_func(gt, noisy)))
                vif_unet.append(float(vif_func(gt, unet)))
                vif_gan.append(float(vif_func(gt, gan)))
            except Exception:
                # fall back to None if computation fails
                vif_noisy.append(None)
                vif_unet.append(None)
                vif_gan.append(None)
    # DISTS
    dists_noisy = []
    dists_unet = []
    dists_gan = []
    if dists_model is not None:
        import torchvision.transforms as T
        def prep_for_dists(img_arr):
            pil = T.ToPILImage()(img_arr.astype(np.uint8))
            proc = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor()
            ])
            return proc(pil).unsqueeze(0).to(device)
        for i in range(len(x_samples)):
            gt = prep_for_dists(x_samples[i])
            noisy = prep_for_dists(x_noisy[i])
            unet = prep_for_dists(x_unet_denoised[i])
            gan = prep_for_dists(x_gan_denoised[i])
            with torch.no_grad():
                dists_noisy.append(float(dists_model(gt, noisy).cpu().item()))
                dists_unet.append(float(dists_model(gt, unet).cpu().item()))
                dists_gan.append(float(dists_model(gt, gan).cpu().item()))
    return {
        'psnr_noisy': np.mean(psnr_noisy),
        'psnr_unet': np.mean(psnr_unet),
        'psnr_gan': np.mean(psnr_gan),
        'mse_noisy': np.mean(mse_noisy),
        'mse_unet': np.mean(mse_unet),
        'mse_gan': np.mean(mse_gan),
        'n_samples': len(x_samples),
        'vif_noisy': np.mean([v for v in vif_noisy if v is not None]) if vif_noisy else None,
        'vif_unet': np.mean([v for v in vif_unet if v is not None]) if vif_unet else None,
        'vif_gan': np.mean([v for v in vif_gan if v is not None]) if vif_gan else None,
        'dists_noisy': np.mean(dists_noisy) if dists_noisy else None,
        'dists_unet': np.mean(dists_unet) if dists_unet else None,
        'dists_gan': np.mean(dists_gan) if dists_gan else None
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
    psnr_unet = [r['metrics']['psnr_unet'] for r in results]
    psnr_gan = [r['metrics']['psnr_gan'] for r in results]

    mse_noisy = [r['metrics']['mse_noisy'] for r in results]
    mse_unet = [r['metrics']['mse_unet'] for r in results]
    mse_gan = [r['metrics']['mse_gan'] for r in results]

    # VIF (si présent)
    vif_noisy = [r['metrics'].get('vif_noisy', None) for r in results]
    vif_unet = [r['metrics'].get('vif_unet', None) for r in results]
    vif_gan = [r['metrics'].get('vif_gan', None) for r in results]

    # DISTS (si présent)
    dists_noisy = [r['metrics'].get('dists_noisy', None) for r in results]
    dists_unet = [r['metrics'].get('dists_unet', None) for r in results]
    dists_gan = [r['metrics'].get('dists_gan', None) for r in results]

    # Créer la figure avec une grille 2x2 de graphiques
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    ax1, ax2, ax3, ax4 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]
    fig.suptitle(f'Comparaison U-Net vs U-Net+GAN - {dataset_name.upper()}', 
                 fontsize=16, fontweight='bold')
    
    # Position des barres
    x = np.arange(len(noise_types))
    width = 0.25
    
    # ========== GRAPHIQUE 1 : PSNR ==========
    bars1 = ax1.bar(x - width, psnr_noisy, width, label='Images bruitées', 
                    color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x, psnr_unet, width, label='U-Net débruité', 
                    color='#3498db', alpha=0.85, edgecolor='black', linewidth=1.5)
    bars3 = ax1.bar(x + width, psnr_gan, width, label='U-Net+GAN débruité', 
                    color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=1.5)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_xlabel('Type de bruit', fontsize=13, fontweight='bold')
    ax1.set_ylabel('PSNR (dB)', fontsize=13, fontweight='bold')
    ax1.set_title('PSNR moyen : Comparaison des méthodes', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(noise_types, fontsize=11)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)

    all_psnr = psnr_noisy + psnr_unet + psnr_gan
    ax1.set_ylim(bottom=min(all_psnr) * 0.92, top=max(all_psnr) * 1.08)

    # ========== GRAPHIQUE 2 : MSE ==========
    bars4 = ax2.bar(x - width, mse_noisy, width, label='Images bruitées', 
                    color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.5)
    bars5 = ax2.bar(x, mse_unet, width, label='U-Net débruité', 
                    color='#3498db', alpha=0.85, edgecolor='black', linewidth=1.5)
    bars6 = ax2.bar(x + width, mse_gan, width, label='U-Net+GAN débruité', 
                    color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=1.5)

    for bars in [bars4, bars5, bars6]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_xlabel('Type de bruit', fontsize=13, fontweight='bold')
    ax2.set_ylabel('MSE', fontsize=13, fontweight='bold')
    ax2.set_title('MSE moyen : Comparaison des méthodes', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(noise_types, fontsize=11)
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)

    all_mse = mse_noisy + mse_unet + mse_gan
    ax2.set_ylim(bottom=0, top=max(all_mse) * 1.15)

    # ========== GRAPHIQUE 3 : VIF ==========
    # Afficher le graphique VIF seulement si les valeurs existent
    if any(vif_noisy) or any(vif_unet) or any(vif_gan):
        bars7 = ax3.bar(x - width, vif_noisy, width, label='Images bruitées', 
                        color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.5)
        bars8 = ax3.bar(x, vif_unet, width, label='U-Net débruité', 
                        color='#3498db', alpha=0.85, edgecolor='black', linewidth=1.5)
        bars9 = ax3.bar(x + width, vif_gan, width, label='U-Net+GAN débruité', 
                        color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=1.5)

        for bars in [bars7, bars8, bars9]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax3.set_xlabel('Type de bruit', fontsize=13, fontweight='bold')
        ax3.set_ylabel('VIF', fontsize=13, fontweight='bold')
        ax3.set_title('VIF moyen : Comparaison des méthodes', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(noise_types, fontsize=11)
        ax3.legend(fontsize=11, loc='upper right')
        ax3.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
        all_vif = [v for v in vif_noisy + vif_unet + vif_gan if v is not None]
        if all_vif:
            ax3.set_ylim(bottom=0, top=max(all_vif) * 1.15)
    else:
        ax3.axis('off')
        ax3.set_title('VIF non disponible', fontsize=14, fontweight='bold')

    # ========== GRAPHIQUE 4 : DISTS ==========
    # Afficher le graphique DISTS seulement si les valeurs existent
    if any(dists_noisy) or any(dists_unet) or any(dists_gan):
        bars10 = ax4.bar(x - width, dists_noisy, width, label='Images bruitées', 
                        color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.5)
        bars11 = ax4.bar(x, dists_unet, width, label='U-Net débruité', 
                        color='#3498db', alpha=0.85, edgecolor='black', linewidth=1.5)
        bars12 = ax4.bar(x + width, dists_gan, width, label='U-Net+GAN débruité', 
                        color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=1.5)

        for bars in [bars10, bars11, bars12]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax4.set_xlabel('Type de bruit', fontsize=13, fontweight='bold')
        ax4.set_ylabel('DISTS', fontsize=13, fontweight='bold')
        ax4.set_title('DISTS moyen : Comparaison des méthodes', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(noise_types, fontsize=11)
        ax4.legend(fontsize=11, loc='upper right')
        ax4.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
        all_dists = [v for v in dists_noisy + dists_unet + dists_gan if v is not None]
        if all_dists:
            ax4.set_ylim(bottom=0, top=max(all_dists) * 1.15)
    else:
        ax4.axis('off')
        ax4.set_title('DISTS non disponible', fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Sauvegarder
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Graphique sauvegardé: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Comparaison U-Net vs U-Net+GAN')
    parser.add_argument('--dataset', type=str, default='stl10',
                        choices=['cifar10', 'stl10'],
                        help='Dataset à utiliser (défaut: stl10, seul disponible actuellement)')
    parser.add_argument('--n-samples', type=int, default=100,
                        help='Nombre d\'images à évaluer (défaut: 100)')
    parser.add_argument('--unet-model', type=str, default=None,
                        help='Chemin vers le modèle U-Net (auto-détecté si non spécifié)')
    parser.add_argument('--gan-model', type=str, default=None,
                        help='Chemin vers le modèle U-Net+GAN (auto-détecté si non spécifié)')
    parser.add_argument('--compute-vif', action='store_true', help='Calculer VIF (Visual Information Fidelity) entre GT et outputs')
    parser.add_argument('--compute-dists', action='store_true', help='Calculer DISTS (perceptual) entre GT et outputs')
    parser.add_argument('--dists-net', type=str, default='vgg', choices=['vgg'], help='Backbone pour DISTS (défaut: vgg)')

    args = parser.parse_args()

    print("=" * 80)
    print("COMPARAISON U-Net vs U-Net+GAN")
    print("=" * 80)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Device: {device}")

    # Charger DISTS si demandé
    dists_model = None
    if args.compute_dists:
        try:
            import DISTS_pytorch as DISTS
            print(f"\nChargement DISTS (net={args.dists_net})...")
            # Vérifier la présence des fichiers .mat nécessaires
            import os
            alpha_beta_path = './code/alpha_beta.mat'
            net_param_path = './code/net_param.mat'
            # Téléchargement automatique si manquant
            if not os.path.exists(alpha_beta_path):
                print("Téléchargement de alpha_beta.mat...")
                url = 'https://github.com/dingkeyan93/DISTS/raw/master/weights/alpha_beta.mat'
                import urllib.request
                urllib.request.urlretrieve(url, alpha_beta_path)
            if not os.path.exists(net_param_path):
                print("Téléchargement de net_param.mat...")
                url = 'https://github.com/dingkeyan93/DISTS/raw/master/weights/net_param.mat'
                import urllib.request
                urllib.request.urlretrieve(url, net_param_path)
            # Instantiate without automatic torch.load to avoid /usr/weights.pt lookup
            dists_model = DISTS.DISTS(load_weights=False).to(device)
            # Load alpha/beta from .mat if present and set them
            try:
                from scipy.io import loadmat
                mat = loadmat(alpha_beta_path)
                alpha = mat.get('alpha')
                beta = mat.get('beta')
                if alpha is not None and beta is not None:
                    import torch as _torch
                    # alpha and beta are shape (1, N), model expects (1, N, 1, 1)
                    alpha_t = _torch.tensor(alpha.astype('float32'))
                    beta_t = _torch.tensor(beta.astype('float32'))
                    alpha_t = alpha_t.view(1, -1, 1, 1)
                    beta_t = beta_t.view(1, -1, 1, 1)
                    dists_model.alpha.data = alpha_t.to(device)
                    dists_model.beta.data = beta_t.to(device)
            except Exception as e:
                print(f"⚠️  Erreur lors du chargement des paramètres DISTS depuis .mat: {e}")
        except Exception as e:
            print(f"\n⚠️  Erreur lors du chargement de DISTS: {e}")
            dists_model = None

    # Vérifier la disponibilité pour CIFAR-10
    if args.dataset == 'cifar10':
        print("\n⚠️  ATTENTION: Aucun modèle U-Net+GAN disponible pour CIFAR-10")
        print("    Pour entraîner un modèle U-Net+GAN sur CIFAR-10:")
        print("    1. Entraîner d'abord un U-Net classique:")
        print("       python code/unet_train_multi_noise.py --dataset cifar10 --epochs 100")
        print("    2. Puis entraîner le GAN:")
        print("       python code/unet_gan_train.py --dataset cifar10 --pretrained-unet ./code/unet_denoising_cifar10_multinoise.pth --epochs 50")
        print("\n    Actuellement, seul STL-10 dispose d'un modèle U-Net+GAN entraîné.")
        print("    Utilisez --dataset stl10 pour l'évaluation comparative.")
        return
    
    # Trouver les modèles automatiquement
    print(f"\n📂 Recherche des modèles pour {args.dataset.upper()}...")
    model_paths = find_model_paths(args.dataset)
    
    # Utiliser les chemins fournis ou auto-détectés
    unet_path = args.unet_model if args.unet_model else model_paths['unet']
    gan_path = args.gan_model if args.gan_model else model_paths['unet_gan']
    
    print(f"\nModèles sélectionnés:")
    print(f"  - U-Net:     {unet_path}")
    print(f"  - U-Net+GAN: {gan_path}")
    
    # Vérifier l'existence des modèles
    if not unet_path or not Path(unet_path).exists():
        print(f"\n❌ Erreur: Modèle U-Net non trouvé")
        print(f"   Attendu: ./code/unet_denoising_{args.dataset}_multinoise.pth")
        return
    
    if not gan_path or not Path(gan_path).exists():
        print(f"\n❌ Erreur: Modèle U-Net+GAN non trouvé")
        print(f"   Attendu: ./code/gan_unet_denoising_{args.dataset}_multinoise.pth")
        return
    
    # Charger les modèles
    print(f"\n📦 Chargement des modèles...")
    unet_model = load_unet_model(unet_path, device)
    print("  ✓ U-Net chargé")
    
    gan_model = load_unet_model(gan_path, device)
    print("  ✓ U-Net+GAN chargé")
    
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
    
    # Charger VIF si demandé
    vif_func = None
    if args.compute_vif:
        try:
            from sewar.full_ref import vifp
            print(f"\nChargement VIF (sewar.vifp)...")
            vif_func = vifp
        except Exception as e:
            print(f"\n⚠️  Erreur lors du chargement de VIF (sewar): {e}")
            vif_func = None

    # Évaluer pour chaque type de bruit
    results = []

    print(f"\n{'='*80}")
    print("ÉVALUATION DES MODÈLES")
    print(f"{'='*80}")

    for config in noise_configs:
        print(f"\n⏳ {config['name']}... ", end='', flush=True)
        metrics = evaluate_models(
            unet_model, gan_model,
            x_test, config['type'], config['params'],
            device, n_samples=args.n_samples,
            vif_func=vif_func,
            dists_model=dists_model
        )
        results.append({
            'name': config['name'],
            'metrics': metrics
        })
        print("✓")
        print(f"   PSNR: Noisy={metrics['psnr_noisy']:.2f} | "
              f"U-Net={metrics['psnr_unet']:.2f} | U-Net+GAN={metrics['psnr_gan']:.2f}")
        print(f"   MSE:  Noisy={metrics['mse_noisy']:.4f} | "
              f"U-Net={metrics['mse_unet']:.4f} | U-Net+GAN={metrics['mse_gan']:.4f}")
        if vif_func is not None:
            print(f"   VIF: Noisy={metrics['vif_noisy']:.4f} | "
                  f"U-Net={metrics['vif_unet']:.4f} | U-Net+GAN={metrics['vif_gan']:.4f}")
        if dists_model is not None:
            print(f"   DISTS: Noisy={metrics['dists_noisy']:.4f} | "
                  f"U-Net={metrics['dists_unet']:.4f} | U-Net+GAN={metrics['dists_gan']:.4f}")
    
    # Afficher le tableau comparatif
    print(f"\n{'='*80}")
    print("TABLEAU COMPARATIF")
    print(f"{'='*80}")
    
    print(f"\n{'Type de bruit':<20s} {'Méthode':<15s} {'PSNR (dB)':<12s} {'MSE':<12s} {'Gain PSNR':<12s}")
    print("-" * 80)
    
    for result in results:
        m = result['metrics']
        print(f"{result['name']:<20s} {'Bruitée':<15s} {m['psnr_noisy']:>10.2f}   {m['mse_noisy']:>10.4f}   {'-':<12s}")
        
        gain_unet = m['psnr_unet'] - m['psnr_noisy']
        print(f"{'':20s} {'U-Net':<15s} {m['psnr_unet']:>10.2f}   {m['mse_unet']:>10.4f}   {gain_unet:>+10.2f}")
        
        gain_gan = m['psnr_gan'] - m['psnr_noisy']
        winner = '🏆' if m['psnr_gan'] > m['psnr_unet'] else ''
        diff = m['psnr_gan'] - m['psnr_unet']
        print(f"{'':20s} {'U-Net+GAN':<15s} {m['psnr_gan']:>10.2f}   {m['mse_gan']:>10.4f}   {gain_gan:>+10.2f} {winner}")
        if abs(diff) > 0.01:
            symbol = '+' if diff > 0 else ''
            print(f"{'':20s} {'  (vs U-Net)':<15s} {'':>10s}   {'':>10s}   {symbol}{diff:>9.2f} dB")
        print()
    
    # Calculer les moyennes globales
    avg_psnr_unet = np.mean([r['metrics']['psnr_unet'] for r in results])
    avg_psnr_gan = np.mean([r['metrics']['psnr_gan'] for r in results])
    avg_mse_unet = np.mean([r['metrics']['mse_unet'] for r in results])
    avg_mse_gan = np.mean([r['metrics']['mse_gan'] for r in results])
    
    print("-" * 80)
    print(f"\n🏆 Performances moyennes sur tous les types de bruit:")
    print(f"   U-Net:     PSNR = {avg_psnr_unet:.2f} dB | MSE = {avg_mse_unet:.4f}")
    print(f"   U-Net+GAN: PSNR = {avg_psnr_gan:.2f} dB | MSE = {avg_mse_gan:.4f}")
    
    diff = avg_psnr_gan - avg_psnr_unet
    if diff > 0:
        print(f"\n   ✅ U-Net+GAN est meilleur de {diff:.2f} dB en moyenne")
    elif diff < 0:
        print(f"\n   ⚠️  U-Net classique est meilleur de {-diff:.2f} dB en moyenne")
    else:
        print(f"\n   ⚖️  Performances équivalentes")
    
    # Créer les graphiques
    print(f"\n{'='*80}")
    print("GÉNÉRATION DES GRAPHIQUES")
    print(f"{'='*80}")
    
    save_path = f'./code/comparison_unet_gan_{args.dataset}.png'
    plot_comparison(results, args.dataset, save_path=save_path)
    
    print(f"\n{'='*80}")
    print("COMPARAISON TERMINÉE ✓")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
