import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import cv2
from UNET.unet_model import UNet
import torch



from tqdm import tqdm
from noise_generator import NoiseGenerator
from utils import add_noise_to_images
from patch_utils import denoise_with_patches
# VIF
try:
    from sewar.full_ref import vifp
except ImportError:
    vifp = None
# DISTS
try:
    import DISTS_pytorch as DISTS
except ImportError:
    DISTS = None


# --- Paramètres ---
#model_path = 'brouillon/best_epoch00000491_psnr33.78.pth'
model_path = 'brouillon/morenoise_best_epoch00000379_psnr33.81.pth'

metrics = ['psnr', 'mse', 'vif', 'dists']  # Ajoute 'vif', 'dists' si dispo

def calculate_psnr(img1, img2, max_pixel_value=255.0):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_pixel_value / np.sqrt(mse))

def calculate_mse(img1, img2):
    return np.mean((img1.astype(float) - img2.astype(float)) ** 2)

def to_gray_uint8(img_arr):
    # img_arr: HxWx3 uint8
    if img_arr.ndim == 3 and img_arr.shape[2] == 3:
        gray = (0.299 * img_arr[:,:,0] + 0.587 * img_arr[:,:,1] + 0.114 * img_arr[:,:,2])
    else:
        gray = img_arr[:,:,0]
    return gray.astype(np.uint8)

def prep_for_dists(img_arr, device):
    import torchvision.transforms as T
    pil = T.ToPILImage()(img_arr.astype(np.uint8))
    proc = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    return proc(pil).unsqueeze(0).to(device)


# --- Chargement du dataset STL-10 ---
import torchvision
from torchvision import transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = './code/code/stl10'  # À adapter si besoin
test_dataset = torchvision.datasets.STL10(
    data_dir, split='test', download=False, transform=transforms.ToTensor()
)

# Charger les images en numpy
x_test = []
for i in range(len(test_dataset)):
    img, _ = test_dataset[i]
    x_test.append(img.permute(1, 2, 0).numpy())
x_test = np.array(x_test)

# Pour le modèle
model = UNet(n_channels=3, n_classes=3, base_features=64)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Charger DISTS si dispo
dists_model = None
if DISTS is not None:
    import os
    alpha_beta_path = './code/alpha_beta.mat'
    net_param_path = './code/net_param.mat'
    # Téléchargement automatique si manquant
    if not os.path.exists(alpha_beta_path):
        url = 'https://github.com/dingkeyan93/DISTS/raw/master/weights/alpha_beta.mat'
        import urllib.request
        urllib.request.urlretrieve(url, alpha_beta_path)
    if not os.path.exists(net_param_path):
        url = 'https://github.com/dingkeyan93/DISTS/raw/master/weights/net_param.mat'
        import urllib.request
        urllib.request.urlretrieve(url, net_param_path)
    dists_model = DISTS.DISTS(load_weights=False).to(device)
    # Charger alpha/beta si possible
    try:
        from scipy.io import loadmat
        mat = loadmat(alpha_beta_path)
        alpha = mat.get('alpha')
        beta = mat.get('beta')
        if alpha is not None and beta is not None:
            import torch as _torch
            alpha_t = _torch.tensor(alpha.astype('float32'))
            beta_t = _torch.tensor(beta.astype('float32'))
            alpha_t = alpha_t.view(1, -1, 1, 1)
            beta_t = beta_t.view(1, -1, 1, 1)
            dists_model.alpha.data = alpha_t.to(device)
            dists_model.beta.data = beta_t.to(device)
    except Exception:
        pass


# --- Types de bruit à traiter ---
# Utiliser les mêmes paramètres que evaluate_unet_gan.py pour cohérence
noise_configs = [
    {
        'name': 'Gaussien',
        'type': 'gaussian',
        'levels': [0, 10, 15, 20, 25, 30, 40, 50, 60],
        'params': lambda lvl: {'std': lvl},
        'xlabel': 'Niveau de bruit (sigma)',
    },
    {
        'name': 'Salt & Pepper',
        'type': 'salt_pepper',
        'levels': [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1],
        'params': lambda lvl: {'salt_prob': lvl, 'pepper_prob': lvl},
        'xlabel': 'Niveau de bruit (proba)',
    },
    {
        'name': 'Mixte (Gaussien + Salt & Pepper)',
        'type': 'mixed',
        'levels': [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
        'params': lambda lvl: {
            'gaussian_std': 20 * lvl,  # 0 à 30 progressivement
            'salt_prob': 0.02 * lvl,   # 0 à 0.03 progressivement
            'pepper_prob': 0.02 * lvl  # 0 à 0.03 progressivement
        },
        'xlabel': 'Intensité du bruit mixte',
        'custom_xticks': True,  # Activer les labels personnalisés
        'xtick_labels': [
            'Aucun\n(σ=0, p=0)',
            'Faible\n(σ=5, p=0.005)',
            'Modéré\n(σ=10, p=0.01)',
            'Moyen\n(σ=15, p=0.015)',
            'Élevé\n(σ=20, p=0.02)',
            'Fort\n(σ=25, p=0.025)',
            'Très fort\n(σ=30, p=0.03)'
        ]
    },
]

for config in noise_configs:
    noise_levels = config['levels']
    psnr_list = []
    mse_list = []
    vif_list = []
    dists_list = []
    n_samples = 1000  # nombre d'images à utiliser (adapter si besoin)
    indices = np.random.choice(len(x_test), n_samples, replace=False)
    x_samples = (x_test[indices] * 255).astype(np.uint8)
    print(f"\nCalcul des métriques pour le bruit : {config['name']}")
    for level in tqdm(noise_levels, desc=f"{config['name']} (niveau de bruit)"):
        # Générer les images bruitées avec add_noise_to_images (comme evaluate_unet_gan)
        noise_params = config['params'](level)
        noisy_imgs = add_noise_to_images(x_samples, noise_type=config['type'], **noise_params)
        
        # Convertir en tensor [0, 1]
        noisy_tensor = torch.FloatTensor(noisy_imgs).permute(0, 3, 1, 2) / 255.0
        
        # Débruiter avec denoise_with_patches (comme evaluate_unet_gan)
        denoised_imgs_list = []
        for i in tqdm(range(n_samples), desc="Débruitage", leave=False):
            img_denoised = denoise_with_patches(
                model, noisy_tensor[i], device,
                patch_size=32, stride=16
            )
            denoised_imgs_list.append(img_denoised)
        
        denoised_tensor = torch.stack(denoised_imgs_list)
        denoised_imgs = (denoised_tensor.permute(0, 2, 3, 1).numpy() * 255.0)
        denoised_imgs = np.clip(denoised_imgs, 0, 255).astype(np.uint8)

        # Calculer les métriques batch
        psnr_vals = [calculate_psnr(denoised_imgs[i], x_samples[i], max_pixel_value=255.0) for i in range(n_samples)]
        mse_vals = [calculate_mse(denoised_imgs[i], x_samples[i]) for i in range(n_samples)]
        psnr_list.append(np.mean(psnr_vals))
        mse_list.append(np.mean(mse_vals))
        # VIF
        if vifp is not None:
            def to_gray_uint8(img_arr):
                if img_arr.ndim == 3 and img_arr.shape[2] == 3:
                    gray = (0.299 * img_arr[:,:,0] + 0.587 * img_arr[:,:,1] + 0.114 * img_arr[:,:,2])
                else:
                    gray = img_arr[:,:,0]
                return gray.astype(np.uint8)
            vif_vals = []
            for i in range(n_samples):
                try:
                    gt_gray = to_gray_uint8(x_samples[i])
                    denoised_gray = to_gray_uint8(denoised_imgs[i])
                    vif_val = float(vifp(gt_gray, denoised_gray))
                except Exception:
                    vif_val = None
                vif_vals.append(vif_val)
            vif_list.append(np.mean([v for v in vif_vals if v is not None]))
        # DISTS
        if dists_model is not None:
            import torchvision.transforms as T
            def prep_for_dists(img_arr, device):
                pil = T.ToPILImage()(img_arr.astype(np.uint8))
                proc = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor()
                ])
                return proc(pil).unsqueeze(0).to(device)
            dists_vals = []
            for i in range(n_samples):
                try:
                    gt_dists = prep_for_dists(x_samples[i], device)
                    denoised_dists = prep_for_dists(denoised_imgs[i], device)
                    with torch.no_grad():
                        dists_val = float(dists_model(gt_dists, denoised_dists).cpu().item())
                except Exception:
                    dists_val = None
                dists_vals.append(dists_val)
            dists_list.append(np.mean([d for d in dists_vals if d is not None]))
    # --- Affichage : 1 figure, 4 subplots (PSNR, MSE, VIF, DISTS) ---
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Évolution des métriques - {config['name']} (STL-10)", fontsize=16, fontweight='bold')

    # Courbe PSNR
    ax1 = axs[0, 0]
    ax1.plot(noise_levels, psnr_list, label='PSNR', marker='o', color='#3498db')
    ax1.set_xlabel(config['xlabel'], fontsize=12)
    ax1.set_ylabel('PSNR (dB)', fontsize=12)
    ax1.set_title('PSNR', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    # Labels personnalisés si configurés
    if config.get('custom_xticks', False):
        ax1.set_xticks(noise_levels)
        ax1.set_xticklabels(config['xtick_labels'], fontsize=9)

    # Courbe MSE
    ax2 = axs[0, 1]
    ax2.plot(noise_levels, mse_list, label='MSE', marker='s', color='#e67e22')
    ax2.set_xlabel(config['xlabel'], fontsize=12)
    ax2.set_ylabel('MSE', fontsize=12)
    ax2.set_title('MSE', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    # Labels personnalisés si configurés
    if config.get('custom_xticks', False):
        ax2.set_xticks(noise_levels)
        ax2.set_xticklabels(config['xtick_labels'], fontsize=9)

    # Courbe VIF
    ax3 = axs[1, 0]
    if vif_list and any(v is not None for v in vif_list):
        ax3.plot(noise_levels, vif_list, label='VIF', marker='^', color='#2ecc71')
        ax3.set_ylim(bottom=0)
    else:
        ax3.text(0.5, 0.5, 'VIF non disponible', ha='center', va='center', fontsize=12)
    ax3.set_xlabel(config['xlabel'], fontsize=12)
    ax3.set_ylabel('VIF', fontsize=12)
    ax3.set_title('VIF', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    # Labels personnalisés si configurés
    if config.get('custom_xticks', False):
        ax3.set_xticks(noise_levels)
        ax3.set_xticklabels(config['xtick_labels'], fontsize=9)

    # Courbe DISTS
    ax4 = axs[1, 1]
    if dists_list and any(d is not None for d in dists_list):
        ax4.plot(noise_levels, dists_list, label='DISTS', marker='x', color='#9b59b6')
        ax4.set_ylim(bottom=0)
    else:
        ax4.text(0.5, 0.5, 'DISTS non disponible', ha='center', va='center', fontsize=12)
    ax4.set_xlabel(config['xlabel'], fontsize=12)
    ax4.set_ylabel('DISTS', fontsize=12)
    ax4.set_title('DISTS', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    # Labels personnalisés si configurés
    if config.get('custom_xticks', False):
        ax4.set_xticks(noise_levels)
        ax4.set_xticklabels(config['xtick_labels'], fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"plot_metrics_{config['type']}_stl10.png", dpi=200)
    plt.show()
