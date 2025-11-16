"""
Script pour évaluer les performances moyennes du modèle vae_denoiser_beta0.pth
Calcule le PSNR moyen et MSE moyen pour chaque type de bruit
AVEC GRAPHIQUE SIMPLE (3 barres)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from load_cifar10 import CIFAR10Loader
from vae_model import Encoder, Decoder
from vae_train import load_model, calculate_psnr, calculate_mse
from utils import add_noise_to_images


def calculate_metrics_for_noise_type(encoder, decoder, x_data, noise_type, noise_params, device, n_samples=1000):
    """
    Calcule les métriques moyennes pour un type de bruit donné
    
    Args:
        encoder: Modèle encodeur
        decoder: Modèle décodeur
        x_data: Images propres en [0,1]
        noise_type: Type de bruit ('gaussian', 'salt_pepper', 'mixed')
        noise_params: Paramètres du bruit
        device: Device PyTorch
        n_samples: Nombre d'images à évaluer
    
    Returns:
        dict: Métriques calculées
    """
    # Limiter le nombre d'échantillons
    if len(x_data) > n_samples:
        indices = np.random.choice(len(x_data), n_samples, replace=False)
        x_data = x_data[indices]
    
    # Convertir en uint8 [0, 255] comme dans test_visualization.py
    x_samples = (x_data * 255).astype(np.uint8)
    
    # Générer les images bruitées (add_noise_to_images attend du uint8)
    x_noisy = add_noise_to_images(x_samples, noise_type=noise_type, **noise_params)
    
    # Normaliser pour le VAE [0, 1]
    x_noisy_tensor = torch.FloatTensor(x_noisy).permute(0, 3, 1, 2) / 255.0
    x_noisy_tensor = x_noisy_tensor.to(device)
    
    # Débruiter avec le VAE
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        # Encoder
        mu, logvar = encoder(x_noisy_tensor)
        # Utiliser mu directement (pas de sampling pour l'évaluation)
        z = mu
        # Décoder
        x_recon_tensor = decoder(z)
    
    # Reconvertir en numpy uint8 [0, 255]
    x_denoised = (x_recon_tensor.permute(0, 2, 3, 1).cpu().numpy() * 255.0)
    x_denoised = np.clip(x_denoised, 0, 255).astype(np.uint8)
    
    # Calculer les métriques (comme dans test_visualization.py)
    psnr_noisy_list = []
    psnr_denoised_list = []
    mse_noisy_list = []
    mse_denoised_list = []
    
    for i in range(len(x_samples)):
        # PSNR et MSE entre image bruitée et image propre
        psnr_noisy = calculate_psnr(x_noisy[i], x_samples[i], max_pixel_value=255.0)
        mse_noisy = calculate_mse(x_noisy[i], x_samples[i])
        
        # PSNR et MSE entre image débruitée et image propre
        psnr_denoised = calculate_psnr(x_denoised[i], x_samples[i], max_pixel_value=255.0)
        mse_denoised = calculate_mse(x_denoised[i], x_samples[i])
        
        psnr_noisy_list.append(psnr_noisy)
        psnr_denoised_list.append(psnr_denoised)
        mse_noisy_list.append(mse_noisy)
        mse_denoised_list.append(mse_denoised)
    
    # Calculer les moyennes
    avg_psnr_noisy = np.mean(psnr_noisy_list)
    avg_psnr_denoised = np.mean(psnr_denoised_list)
    avg_mse_noisy = np.mean(mse_noisy_list)
    avg_mse_denoised = np.mean(mse_denoised_list)
    
    # Calculer les gains
    psnr_gain = avg_psnr_denoised - avg_psnr_noisy
    mse_reduction = avg_mse_noisy - avg_mse_denoised
    mse_reduction_percent = (mse_reduction / avg_mse_noisy) * 100
    
    return {
        'avg_psnr_noisy': avg_psnr_noisy,
        'avg_psnr_denoised': avg_psnr_denoised,
        'psnr_gain': psnr_gain,
        'avg_mse_noisy': avg_mse_noisy,
        'avg_mse_denoised': avg_mse_denoised,
        'mse_reduction': mse_reduction,
        'mse_reduction_percent': mse_reduction_percent,
        'n_samples': len(x_samples)
    }


def plot_simple_results(results):
    """
    Crée 2 graphiques : PSNR et MSE avec 6 barres chacun (2 par type de bruit)
    
    Args:
        results: Liste de dictionnaires contenant les résultats pour chaque type de bruit
    """
    noise_types = [r['name'] for r in results]
    
    # Extraire les PSNR moyens pour images bruitées et débruitées
    psnr_noisy = [r['metrics']['avg_psnr_noisy'] for r in results]
    psnr_denoised = [r['metrics']['avg_psnr_denoised'] for r in results]
    
    # Extraire les MSE moyens pour images bruitées et débruitées
    mse_noisy = [r['metrics']['avg_mse_noisy'] for r in results]
    mse_denoised = [r['metrics']['avg_mse_denoised'] for r in results]
    
    # Créer la figure avec 2 sous-graphiques côte à côte
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Position des barres
    x = np.arange(len(noise_types))
    width = 0.35
    
    # ========== GRAPHIQUE 1 : PSNR ==========
    bars1 = ax1.bar(x - width/2, psnr_noisy, width, label='Images bruitées', 
                     color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, psnr_denoised, width, label='Images débruitées', 
                     color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Ajouter les valeurs sur les barres
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Configuration du graphique PSNR
    ax1.set_xlabel('Type de bruit', fontsize=13, fontweight='bold')
    ax1.set_ylabel('PSNR (dB)', fontsize=13, fontweight='bold')
    ax1.set_title('PSNR moyen : Images bruitées vs Images débruitées', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(noise_types, fontsize=11)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)  # Ligne à y=0
    
    # Ajuster les marges
    all_psnr = psnr_noisy + psnr_denoised
    ax1.set_ylim(bottom=min(all_psnr) * 0.95, top=max(all_psnr) * 1.08)
    
    # ========== GRAPHIQUE 2 : MSE ==========
    bars3 = ax2.bar(x - width/2, mse_noisy, width, label='Images bruitées', 
                     color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars4 = ax2.bar(x + width/2, mse_denoised, width, label='Images débruitées', 
                     color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Ajouter les valeurs sur les barres
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    for bar in bars4:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Configuration du graphique MSE
    ax2.set_xlabel('Type de bruit', fontsize=13, fontweight='bold')
    ax2.set_ylabel('MSE', fontsize=13, fontweight='bold')
    ax2.set_title('MSE moyen : Images bruitées vs Images débruitées', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(noise_types, fontsize=11)
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
    
    # Ajuster les marges
    all_mse = mse_noisy + mse_denoised
    ax2.set_ylim(bottom=0, top=max(all_mse) * 1.12)
    
    plt.tight_layout()
    
    # Sauvegarder la figure
    output_path = './code/evaluation_beta0_simple.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Graphique sauvegardé dans: {output_path}")
    
    plt.show()


def main():
    """
    Fonction principale - évalue le modèle beta0 sur les 3 types de bruit
    """
    print("=" * 80)
    print("ÉVALUATION DU MODÈLE vae_denoiser_beta0.pth")
    print("=" * 80)
    
    # Charger les données CIFAR-10
    print("\nChargement du dataset CIFAR-10...")
    loader = CIFAR10Loader()
    x_train, y_train, x_test, y_test = loader.load_all_data()
    
    # Utiliser le test set pour l'évaluation
    x_eval = x_test
    
    print(f"Nombre d'images d'évaluation disponibles: {len(x_eval)}")
    
    # Configuration du modèle
    latent_dim = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device utilisé: {device}")
    
    # Initialiser le modèle
    encoder = Encoder(latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim)
    
    # Charger les poids du modèle beta0
    model_path = './code/vae_denoiser_beta0.pth'
    print(f"\nChargement du modèle depuis: {model_path}")
    
    encoder, decoder, history = load_model(
        encoder,
        decoder,
        filepath=model_path,
        device=device
    )
    
    print("✓ Modèle chargé avec succès!")
    
    # Définir les configurations de bruit à tester
    noise_configs = [
        {
            'name': 'Gaussien',
            'type': 'gaussian',
            'params': {'std': 25}
        },
        {
            'name': 'Salt & Pepper',
            'type': 'salt_pepper',
            'params': {'salt_prob': 0.02, 'pepper_prob': 0.02}
        },
        {
            'name': 'Mixte',
            'type': 'mixed',
            'params': {'gaussian_std': 20, 'salt_prob': 0.01, 'pepper_prob': 0.01}
        }
    ]
    
    # Évaluer pour chaque type de bruit
    results = []
    
    for config in noise_configs:
        print("\n" + "=" * 80)
        print(f"ÉVALUATION: {config['name']}")
        print("=" * 80)
        print(f"Paramètres: {config['params']}")
        
        metrics = calculate_metrics_for_noise_type(
            encoder=encoder,
            decoder=decoder,
            x_data=x_eval,
            noise_type=config['type'],
            noise_params=config['params'],
            device=device,
            n_samples=1000  # Évaluer sur 1000 images pour avoir une bonne moyenne
        )
        
        results.append({
            'name': config['name'],
            'metrics': metrics
        })
        
        # Afficher les résultats
        print(f"\nNombre d'images évaluées: {metrics['n_samples']}")
        print("\n--- PSNR ---")
        print(f"PSNR moyen (images bruitées):   {metrics['avg_psnr_noisy']:.2f} dB")
        print(f"PSNR moyen (images débruitées): {metrics['avg_psnr_denoised']:.2f} dB")
        print(f"→ GAIN PSNR:                    +{metrics['psnr_gain']:.2f} dB")
        
        print("\n--- MSE ---")
        print(f"MSE moyen (images bruitées):    {metrics['avg_mse_noisy']:.6f}")
        print(f"MSE moyen (images débruitées):  {metrics['avg_mse_denoised']:.6f}")
        print(f"→ RÉDUCTION MSE:                -{metrics['mse_reduction']:.6f} ({metrics['mse_reduction_percent']:.1f}%)")
    
    # Résumé final
    print("\n" + "=" * 80)
    print("RÉSUMÉ COMPARATIF")
    print("=" * 80)
    
    print("\n{:<20s} {:>15s} {:>15s} {:>15s}".format(
        "Type de bruit", "PSNR gain (dB)", "MSE réduction", "MSE réd. (%)"
    ))
    print("-" * 80)
    
    for result in results:
        m = result['metrics']
        print("{:<20s} {:>15.2f} {:>15.6f} {:>15.1f}".format(
            result['name'],
            m['psnr_gain'],
            m['mse_reduction'],
            m['mse_reduction_percent']
        ))
    
    print("-" * 80)
    
    # Meilleur type de bruit
    best_psnr = max(results, key=lambda x: x['metrics']['psnr_gain'])
    best_mse = max(results, key=lambda x: x['metrics']['mse_reduction_percent'])
    
    print(f"\n✓ Meilleur gain PSNR: {best_psnr['name']} (+{best_psnr['metrics']['psnr_gain']:.2f} dB)")
    print(f"✓ Meilleure réduction MSE: {best_mse['name']} ({best_mse['metrics']['mse_reduction_percent']:.1f}%)")
    
    print("\n" + "=" * 80)
    print("GÉNÉRATION DU GRAPHIQUE")
    print("=" * 80)
    
    # Créer le graphique simple
    plot_simple_results(results)
    
    print("\n" + "=" * 80)
    print("ÉVALUATION TERMINÉE ✓")
    print("=" * 80)


if __name__ == "__main__":
    main()
