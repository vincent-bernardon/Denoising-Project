import os
import argparse
import torch
import cv2
import numpy as np

from UNET.unet_model import UNet
from patch_utils import denoise_with_patches

def read_model_path(path_file):
    if not os.path.isfile(path_file):
        raise FileNotFoundError(f"Fichier de chemin modèle non trouvé: {path_file}")
    with open(path_file, 'r') as f:
        return f.read().strip()

def write_model_path(path_file, new_path):
    with open(path_file, 'w') as f:
        f.write(new_path.strip())

def denoise_image(image_path, model_path, method='global'):
    # Charger l'image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Impossible de lire l'image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_np = img_rgb.astype(np.float32) / 255.0
    img_tensor = torch.FloatTensor(img_np).permute(2, 0, 1)

    # Charger le modèle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=3, n_classes=3, base_features=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    from patch_utils import denoise_with_patches_edge
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        if method == 'patch':
            # Débruitage par patchs classique
            denoised = denoise_with_patches(model, img_tensor, device, patch_size=32, stride=16)
        elif method == 'patch_edge':
            # Débruitage par patchs avec gestion des bords
            denoised = denoise_with_patches_edge(model, img_tensor, device, patch_size=32, stride=16)
        else:
            # Débruitage global
            denoised = model(img_tensor.unsqueeze(0))[0]
        denoised_np = denoised.cpu().permute(1, 2, 0).numpy()
        denoised_np = np.clip(denoised_np * 255, 0, 255).astype(np.uint8)
        denoised_bgr = cv2.cvtColor(denoised_np, cv2.COLOR_RGB2BGR)
    return denoised_bgr

def main():
    print("Débruitage d'image avec U-Net\n")
    model_path_file = 'model_path.txt'
    # Lire le chemin du modèle
    try:
        model_path = read_model_path(model_path_file)
        print(f"Modèle utilisé: {model_path}")
    except Exception as e:
        print(e)
        model_path = input("Chemin du modèle .pth à utiliser : ").strip()
        write_model_path(model_path_file, model_path)
        print(f"Chemin modèle enregistré dans {model_path_file}")

    change_model = input("Voulez-vous changer le modèle ? (o/n) : ").strip().lower()
    if change_model == 'o':
        model_path = input("Nouveau chemin du modèle .pth : ").strip()
        write_model_path(model_path_file, model_path)
        print(f"Nouveau chemin modèle enregistré dans {model_path_file}")

    image_path = input("Chemin de l'image à débruiter : ").strip()
    if not os.path.isfile(image_path):
        print("Fichier image non trouvé.")
        return

    # Choix de la méthode
    print("Méthode de débruitage :")
    print("1. Global (image entière)")
    print("2. Par patchs (classique)")
    print("3. Par patchs (bord ajusté, recommandé si artefacts)")
    method_choice = input("Choisir la méthode (1/2/3) : ").strip()
    if method_choice == '2':
        method = 'patch'
    elif method_choice == '3':
        method = 'patch_edge'
    else:
        method = 'global'

    # Débruiter l'image
    denoised_img = denoise_image(image_path, model_path, method=method)

    # Sauvegarder l'image débruitée
    dir_name, file_name = os.path.split(image_path)
    name, ext = os.path.splitext(file_name)
    out_name = f"denoise_{method}_{name}{ext}"
    out_path = os.path.join(dir_name, out_name)
    cv2.imwrite(out_path, denoised_img)
    print(f"Image débruitée enregistrée: {out_path}")

if __name__ == "__main__":
    main()
