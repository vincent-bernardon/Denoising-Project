import os
import cv2
import numpy as np
from utils import add_noise_to_images

def main():
    path = input("Chemin de l'image à bruiter : ").strip()
    if not os.path.isfile(path):
        print("Fichier non trouvé.")
        return

    img = cv2.imread(path)
    if img is None:
        print("Impossible de lire l'image.")
        return

    print("Type de bruit :\n1. Gaussien\n2. Salt&Pepper\n3. Mixte")
    noise_type = input("Choix (1/2/3) : ").strip()

    params = ""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_np = img_rgb.astype(np.uint8)
    if noise_type == "1":
        std = float(input("std (sigma) du bruit gaussien (ex: 25) : "))
        noisy_img = add_noise_to_images(
            img_np[np.newaxis, ...],
            noise_type="gaussian",
            std=std
        )[0]
        params = f"{std}"
        noise_name = "gaussian"
    elif noise_type == "2":
        salt_prob = float(input("salt_prob (ex: 0.02) : "))
        pepper_prob = float(input("pepper_prob (ex: 0.02) : "))
        noisy_img = add_noise_to_images(
            img_np[np.newaxis, ...],
            noise_type="salt_pepper",
            salt_prob=salt_prob,
            pepper_prob=pepper_prob
        )[0]
        params = f"{salt_prob}_{pepper_prob}"
        noise_name = "saltpepper"
    elif noise_type == "3":
        gaussian_std = float(input("gaussian_std (ex: 20) : "))
        salt_prob = float(input("salt_prob (ex: 0.01) : "))
        pepper_prob = float(input("pepper_prob (ex: 0.01) : "))
        noisy_img = add_noise_to_images(
            img_np[np.newaxis, ...],
            noise_type="mixed",
            gaussian_std=gaussian_std,
            salt_prob=salt_prob,
            pepper_prob=pepper_prob
        )[0]
        params = f"{gaussian_std}_{salt_prob}_{pepper_prob}"
        noise_name = "mixed"
    else:
        print("Choix invalide.")
        return
    # Convertir en uint8 si nécessaire, puis repasser en BGR pour sauvegarde
    if not noisy_img.dtype == np.uint8:
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    noisy_img_bgr = cv2.cvtColor(noisy_img, cv2.COLOR_RGB2BGR)

    dir_name, file_name = os.path.split(path)
    name, ext = os.path.splitext(file_name)
    out_name = f"{name}_noise_{noise_name}_{params}{ext}"
    out_path = os.path.join(dir_name, out_name)
    cv2.imwrite(out_path, noisy_img_bgr)
    print(f"Image bruitée enregistrée : {out_path}")

if __name__ == "__main__":
    main()