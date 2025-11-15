#!/usr/bin/env python3
"""Utilitaires pour charger et explorer le dataset STL-10 (format torchvision)."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import STL10


class STL10Loader:
    """
    Chargeur convivial pour STL-10, inspiré de `CIFAR10Loader`.

    Attributs principaux:
        data_dir (str): dossier racine où stocker les fichiers téléchargés
        class_names (list[str]): noms des classes (10 catégories)
        x_train/y_train, x_test/y_test, x_unlabeled (np.ndarray): caches des données
    """

    def __init__(self, data_dir: str = 'code/stl10', download: bool = True) -> None:
        self.data_dir = data_dir
        self.download = download

        self.class_names: Optional[list[str]] = None
        self.x_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.x_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.x_unlabeled: Optional[np.ndarray] = None

        os.makedirs(self.data_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Chargements bas niveau
    # ------------------------------------------------------------------
    def _load_split(self, split: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        dataset = STL10(root=self.data_dir, split=split, download=self.download)

        # La première fois, mémoriser les noms de classes fournis par torchvision
        if self.class_names is None:
            self.class_names = list(dataset.classes)

        images = dataset.data  # shape (N, 3, 96, 96) en uint8
        images = np.transpose(images, (0, 2, 3, 1))  # -> (N, 96, 96, 3)

        if split == 'unlabeled':
            labels = None
        else:
            labels = np.array(dataset.labels)

        return images, labels

    # ------------------------------------------------------------------
    # API publique similaire à CIFAR10Loader
    # ------------------------------------------------------------------
    def load_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.x_train is None or self.y_train is None:
            self.x_train, self.y_train = self._load_split('train')
        return self.x_train, self.y_train

    def load_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.x_test is None or self.y_test is None:
            self.x_test, self.y_test = self._load_split('test')
        return self.x_test, self.y_test

    def load_unlabeled_data(self) -> np.ndarray:
        if self.x_unlabeled is None:
            self.x_unlabeled, _ = self._load_split('unlabeled')
        return self.x_unlabeled

    def load_all_data(self):
        x_train, y_train = self.load_train_data()
        x_test, y_test = self.load_test_data()
        x_unlabeled = self.load_unlabeled_data()
        return x_train, y_train, x_test, y_test, x_unlabeled

    # ------------------------------------------------------------------
    # Utilitaires d'analyse similaires à CIFAR10Loader
    # ------------------------------------------------------------------
    def normalize(self, data: Optional[np.ndarray] = None, method: str = '0-1'):
        if data is None:
            if self.x_train is not None:
                self.x_train = self.normalize(self.x_train, method)
            if self.x_test is not None:
                self.x_test = self.normalize(self.x_test, method)
            if self.x_unlabeled is not None:
                self.x_unlabeled = self.normalize(self.x_unlabeled, method)
            return self.x_train, self.x_test, self.x_unlabeled

        if method == '0-1':
            return data.astype('float32') / 255.0
        if method == 'standard':
            mean = np.mean(data, axis=(0, 1, 2))
            std = np.std(data, axis=(0, 1, 2))
            return (data.astype('float32') - mean) / std
        raise ValueError(f"Méthode de normalisation inconnue: {method}")

    def get_class_distribution(self, labels: Optional[np.ndarray] = None):
        if labels is None:
            if self.y_train is None:
                raise ValueError("Les labels train ne sont pas chargés")
            labels = self.y_train

        unique, counts = np.unique(labels, return_counts=True)
        return {
            self.class_names[label] if self.class_names else str(label): count
            for label, count in zip(unique, counts)
        }

    def visualize_samples(self, split: str = 'train', n_samples: int = 10,
                           save_path: Optional[str] = None) -> None:
        if split not in {'train', 'test', 'unlabeled'}:
            raise ValueError("split doit être 'train', 'test' ou 'unlabeled'")

        if split == 'train':
            images, labels = self.load_train_data()
        elif split == 'test':
            images, labels = self.load_test_data()
        else:
            images = self.load_unlabeled_data()
            labels = None

        fig, axes = plt.subplots(2, max(1, n_samples // 2), figsize=(12, 5))
        axes = axes.ravel()

        idxs = np.random.choice(len(images), size=min(n_samples, len(images)), replace=False)
        for ax, idx in zip(axes, idxs):
            ax.imshow(images[idx])
            if labels is not None and self.class_names:
                ax.set_title(self.class_names[int(labels[idx])])
            ax.axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Échantillons STL-10 sauvegardés dans '{save_path}'")
        plt.show()

    def print_info(self) -> None:
        print("=" * 60)
        print("Informations sur le dataset STL-10")
        print("=" * 60)
        if self.class_names:
            print(f"\nClasses ({len(self.class_names)}):")
            for idx, name in enumerate(self.class_names):
                print(f"  {idx}: {name}")
        else:
            print("\nClasses disponibles après premier chargement.")

        if self.x_train is not None:
            print(f"\nTrain: {self.x_train.shape}, dtype={self.x_train.dtype}, min/max={self.x_train.min()} / {self.x_train.max()}")
        if self.x_test is not None:
            print(f"Test: {self.x_test.shape}, dtype={self.x_test.dtype}, min/max={self.x_test.min()} / {self.x_test.max()}")
        if self.x_unlabeled is not None:
            print(f"Unlabeled: {self.x_unlabeled.shape}")
        print("=" * 60)


if __name__ == '__main__':
    loader = STL10Loader(download=True)
    x_train, y_train = loader.load_train_data()
    x_test, y_test = loader.load_test_data()
    print("Train:", x_train.shape, y_train.shape)
    print("Test:", x_test.shape, y_test.shape)
    loader.print_info()
