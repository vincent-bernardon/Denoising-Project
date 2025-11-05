#!/usr/bin/env python3
"""
Classe pour charger et manipuler le dataset CIFAR-10
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os


class CIFAR10Loader:
    """
    Classe pour charger et manipuler le dataset CIFAR-10
    
    Attributes:
        data_dir (str): Chemin vers le répertoire contenant les données CIFAR-10
        x_train (np.ndarray): Images d'entraînement (50000, 32, 32, 3)
        y_train (np.ndarray): Labels d'entraînement (50000,)
        x_test (np.ndarray): Images de test (10000, 32, 32, 3)
        y_test (np.ndarray): Labels de test (10000,)
        class_names (list): Noms des 10 classes
    """
    
    def __init__(self, data_dir='code/cifar-10-python/cifar-10-batches-py'):
        """
        Initialise le loader CIFAR-10
        
        Args:
            data_dir (str): Chemin vers le répertoire contenant les données
        """
        self.data_dir = data_dir
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.class_names = None
        
        self._load_class_names()
    
    @staticmethod
    def unpickle(file):
        """Charge un fichier pickle de CIFAR-10"""
        with open(file, 'rb') as fo:
            dict_data = pickle.load(fo, encoding='bytes')
        return dict_data
    
    def _load_class_names(self):
        """Récupère les noms des classes"""
        meta = self.unpickle(os.path.join(self.data_dir, 'batches.meta'))
        self.class_names = [name.decode('utf-8') for name in meta[b'label_names']]
    
    def _load_batch(self, file_path):
        """
        Charge un batch de CIFAR-10
        
        Args:
            file_path (str): Chemin vers le fichier batch
            
        Returns:
            tuple: (data, labels, filenames)
        """
        batch = self.unpickle(file_path)
        data = batch[b'data']
        labels = batch[b'labels']
        filenames = batch[b'filenames']
        return data, labels, filenames
    
    def load_train_data(self):
        """
        Charge tous les batches d'entraînement de CIFAR-10
        
        Returns:
            tuple: (x_train, y_train)
                - x_train: array de shape (50000, 32, 32, 3)
                - y_train: array de shape (50000,)
        """
        x_train = []
        y_train = []
        
        #chargé les 5 batches d'entraînement
        for i in range(1, 6):
            file_path = os.path.join(self.data_dir, f'data_batch_{i}')
            data, labels, _ = self._load_batch(file_path)
            x_train.append(data)
            y_train.extend(labels)
        
        #concaténer tous les batches
        x_train = np.concatenate(x_train)
        y_train = np.array(y_train)
        
        # Reshape les images: (50000, 3072) -> (50000, 3, 32, 32) -> (50000, 32, 32, 3)
        x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        
        self.x_train = x_train
        self.y_train = y_train
        
        return x_train, y_train
    
    def load_test_data(self):
        """
        Charge le batch de test de CIFAR-10
        
        Returns:
            tuple: (x_test, y_test)
                - x_test: array de shape (10000, 32, 32, 3)
                - y_test: array de shape (10000,)
        """
        file_path = os.path.join(self.data_dir, 'test_batch')
        data, labels, _ = self._load_batch(file_path)
        
        # Reshape les images
        x_test = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        y_test = np.array(labels)
        
        self.x_test = x_test
        self.y_test = y_test
        
        return x_test, y_test
    
    def load_all_data(self):
        """
        Charge toutes les données (entraînement + test)
        
        Returns:
            tuple: (x_train, y_train, x_test, y_test)
        """
        self.load_train_data()
        self.load_test_data()
        return self.x_train, self.y_train, self.x_test, self.y_test
    
    def normalize(self, data=None, method='0-1'):
        """
        Normalise les données
        
        Args:
            data (np.ndarray, optional): Données à normaliser. 
                                        Si None, normalise x_train et x_test
            method (str): Méthode de normalisation
                - '0-1': Normalisation entre 0 et 1
                - 'standard': Normalisation z-score (mean=0, std=1)
        
        Returns:
            np.ndarray: Données normalisées
        """
        if data is None:
            # Normaliser les données chargées
            if self.x_train is not None:
                self.x_train = self.normalize(self.x_train, method)
            if self.x_test is not None:
                self.x_test = self.normalize(self.x_test, method)
            return self.x_train, self.x_test
        
        if method == '0-1':
            return data.astype('float32') / 255.0
        elif method == 'standard':
            mean = np.mean(data, axis=(0, 1, 2))
            std = np.std(data, axis=(0, 1, 2))
            return (data.astype('float32') - mean) / std
        else:
            raise ValueError(f"Méthode de normalisation inconnue: {method}")
    
    def get_class_distribution(self, labels=None):
        """
        Obtient la distribution des classes
        
        Args:
            labels (np.ndarray, optional): Labels à analyser. Si None, utilise y_train
        
        Returns:
            dict: Dictionnaire {nom_classe: nombre_images}
        """
        if labels is None:
            labels = self.y_train
        
        unique, counts = np.unique(labels, return_counts=True)
        distribution = {self.class_names[label]: count 
                       for label, count in zip(unique, counts)}
        return distribution
    
    def visualize_samples(self, n_samples=10, save_path='code/cifar10_samples.png'):
        """
        Affiche quelques exemples d'images
        
        Args:
            n_samples (int): Nombre d'échantillons à afficher
            save_path (str): Chemin pour sauvegarder l'image
        """
        if self.x_train is None or self.y_train is None:
            raise ValueError("Les données d'entraînement ne sont pas chargées. Appelez load_train_data() d'abord.")
        
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        axes = axes.ravel()
        
        # Gérer la normalisation pour l'affichage
        images = self.x_train[:n_samples]
        if images.max() <= 1.0:
            # Les images sont normalisées, les ramener à 0-255
            images = (images * 255).astype(np.uint8)
        
        for i in range(n_samples):
            axes[i].imshow(images[i])
            axes[i].set_title(f'{self.class_names[self.y_train[i]]}')
            axes[i].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Échantillons sauvegardés dans '{save_path}'")
        plt.show()
    
    def get_subset(self, indices, train=True):
        """
        Obtient un sous-ensemble des données
        
        Args:
            indices (array-like): Indices des échantillons à récupérer
            train (bool): True pour les données d'entraînement, False pour test
        
        Returns:
            tuple: (x_subset, y_subset)
        """
        if train:
            return self.x_train[indices], self.y_train[indices]
        else:
            return self.x_test[indices], self.y_test[indices]
    
    def print_info(self):
        """Affiche les informations sur le dataset"""
        print("=" * 60)
        print("Informations sur le dataset CIFAR-10")
        print("=" * 60)
        
        print(f"\nClasses ({len(self.class_names)}):")
        for i, name in enumerate(self.class_names):
            print(f"  {i}: {name}")
        
        if self.x_train is not None:
            print(f"\nDonnées d'entraînement:")
            print(f"  Shape: {self.x_train.shape}")
            print(f"  Type: {self.x_train.dtype}")
            print(f"  Min/Max: {self.x_train.min():.2f}/{self.x_train.max():.2f}")
            
            distribution = self.get_class_distribution(self.y_train)
            print(f"\nDistribution des classes (entraînement):")
            for class_name, count in distribution.items():
                print(f"  {class_name}: {count} images")
        
        if self.x_test is not None:
            print(f"\nDonnées de test:")
            print(f"  Shape: {self.x_test.shape}")
            print(f"  Type: {self.x_test.dtype}")
            print(f"  Min/Max: {self.x_test.min():.2f}/{self.x_test.max():.2f}")
        
        print("\n" + "=" * 60)


if __name__ == "__main__":
    # Exemple d'utilisation
    print("Exemple d'utilisation de la classe CIFAR10Loader\n")
    
    # Créer une instance du loader
    loader = CIFAR10Loader()
    
    # Charger toutes les données
    print("Chargement des données...")
    x_train, y_train, x_test, y_test = loader.load_all_data()
    
    # Afficher les informations
    loader.print_info()
    
    # Normaliser les données
    print("\nNormalisation des données (0-1)...")
    loader.normalize(method='0-1')
    print(f"Après normalisation: Min/Max = {loader.x_train.min():.2f}/{loader.x_train.max():.2f}")
    
    # Visualiser quelques exemples
    print("\nAffichage de quelques exemples...")
    loader.visualize_samples(n_samples=10)
