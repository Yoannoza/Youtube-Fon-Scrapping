import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import random
import logging
import json
from tqdm import tqdm
from pathlib import Path
from scipy import signal
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("audio_augmentation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Configuration
SEGMENTS_DIR = "Audio_Segments/"
AUGMENTED_DIR = "Augmented_Audio/"
METADATA_FILE = "huggingface_dataset_metadata.csv"
SAMPLE_RATE = 16000
MAX_WORKERS = multiprocessing.cpu_count() - 1
NUM_AUGMENTATIONS_PER_FILE = 2  # Nombre d'augmentations par fichier original

# Créer le répertoire pour les données augmentées
os.makedirs(AUGMENTED_DIR, exist_ok=True)

# Classes pour les différentes techniques d'augmentation
class AudioAugmenter:
    """Classe de base pour les augmentations audio"""
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
    def augment(self, audio):
        """Méthode d'augmentation à implémenter dans les sous-classes"""
        raise NotImplementedError("Les sous-classes doivent implémenter cette méthode")

class TimeStretch(AudioAugmenter):
    """Étirement temporel (sans changement de pitch)"""
    def __init__(self, sample_rate=16000, min_rate=0.85, max_rate=1.15):
        super().__init__(sample_rate)
        self.min_rate = min_rate
        self.max_rate = max_rate
        
    def augment(self, audio):
        rate = random.uniform(self.min_rate, self.max_rate)
        return librosa.effects.time_stretch(audio, rate=rate)

class PitchShift(AudioAugmenter):
    """Modification de la hauteur (pitch)"""
    def __init__(self, sample_rate=16000, min_steps=-3, max_steps=3):
        super().__init__(sample_rate)
        self.min_steps = min_steps
        self.max_steps = max_steps
        
    def augment(self, audio):
        n_steps = random.uniform(self.min_steps, self.max_steps)
        return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)

class AddNoise(AudioAugmenter):
    """Ajout de bruit blanc à l'audio"""
    def __init__(self, sample_rate=16000, min_snr=15, max_snr=30):
        super().__init__(sample_rate)
        self.min_snr = min_snr
        self.max_snr = max_snr
        
    def augment(self, audio):
        snr = random.uniform(self.min_snr, self.max_snr)
        # Calculer la puissance du signal
        audio_power = np.mean(audio ** 2)
        # Calculer la puissance du bruit basée sur le SNR
        noise_power = audio_power / (10 ** (snr / 10))
        # Générer le bruit
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        # Ajouter le bruit au signal
        return audio + noise

class RoomSimulation(AudioAugmenter):
    """Simulation d'effets de réverbération d'une pièce"""
    def __init__(self, sample_rate=16000, min_room_size=0.1, max_room_size=0.5):
        super().__init__(sample_rate)
        self.min_room_size = min_room_size
        self.max_room_size = max_room_size
        
    def augment(self, audio):
        room_size = random.uniform(self.min_room_size, self.max_room_size)
        
        # Créer un filtre de réverbération simple
        n = int(room_size * self.sample_rate)
        decay = np.exp(-np.linspace(0, 5, n))
        ir = np.random.randn(n) * decay
        
        # Appliquer la convolution
        return signal.convolve(audio, ir, mode='same')

class SpeedTuning(AudioAugmenter):
    """Modification de la vitesse (affecte aussi le pitch)"""
    def __init__(self, sample_rate=16000, min_speed=0.9, max_speed=1.1):
        super().__init__(sample_rate)
        self.min_speed = min_speed
        self.max_speed = max_speed
        
    def augment(self, audio):
        speed_factor = random.uniform(self.min_speed, self.max_speed)
        # Modification directe du taux d'échantillonnage (rééchantillonnage)
        return librosa.resample(audio, orig_sr=self.sample_rate, 
                                target_sr=int(self.sample_rate * speed_factor))

class VolumeAdjustment(AudioAugmenter):
    """Ajustement du volume"""
    def __init__(self, sample_rate=16000, min_gain_db=-6, max_gain_db=3):
        super().__init__(sample_rate)
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db
        
    def augment(self, audio):
        gain_db = random.uniform(self.min_gain_db, self.max_gain_db)
        gain_factor = 10 ** (gain_db / 20)
        return audio * gain_factor

class CompressionSimulation(AudioAugmenter):
    """Simulation de compression audio légère"""
    def __init__(self, sample_rate=16000, threshold=0.3, ratio=2):
        super().__init__(sample_rate)
        self.threshold = threshold
        self.ratio = ratio
        
    def augment(self, audio):
        # Compression simple
        audio_abs = np.abs(audio)
        mask = audio_abs > self.threshold
        audio_compressed = np.copy(audio)
        audio_compressed[mask] = (
            np.sign(audio[mask]) * 
            (self.threshold + (audio_abs[mask] - self.threshold) / self.ratio)
        )
        return audio_compressed

class BandPassFilter(AudioAugmenter):
    """Application d'un filtre passe-bande pour simuler différentes qualités d'enregistrement"""
    def __init__(self, sample_rate=16000, min_low_cutoff=300, max_low_cutoff=600, 
                 min_high_cutoff=3000, max_high_cutoff=5000):
        super().__init__(sample_rate)
        self.min_low_cutoff = min_low_cutoff
        self.max_low_cutoff = max_low_cutoff
        self.min_high_cutoff = min_high_cutoff
        self.max_high_cutoff = max_high_cutoff
        
    def augment(self, audio):
        low_cutoff = random.uniform(self.min_low_cutoff, self.max_low_cutoff)
        high_cutoff = random.uniform(self.min_high_cutoff, self.max_high_cutoff)
        
        # Normaliser les fréquences de coupure
        nyquist = self.sample_rate / 2
        low = low_cutoff / nyquist
        high = high_cutoff / nyquist
        
        # Concevoir le filtre
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Appliquer le filtre
        return signal.filtfilt(b, a, audio)


# Fonction pour appliquer des augmentations aléatoires à un fichier audio
def apply_random_augmentations(file_path, output_path, num_augmentations=1):
    """Applique des augmentations aléatoires à un fichier audio"""
    try:
        # Charger l'audio
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Liste des augmentations disponibles avec leurs poids (probabilités)
        augmenters = [
            (TimeStretch(sample_rate=sr), 0.7),
            (PitchShift(sample_rate=sr), 0.7),
            (AddNoise(sample_rate=sr), 0.5),
            (RoomSimulation(sample_rate=sr), 0.4),
            (SpeedTuning(sample_rate=sr), 0.6),
            (VolumeAdjustment(sample_rate=sr), 0.8),
            (CompressionSimulation(sample_rate=sr), 0.3),
            (BandPassFilter(sample_rate=sr), 0.4)
        ]
        
        # Générer plusieurs versions augmentées
        results = []
        for i in range(num_augmentations):
            # Copie de l'audio original
            augmented = np.copy(y)
            
            # Nombre d'augmentations à appliquer (1 à 3)
            num_transforms = random.randint(1, 3)
            
            # Sélectionner des augmentations aléatoires en fonction de leurs poids
            selected_augmenters = random.choices(
                [aug for aug, _ in augmenters],
                weights=[weight for _, weight in augmenters],
                k=num_transforms
            )
            
            # Appliquer les augmentations sélectionnées
            for augmenter in selected_augmenters:
                augmented = augmenter.augment(augmented)
            
            # Normaliser l'audio augmenté
            augmented = librosa.util.normalize(augmented)
            
            # Créer le nom de fichier
            file_basename = os.path.basename(file_path)
            name_parts = os.path.splitext(file_basename)
            aug_filename = f"{name_parts[0]}_aug{i+1}{name_parts[1]}"
            aug_path = os.path.join(output_path, aug_filename)
            
            # Enregistrer l'audio augmenté
            sf.write(aug_path, augmented, sr, subtype='PCM_16')
            
            results.append({
                "original_path": file_path,
                "augmented_path": aug_path,
                "augmentations": [type(aug).__name__ for aug in selected_augmenters]
            })
            
        return results
    
    except Exception as e:
        logger.error(f"Erreur lors de l'augmentation de {file_path}: {str(e)}")
        return []

def process_augmentation_batch(files_batch):
    """Traite un batch de fichiers pour l'augmentation"""
    results = []
    for file_path in files_batch:
        res = apply_random_augmentations(file_path, AUGMENTED_DIR, NUM_AUGMENTATIONS_PER_FILE)
        results.extend(res)
    return results

def main():
    print("🚀 DÉBUT DE L'AUGMENTATION DES DONNÉES AUDIO")
    
    try:
        # Charger les métadonnées
        if os.path.exists(METADATA_FILE):
            metadata_df = pd.read_csv(METADATA_FILE)
            audio_files = metadata_df['audio'].tolist()
        else:
            # Sinon, utiliser tous les fichiers WAV du répertoire des segments
            audio_files = glob.glob(os.path.join(SEGMENTS_DIR, "*.wav"))
        
        print(f"Fichiers audio à augmenter: {len(audio_files)}")
        
        # Traiter les fichiers en parallèle
        augmentation_results = []
        
        # Diviser en batches pour le traitement parallèle
        batch_size = max(1, len(audio_files) // MAX_WORKERS)
        batches = [audio_files[i:i+batch_size] for i in range(0, len(audio_files), batch_size)]
        
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for batch_results in tqdm(executor.map(process_augmentation_batch, batches), 
                                      total=len(batches), 
                                      desc="Traitement des lots d'augmentation"):
                augmentation_results.extend(batch_results)
        
        # Générer un rapport d'augmentation
        print(f"Génération de {len(augmentation_results)} versions augmentées")
        
        # Enregistrer les métadonnées d'augmentation
        augmentation_df = pd.DataFrame(augmentation_results)
        augmentation_df.to_csv("augmentation_metadata.csv", index=False)
        
        # Enregistrer un résumé JSON
        augmentation_summary = {
            "total_original_files": len(audio_files),
            "total_augmented_files": len(augmentation_results),
            "augmentation_ratio": len(augmentation_results) / len(audio_files) if audio_files else 0,
            "augmentation_techniques_usage": {}
        }
        
        # Compter l'utilisation de chaque technique d'augmentation
        all_techniques = []
        for result in augmentation_results:
            all_techniques.extend(result["augmentations"])
        
        for technique in set(all_techniques):
            count = all_techniques.count(technique)
            augmentation_summary["augmentation_techniques_usage"][technique] = {
                "count": count,
                "percentage": count / len(all_techniques) * 100 if all_techniques else 0
            }
        
        with open("augmentation_summary.json", "w") as f:
            json.dump(augmentation_summary, f, indent=4)
        
        print("\n" + "="*60)
        print("RÉSUMÉ DE L'AUGMENTATION")
        print(f"Fichiers originaux: {len(audio_files)}")
        print(f"Versions augmentées générées: {len(augmentation_results)}")
        print(f"Facteur d'augmentation: {augmentation_summary['augmentation_ratio']:.2f}x")
        print("Techniques d'augmentation utilisées:")
        for technique, stats in augmentation_summary["augmentation_techniques_usage"].items():
            print(f"  • {technique}: {stats['count']} fois ({stats['percentage']:.1f}%)")
        print("="*60 + "\n")
        
        print("🏁 AUGMENTATION DES DONNÉES TERMINÉE")
        
    except Exception as e:
        logger.error(f"Erreur dans le processus d'augmentation: {str(e)}")
        print(f"❌ Erreur: {str(e)}")

if __name__ == "__main__":
    main()