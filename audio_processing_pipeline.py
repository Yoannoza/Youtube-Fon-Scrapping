import os
import sys
import csv
import json
import glob
import shutil
import subprocess
import multiprocessing
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from pydub import AudioSegment
from scipy.io import wavfile
import soundfile as sf
import librosa
import noisereduce as nr
from transformers import pipeline
from datasets import Dataset, Audio, Features, Value, ClassLabel
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from concurrent.futures import ProcessPoolExecutor

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("audio_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Configuration globale
INPUT_DIR = "Audios/"
OUTPUT_DIR = "Processed_Audios/"
SEGMENTS_DIR = "Audio_Segments/"
METADATA_FILE = "youtube__fon_audio_dataset.csv"
MAX_WORKERS = multiprocessing.cpu_count() - 1
SAMPLE_RATE = 16000  # Standard pour la plupart des modèles Speech
MIN_SEGMENT_DURATION = 2  # En secondes
MAX_SEGMENT_DURATION = 20  # En secondes
SNR_THRESHOLD = 10  # Rapport signal/bruit minimum acceptable
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Créer les répertoires nécessaires
for directory in [OUTPUT_DIR, SEGMENTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Fonction pour charger les métadonnées des vidéos
def load_metadata(file_path):
    """Charger les métadonnées depuis le fichier CSV"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        logger.info(f"Métadonnées chargées: {len(df)} entrées")
        return df
    except Exception as e:
        logger.error(f"Erreur lors du chargement des métadonnées: {str(e)}")
        sys.exit(1)

# Vérification initiale des fichiers audio
def verify_audio_files(metadata_df):
    """Vérifier l'existence et l'intégrité des fichiers audio"""
    verified_files = []
    corrupted_files = []
    missing_files = []

    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Vérification des fichiers audio"):
        if not pd.isna(row['Audio Path']):
            file_path = row['Audio Path']
            if os.path.exists(file_path):
                try:
                    # Essayer de charger l'audio pour vérifier l'intégrité
                    audio = AudioSegment.from_file(file_path)
                    verified_files.append(row)
                except Exception as e:
                    logger.warning(f"Fichier corrompu: {file_path}, Erreur: {str(e)}")
                    corrupted_files.append(row)
            else:
                logger.warning(f"Fichier non trouvé: {file_path}")
                missing_files.append(row)
        else:
            missing_files.append(row)

    logger.info(f"Vérification terminée: {len(verified_files)} OK, {len(corrupted_files)} corrompus, {len(missing_files)} manquants")
    return pd.DataFrame(verified_files), pd.DataFrame(corrupted_files), pd.DataFrame(missing_files)

# Fonction pour convertir les fichiers MP3 en WAV normalisé
def convert_to_wav(file_path, output_path):
    """Convertir un fichier audio en WAV 16kHz mono et le normaliser"""
    try:
        # Charger l'audio avec librosa pour un rééchantillonnage de haute qualité
        y, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        
        # Normaliser l'audio (mise à l'échelle)
        y = librosa.util.normalize(y) * 0.95  # 95% de l'amplitude maximale pour éviter l'écrêtage
        
        # Enregistrer en WAV
        sf.write(output_path, y, SAMPLE_RATE, subtype='PCM_16')
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la conversion de {file_path}: {str(e)}")
        return False

# Fonction pour détecter la parole dans l'audio
def detect_speech_segments(file_path, vad_model):
    """Utiliser un modèle VAD (Voice Activity Detection) pour détecter les segments de parole"""
    try:
        # Charger l'audio
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Durée totale en secondes
        total_duration = len(y) / sr
        
        # Préparation des segments
        segment_length = 30  # Traiter par segments de 30 secondes
        segments = []
        
        # Traiter l'audio par segments pour éviter les problèmes de mémoire
        for i in range(0, len(y), int(segment_length * sr)):
            end = min(i + int(segment_length * sr), len(y))
            segment = y[i:end]
            
            # Convertir en float32
            waveform = np.array(segment).astype('float32')
            
            # Effectuer la détection de voix
            result = vad_model({"waveform": waveform, "sample_rate": sr})
            
            # Convertir les résultats en segments avec timestamp
            for seg in result["segments"]:
                # Ajuster les timestamps relatifs au segment
                start_time = i/sr + seg["start"]
                end_time = i/sr + seg["end"]
                
                # Filtrer les segments trop courts
                if end_time - start_time >= MIN_SEGMENT_DURATION:
                    segments.append({
                        "start": start_time,
                        "end": min(end_time, total_duration),
                        "confidence": seg["score"]
                    })
        
        # Fusionner les segments proches
        if segments:
            merged_segments = [segments[0]]
            for segment in segments[1:]:
                last_segment = merged_segments[-1]
                # Si le début du segment actuel est proche de la fin du dernier segment
                if segment["start"] - last_segment["end"] < 0.5:  # 500ms de tolérance
                    # Étendre le dernier segment
                    last_segment["end"] = segment["end"]
                    last_segment["confidence"] = (last_segment["confidence"] + segment["confidence"]) / 2
                else:
                    merged_segments.append(segment)
                    
            return merged_segments
        return []
    except Exception as e:
        logger.error(f"Erreur lors de la détection de parole dans {file_path}: {str(e)}")
        return []

# Fonction pour extraire un segment audio
def extract_segment(file_path, start_time, end_time, output_path):
    """Extraire un segment spécifique d'un fichier audio"""
    try:
        # Limiter la durée maximale des segments
        duration = end_time - start_time
        if duration > MAX_SEGMENT_DURATION:
            # Diviser en sous-segments
            sub_segments = []
            current_start = start_time
            while current_start < end_time:
                current_end = min(current_start + MAX_SEGMENT_DURATION, end_time)
                sub_segments.append((current_start, current_end))
                current_start = current_end
            
            # Extraire chaque sous-segment
            extracted_paths = []
            for i, (sub_start, sub_end) in enumerate(sub_segments):
                sub_output_path = output_path.replace(".wav", f"_{i+1}.wav")
                if extract_segment(file_path, sub_start, sub_end, sub_output_path):
                    extracted_paths.append(sub_output_path)
            return extracted_paths
        
        # Charger l'audio complet
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, offset=start_time, duration=duration)
        
        # Réduire le bruit
        y_reduced = nr.reduce_noise(y=y, sr=sr, stationary=False)
        
        # Enregistrer le segment
        sf.write(output_path, y_reduced, sr, subtype='PCM_16')
        return [output_path]
    
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction d'un segment de {file_path}: {str(e)}")
        return []

# Fonction pour évaluer la qualité audio d'un segment
def evaluate_audio_quality(file_path):
    """Évaluer la qualité audio (SNR, niveau sonore, etc.)"""
    try:
        # Charger l'audio
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Calculer le RMS (Root Mean Square) de l'amplitude
        rms = np.sqrt(np.mean(y**2))
        
        # Estimer le SNR (rapport signal/bruit)
        # Méthode simple: comparer les parties de signal fort vs faible
        y_abs = np.abs(y)
        y_sorted = np.sort(y_abs)
        noise_level = np.mean(y_sorted[:int(len(y_sorted)*0.1)])  # 10% les plus bas = bruit
        signal_level = np.mean(y_sorted[int(len(y_sorted)*0.9):])  # 10% les plus hauts = signal
        
        snr = 20 * np.log10(signal_level / (noise_level + 1e-10))  # Éviter division par 0
        
        # Vérifier si l'audio contient principalement du silence
        silence_threshold = 0.01
        silence_percentage = np.sum(y_abs < silence_threshold) / len(y)
        
        # Retourner les métriques de qualité
        return {
            "rms": float(rms),
            "snr": float(snr),
            "silence_percentage": float(silence_percentage),
            "duration": len(y) / sr
        }
    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation de la qualité de {file_path}: {str(e)}")
        return None

# Fonction de traitement complet d'un fichier audio
def process_audio_file(row):
    """Traiter un fichier audio complet: conversion, détection de parole, segmentation"""
    try:
        input_path = row['Audio Path']
        video_id = row['ID']
        base_filename = os.path.basename(input_path)
        output_path = os.path.join(OUTPUT_DIR, f"{video_id}.wav")
        
        # 1. Convertir en WAV normalisé
        if not convert_to_wav(input_path, output_path):
            return {"video_id": video_id, "status": "failed", "error": "Conversion error"}
        
        # 2. Initialiser le modèle VAD
        vad_model = pipeline("automatic-speech-recognition", 
                             model="jonatasgrosman/wav2vec2-large-xlsr-53-english", 
                             chunk_length_s=30,
                             device=DEVICE)
        
        # 3. Détecter les segments de parole
        speech_segments = detect_speech_segments(output_path, vad_model)
        
        if not speech_segments:
            return {"video_id": video_id, "status": "no_speech", "segments": []}
        
        # 4. Extraire et évaluer chaque segment
        segments_info = []
        for i, segment in enumerate(speech_segments):
            segment_output_path = os.path.join(SEGMENTS_DIR, f"{video_id}_segment_{i+1}.wav")
            extracted_paths = extract_segment(output_path, segment["start"], segment["end"], segment_output_path)
            
            for path in extracted_paths:
                # Évaluer la qualité du segment
                quality_metrics = evaluate_audio_quality(path)
                
                if quality_metrics and quality_metrics["snr"] > SNR_THRESHOLD:
                    segments_info.append({
                        "path": path,
                        "start": segment["start"],
                        "end": segment["end"],
                        "duration": quality_metrics["duration"],
                        "quality": quality_metrics,
                        "status": "kept"
                    })
                else:
                    # Supprimer les segments de mauvaise qualité
                    if os.path.exists(path):
                        os.remove(path)
                    segments_info.append({
                        "path": None,
                        "start": segment["start"],
                        "end": segment["end"],
                        "duration": segment["end"] - segment["start"],
                        "quality": quality_metrics,
                        "status": "rejected"
                    })
        
        return {"video_id": video_id, "status": "processed", "segments": segments_info}
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement de {row['ID']}: {str(e)}")
        return {"video_id": row['ID'], "status": "failed", "error": str(e)}

# Fonction pour générer des statistiques sur les segments traités
def generate_statistics(processing_results):
    """Générer des statistiques sur les segments audio traités"""
    try:
        total_videos = len(processing_results)
        processed_videos = sum(1 for res in processing_results if res["status"] == "processed")
        failed_videos = sum(1 for res in processing_results if res["status"] == "failed")
        no_speech_videos = sum(1 for res in processing_results if res["status"] == "no_speech")
        
        all_segments = []
        for res in processing_results:
            if res["status"] == "processed":
                all_segments.extend(res["segments"])
        
        kept_segments = [seg for seg in all_segments if seg["status"] == "kept"]
        rejected_segments = [seg for seg in all_segments if seg["status"] == "rejected"]
        
        total_duration = sum(seg["duration"] for seg in kept_segments)
        hours = int(total_duration // 3600)
        minutes = int((total_duration % 3600) // 60)
        seconds = int(total_duration % 60)
        
        # Générer un rapport
        report = {
            "total_videos": total_videos,
            "processed_videos": processed_videos,
            "failed_videos": failed_videos,
            "no_speech_videos": no_speech_videos,
            "total_segments": len(all_segments),
            "kept_segments": len(kept_segments),
            "rejected_segments": len(rejected_segments),
            "total_duration_hours": hours,
            "total_duration_minutes": minutes,
            "total_duration_seconds": seconds,
            "total_duration_total_seconds": total_duration,
            "average_segment_duration": np.mean([seg["duration"] for seg in kept_segments]) if kept_segments else 0,
            "average_segment_snr": np.mean([seg["quality"]["snr"] for seg in kept_segments if seg["quality"]]) if kept_segments else 0
        }
        
        # Afficher le rapport
        print("\n" + "="*60)
        print("STATISTIQUES DE TRAITEMENT AUDIO")
        print(f"Vidéos traitées: {processed_videos}/{total_videos} ({processed_videos/total_videos*100:.1f}%)")
        print(f"Vidéos sans parole détectée: {no_speech_videos}")
        print(f"Vidéos en échec: {failed_videos}")
        print(f"Segments totaux: {len(all_segments)}")
        print(f"Segments conservés: {len(kept_segments)} ({len(kept_segments)/len(all_segments)*100:.1f}% des segments)")
        print(f"Segments rejetés: {len(rejected_segments)}")
        print(f"Durée totale conservée: {hours}h {minutes}m {seconds}s ({total_duration:.1f} secondes)")
        print(f"Durée moyenne par segment: {report['average_segment_duration']:.2f} secondes")
        print(f"SNR moyen des segments conservés: {report['average_segment_snr']:.2f} dB")
        print("="*60 + "\n")
        
        # Sauvegarder les statistiques dans un fichier JSON
        with open("processing_statistics.json", "w") as f:
            json.dump(report, f, indent=4)
            
        # Visualiser les distributions
        if kept_segments:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            durations = [seg["duration"] for seg in kept_segments]
            sns.histplot(durations, kde=True)
            plt.title("Distribution des durées des segments")
            plt.xlabel("Durée (secondes)")
            
            plt.subplot(2, 2, 2)
            snr_values = [seg["quality"]["snr"] for seg in kept_segments if seg["quality"]]
            sns.histplot(snr_values, kde=True)
            plt.title("Distribution du SNR")
            plt.xlabel("SNR (dB)")
            
            plt.subplot(2, 2, 3)
            rms_values = [seg["quality"]["rms"] for seg in kept_segments if seg["quality"]]
            sns.histplot(rms_values, kde=True)
            plt.title("Distribution du RMS")
            plt.xlabel("RMS")
            
            plt.subplot(2, 2, 4)
            silence_values = [seg["quality"]["silence_percentage"]*100 for seg in kept_segments if seg["quality"]]
            sns.histplot(silence_values, kde=True)
            plt.title("Distribution du pourcentage de silence")
            plt.xlabel("Silence (%)")
            
            plt.tight_layout()
            plt.savefig("audio_quality_distributions.png")
            
        return report
    
    except Exception as e:
        logger.error(f"Erreur lors de la génération des statistiques: {str(e)}")
        return None

# Fonction pour préparer le dataset pour HuggingFace
def prepare_huggingface_dataset(processing_results, metadata_df):
    """Préparer les données pour l'upload sur HuggingFace"""
    try:
        # Collecter les informations sur les segments valides
        dataset_items = []
        
        for result in processing_results:
            if result["status"] == "processed":
                video_id = result["video_id"]
                
                # Trouver les métadonnées correspondantes
                video_info = metadata_df[metadata_df['ID'] == video_id].iloc[0] if any(metadata_df['ID'] == video_id) else None
                
                for i, segment in enumerate(result["segments"]):
                    if segment["status"] == "kept":
                        item = {
                            "audio": segment["path"],
                            "duration": segment["duration"],
                            "video_id": video_id,
                            "segment_id": i+1,
                            "start_time": segment["start"],
                            "end_time": segment["end"],
                            "snr": segment["quality"]["snr"] if segment["quality"] else None
                        }
                        
                        # Ajouter les métadonnées disponibles
                        if video_info is not None:
                            item["title"] = video_info["Title"]
                            item["url"] = video_info["URL"]
                        
                        dataset_items.append(item)
        
        # Créer le dataset HuggingFace
        df = pd.DataFrame(dataset_items)
        
        # Enregistrer les métadonnées dans un CSV
        df.to_csv("huggingface_dataset_metadata.csv", index=False)
        
        # Créer un dataset HuggingFace
        features = Features({
            "audio": Audio(sampling_rate=SAMPLE_RATE),
            "duration": Value("float"),
            "video_id": Value("string"),
            "segment_id": Value("int64"),
            "start_time": Value("float"),
            "end_time": Value("float"),
            "snr": Value("float"),
            "title": Value("string"),
            "url": Value("string")
        })
        
        dataset = Dataset.from_pandas(df, features=features)
        
        # Enregistrer le dataset localement
        dataset.save_to_disk("huggingface_fon_dataset")
        
        logger.info(f"Dataset HuggingFace préparé avec {len(dataset_items)} entrées")
        print(f"Dataset préparé avec {len(dataset_items)} entrées audio")
        
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la préparation du dataset HuggingFace: {str(e)}")
        return False

# Fonction principale
def main():
    print("🚀 DÉBUT DU TRAITEMENT AUDIO")
    
    # 1. Charger les métadonnées
    metadata_df = load_metadata(METADATA_FILE)
    
    # 2. Vérifier les fichiers audio
    valid_files_df, corrupted_files_df, missing_files_df = verify_audio_files(metadata_df)
    
    print(f"Traitement de {len(valid_files_df)} fichiers audio valides...")
    
    # 3. Traiter les fichiers en parallèle
    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_audio_file, row) for _, row in valid_files_df.iterrows()]
        
        for future in tqdm(futures, total=len(futures), desc="Traitement des fichiers audio"):
            results.append(future.result())
    
    # 4. Générer des statistiques
    statistics = generate_statistics(results)
    
    # 5. Préparer le dataset pour HuggingFace
    prepare_huggingface_dataset(results, metadata_df)
    
    print("\n🏁 TRAITEMENT AUDIO TERMINÉ\n")

if __name__ == "__main__":
    main()