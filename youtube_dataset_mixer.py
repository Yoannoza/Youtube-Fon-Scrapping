import os
import csv
import glob
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import numpy as np
import argparse
from collections import Counter

# Configuration
AUDIO_DIR = "Audios/"
OUTPUT_CSV = "youtube__fon_audio_dataset.csv"
STATS_FILE = "dataset_stats.txt"

def parse_arguments():
    parser = argparse.ArgumentParser(description='Mixer et analyser des CSV de données YouTube')
    parser.add_argument('--csv_files', nargs='+', default=glob.glob("videos_downloaded_*.csv"), 
                        help='Liste des fichiers CSV à combiner')
    parser.add_argument('--output', default=OUTPUT_CSV, help='Nom du fichier CSV de sortie')
    parser.add_argument('--stats', default=STATS_FILE, help='Nom du fichier de statistiques')
    parser.add_argument('--plots', action='store_true', help='Générer des graphiques')
    return parser.parse_args()

def verify_audio_file(file_path):
    """Vérifie si un fichier audio existe et sa taille"""
    if not file_path or file_path == '':
        return {'exists': False, 'size': 0, 'status': 'missing_path'}
        
    if not os.path.exists(file_path):
        return {'exists': False, 'size': 0, 'status': 'file_not_found'}
        
    size = os.path.getsize(file_path)
    if size < 1024:  # Moins de 1KB est probablement corrompu
        return {'exists': True, 'size': size, 'status': 'corrupted'}
        
    return {'exists': True, 'size': size, 'status': 'ok'}

def format_size(size_bytes):
    """Convertit une taille en bytes en format lisible"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes/1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes/(1024*1024):.2f} MB"
    else:
        return f"{size_bytes/(1024*1024*1024):.2f} GB"

def mix_and_analyze_csv(csv_files, output_file):
    """Combine plusieurs fichiers CSV et vérifie les fichiers audio"""
    # Charger les données
    all_data = []
    unique_ids = set()
    
    print(f"🔄 Chargement et fusion de {len(csv_files)} fichiers CSV...")
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
            # Standardiser les noms de colonnes
            if 'ID' in df.columns:
                df.rename(columns={'ID': 'video_id'}, inplace=True)
            if 'Status' in df.columns:
                df.rename(columns={'Status': 'status'}, inplace=True)
            if 'Audio Path' in df.columns:
                df.rename(columns={'Audio Path': 'audio_path'}, inplace=True)
            if 'Duration (minutes)' in df.columns:
                df.rename(columns={'Duration (minutes)': 'duration'}, inplace=True)
            
            # Filtrer les doublons 
            new_rows = []
            for _, row in df.iterrows():
                if row['video_id'] not in unique_ids:
                    unique_ids.add(row['video_id'])
                    new_rows.append(row.to_dict())
            
            all_data.extend(new_rows)
            print(f"  ✓ {csv_file}: {len(new_rows)} nouvelles entrées ajoutées")
            
        except Exception as e:
            print(f"  ❌ Erreur lors du traitement de {csv_file}: {str(e)}")
    
    # Créer un DataFrame combiné
    combined_df = pd.DataFrame(all_data)
    
    # Vérifier l'existence et l'état des fichiers audio
    print("\n🔍 Vérification des fichiers audio...")
    
    # Ajouter des colonnes pour l'analyse des fichiers
    combined_df['audio_exists'] = False
    combined_df['audio_size'] = 0
    combined_df['audio_status'] = 'unknown'
    
    for i, row in tqdm(combined_df.iterrows(), total=len(combined_df), desc="Vérification des fichiers"):
        audio_result = verify_audio_file(row['audio_path'])
        combined_df.at[i, 'audio_exists'] = audio_result['exists']
        combined_df.at[i, 'audio_size'] = audio_result['size']
        combined_df.at[i, 'audio_status'] = audio_result['status']
    
    # Sauvegarder le DataFrame combiné
    combined_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n✅ Données combinées sauvegardées dans {output_file}")
    
    return combined_df

def generate_statistics(df):
    """Génère des statistiques détaillées sur le dataset"""
    stats = []
    
    # Statistiques générales
    stats.append("=" * 60)
    stats.append("STATISTIQUES GÉNÉRALES DU DATASET YOUTUBE")
    stats.append("=" * 60)
    
    # Nombre total de vidéos
    total_videos = len(df)
    stats.append(f"Nombre total de vidéos : {total_videos}")
    
    # Statistiques sur l'état des audios
    audio_statuses = df['audio_status'].value_counts().to_dict()
    stats.append("\nÉTAT DES FICHIERS AUDIO:")
    stats.append(f"  ✓ Fichiers OK : {audio_statuses.get('ok', 0)} ({audio_statuses.get('ok', 0)/total_videos*100:.1f}%)")
    stats.append(f"  ⚠ Fichiers corrompus : {audio_statuses.get('corrupted', 0)} ({audio_statuses.get('corrupted', 0)/total_videos*100:.1f}%)")
    stats.append(f"  ❌ Fichiers non trouvés : {audio_statuses.get('file_not_found', 0)} ({audio_statuses.get('file_not_found', 0)/total_videos*100:.1f}%)")
    stats.append(f"  ❓ Chemins manquants : {audio_statuses.get('missing_path', 0)} ({audio_statuses.get('missing_path', 0)/total_videos*100:.1f}%)")
    
    # Statistiques sur les statuts de téléchargement
    if 'status' in df.columns:
        download_statuses = df['status'].value_counts().to_dict()
        stats.append("\nSTATUT DE TÉLÉCHARGEMENT:")
        for status, count in download_statuses.items():
            stats.append(f"  • {status}: {count} ({count/total_videos*100:.1f}%)")
    
    # Calcul de la durée totale
    if 'duration' in df.columns:
        df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
        total_duration = df[df['audio_status'] == 'ok']['duration'].sum()
        hours = int(total_duration // 60)
        minutes = int(total_duration % 60)
        stats.append(f"\nDURÉE TOTALE DES AUDIOS VALIDES: {hours}h {minutes}m ({total_duration:.1f} minutes)")
        
        # Durée moyenne
        avg_duration = df[df['audio_status'] == 'ok']['duration'].mean()
        stats.append(f"Durée moyenne par vidéo: {avg_duration:.2f} minutes")
        
        # Répartition des durées
        duration_ranges = [
            (0, 5, "Très court (0-5 min)"),
            (5, 10, "Court (5-10 min)"),
            (10, 20, "Moyen (10-20 min)"),
            (20, 60, "Long (20-60 min)"),
            (60, float('inf'), "Très long (60+ min)")
        ]
        
        stats.append("\nRÉPARTITION PAR DURÉE:")
        for min_dur, max_dur, label in duration_ranges:
            count = len(df[(df['duration'] >= min_dur) & (df['duration'] < max_dur) & (df['audio_status'] == 'ok')])
            stats.append(f"  • {label}: {count} vidéos ({count/len(df[df['audio_status'] == 'ok'])*100:.1f}%)")
    
    # Statistiques sur la taille des fichiers
    valid_files = df[df['audio_status'] == 'ok']
    if not valid_files.empty:
        total_size = valid_files['audio_size'].sum()
        avg_size = valid_files['audio_size'].mean()
        stats.append(f"\nTAILLE TOTALE DES FICHIERS: {format_size(total_size)}")
        stats.append(f"Taille moyenne par fichier: {format_size(avg_size)}")
        
        # Répartition des tailles
        size_ranges = [
            (0, 1024*1024, "Petit (<1 MB)"),
            (1024*1024, 5*1024*1024, "Moyen (1-5 MB)"),
            (5*1024*1024, 20*1024*1024, "Grand (5-20 MB)"),
            (20*1024*1024, float('inf'), "Très grand (>20 MB)")
        ]
        
        stats.append("\nRÉPARTITION PAR TAILLE:")
        for min_size, max_size, label in size_ranges:
            count = len(valid_files[(valid_files['audio_size'] >= min_size) & (valid_files['audio_size'] < max_size)])
            stats.append(f"  • {label}: {count} fichiers ({count/len(valid_files)*100:.1f}%)")
    
    # Vidéos manquantes
    missing_videos = df[df['audio_status'] != 'ok']
    if not missing_videos.empty:
        stats.append("\nVIDÉOS À RETÉLÉCHARGER:")
        stats.append(f"Nombre total: {len(missing_videos)}")
        if 'duration' in missing_videos.columns:
            missing_duration = missing_videos['duration'].sum()
            missing_hours = int(missing_duration // 60)
            missing_minutes = int(missing_duration % 60)
            stats.append(f"Durée manquante: {missing_hours}h {missing_minutes}m ({missing_duration:.1f} minutes)")
    
    stats.append("\n" + "=" * 60)
    
    return "\n".join(stats)

def generate_plots(df, prefix="plot"):
    """Génère des graphiques pour visualiser les données"""
    if len(df) == 0:
        print("Pas de données disponibles pour les graphiques")
        return
    
    print("\n📊 Génération des graphiques...")
    
    # Configuration de seaborn
    sns.set_theme(style="whitegrid")
    
    # 1. Répartition des statuts audio
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='audio_status', data=df, palette='viridis')
    plt.title('Répartition des statuts des fichiers audio')
    plt.xlabel('Statut')
    plt.ylabel('Nombre de fichiers')
    plt.tight_layout()
    plt.savefig(f"{prefix}_audio_status.png")
    
    # 2. Histogramme des durées
    if 'duration' in df.columns:
        plt.figure(figsize=(12, 6))
        # Filtrer les valeurs aberrantes
        duration_data = df[df['duration'] <= df['duration'].quantile(0.99)]
        ax = sns.histplot(data=duration_data, x='duration', bins=30, kde=True)
        plt.title('Distribution des durées des vidéos')
        plt.xlabel('Durée (minutes)')
        plt.ylabel('Nombre de vidéos')
        plt.tight_layout()
        plt.savefig(f"{prefix}_duration_histogram.png")
    
    # 3. Relation entre taille et durée
    if 'duration' in df.columns and 'audio_size' in df.columns:
        plt.figure(figsize=(10, 6))
        valid_data = df[(df['audio_status'] == 'ok') & (df['duration'] <= df['duration'].quantile(0.99))]
        valid_data['audio_size_mb'] = valid_data['audio_size'] / (1024*1024)
        ax = sns.scatterplot(data=valid_data, x='duration', y='audio_size_mb', alpha=0.6)
        plt.title('Relation entre durée et taille des fichiers')
        plt.xlabel('Durée (minutes)')
        plt.ylabel('Taille (MB)')
        plt.tight_layout()
        plt.savefig(f"{prefix}_size_vs_duration.png")
    
    print(f"  ✓ Graphiques sauvegardés avec préfixe '{prefix}'")

def main():
    args = parse_arguments()
    
    print("🚀 ANALYSE DU DATASET YOUTUBE")
    print("-" * 60)
    
    # Vérifier si les fichiers existent
    if not args.csv_files:
        print("❌ Aucun fichier CSV trouvé!")
        return
    
    print(f"Fichiers CSV à traiter: {', '.join(args.csv_files)}")
    
    # Fusionner et analyser les CSV
    combined_df = mix_and_analyze_csv(args.csv_files, args.output)
    
    # Générer les statistiques
    print("\n📊 Génération des statistiques...")
    stats_text = generate_statistics(combined_df)
    
    # Écrire les statistiques dans un fichier
    with open(args.stats, 'w', encoding='utf-8') as f:
        f.write(stats_text)
    
    print(f"✅ Statistiques sauvegardées dans {args.stats}")
    
    # Afficher les statistiques principales
    print("\n" + stats_text)
    
    # Générer des graphiques si demandé
    if args.plots:
        generate_plots(combined_df)
    
    # Générer un fichier de vidéos à retélécharger
    missing_videos = combined_df[combined_df['audio_status'] != 'ok']
    if not missing_videos.empty:
        missing_file = "videos_to_redownload.csv"
        missing_videos.to_csv(missing_file, index=False)
        print(f"\n⚠️ Liste des {len(missing_videos)} vidéos à retélécharger sauvegardée dans {missing_file}")
    
    print("\n🏁 ANALYSE TERMINÉE")

if __name__ == "__main__":
    main()