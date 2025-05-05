import os
from dotenv import load_dotenv
import csv
import random
import isodate
import logging
import time
import re
import concurrent.futures
from datetime import timedelta
from googleapiclient.discovery import build
import requests
from tqdm import tqdm
from playwright.sync_api import sync_playwright

load_dotenv()
# Configuration du logging (minimaliste)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("youtube_download.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Configuration
API_KEY = os.getenv("GOOGLE_API")  # Utilisez votre propre clé API
AUDIO_DIR = "Audios/"  # Répertoire local pour les fichiers audio
MAX_DOWNLOAD_RETRIES = 5  # Nombre de tentatives de téléchargement
MAX_WORKERS = 10  # Nombre de téléchargements simultanés
DELAY_BETWEEN_DOWNLOADS = (3, 10)  # Intervalle réduit pour plus de vitesse

# Système de rotation de User-Agents
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:94.0) Gecko/20100101 Firefox/94.0"
]

# Liste des playlists
PLAYLISTS = [
    # "PLNbBDYo93kPTQNzFmHd5AtisArjdWMRO-", "PL3-sPokHUCobRjs9VfcIwgnjtweYxoeME",
    # "PLYI_vH9CKTesTAcexTubA2rVqPotud-Fl", "PLD-g9JtxYcY4wmDhN69V7SaR7y7nRIrGv",
    # "PLD-g9JtxYcY7LI6zMwiQ-aGuKEQwh6DEX", "PLD-g9JtxYcY51VQRtGOh6yqGYx4cXuFtg",
    # "PLLzGd5fgj8FwHUTL3KiLXNjSmGWaUYqfA", "PLLzGd5fgj8FyBQ2PKD70p7hQjSPXGFLMv",
    # "PLNbUwjT0YUDPd6Cn9xdWl4WSGqTtS6JxB", "PLNbUwjT0YUDMFth1vfKCToaw3lkqoh0oC",
    # "PLNbUwjT0YUDMVt1Yt74EWty0SAg43swQu", "PLNbUwjT0YUDNsBm3PSpQdt4pYM1OIKlc-",
    # "PLNbUwjT0YUDOkhP-mHfgfZFmqvMrF-Jy0", "PLNbUwjT0YUDM_q2slswYHmh8HEjKLID_1",
    "PLp8fd-JP1pLqDoMJ_VKrNFkoIsOv76eI_", "PL36d3riubqC1c6ayEY1LTRHG7pUB7hAvf",
    "PL36d3riubqC0w9onxGXQIh0lPTmi6L4Sf", "PLVYNpA4FYAHytNDBF9OhvYjnqgo4SGVCu",
    "PL6x4b-eXwRRYSbXbZSH4ExkQxCafXpgX7"
]

# Création du répertoire de stockage s'il n'existe pas
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)

def nettoyer_nom_fichier(titre):
    """Nettoie le nom du fichier pour éviter les caractères illégaux"""
    titre = re.sub(r'[\\/*?:"<>|]', "", titre)  # Supprime les caractères illégaux
    return titre.strip()[:150]  # Limite à 150 caractères

def telecharger_fichier(url, nom_fichier):
    """Télécharge un fichier avec barre de progression minimale"""
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            with open(nom_fichier, 'wb') as f, tqdm(
                desc=os.path.basename(nom_fichier),
                total=total,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                leave=False
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
            return True
    except Exception as e:
        logger.error(f"Téléchargement échoué pour {nom_fichier}: {str(e)}")
        return False

def telecharger_video_savetube(video_info):
    """Télécharge une vidéo YouTube en MP3 via SaveTube"""
    video_id = video_info['video_id']
    url_video = video_info['url']
    mp3_path = os.path.join(AUDIO_DIR, f"{video_id}.mp3")
    
    # Vérifier si le fichier existe déjà
    if os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 0:
        logger.info(f"Existant: {video_id}")
        video_info['audio_path'] = mp3_path
        video_info['status'] = 'success'
        return video_info
    
    logger.info(f"Début téléchargement: {video_id}")
    
    for tentative in range(MAX_DOWNLOAD_RETRIES):
        try:
            with sync_playwright() as p:
                # Configuration du navigateur
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent=random.choice(USER_AGENTS),
                    viewport={'width': 1280, 'height': 720}
                )
                page = context.new_page()
                
                # Bloquer les publicités
                def block_ads(route, request):
                    ads_domains = ["doubleclick.net", "googlesyndication.com", "adservice.google.com"]
                    if any(d in request.url for d in ads_domains):
                        route.abort()
                    else:
                        route.continue_()
                
                page.route("**/*", block_ads)
                
                # Accès à SaveTube
                page.goto("https://yt.savetube.me/1kejjj1", timeout=30000)
                
                # Remplir le formulaire et soumettre
                page.wait_for_selector("input.search-input", timeout=10000)
                page.fill("input.search-input", url_video)
                page.click("button:text('Get Video')")
                
                # Attendre la page de résultats
                page.wait_for_selector("#downloadSection", timeout=20000)
                
                # Récupérer le titre
                titre_raw = page.query_selector("h3.text-left").inner_text()
                titre_video = nettoyer_nom_fichier(titre_raw)
                
                # Sélectionner la qualité MP3
                page.select_option("select#quality", label="MP3 320kbps")
                
                # Obtenir le lien de téléchargement
                page.click("button:has-text('Get Link')")
                page.wait_for_url("**/start-download**", timeout=30000)
                page.wait_for_selector("a.text-white:has-text('Download')", timeout=20000)
                
                btn = page.query_selector("a.text-white:has-text('Download')")
                download_link = btn.get_attribute("href")
                
                # Fermeture du navigateur pour libérer les ressources
                browser.close()
                
                # Téléchargement avec barre de progression minimaliste
                if telecharger_fichier(download_link, mp3_path):
                    logger.info(f"Succès: {video_id}")
                    video_info['audio_path'] = mp3_path
                    video_info['status'] = 'success'
                    return video_info
        
        except Exception as e:
            details = f"Erreur pour {video_id} (tentative {tentative+1}/{MAX_DOWNLOAD_RETRIES}): {str(e)}"
            logger.error(details)
            
            # Pause avant la prochaine tentative
            if tentative < MAX_DOWNLOAD_RETRIES - 1:
                time.sleep(random.uniform(2, 5))
    
    # Si toutes les tentatives ont échoué
    video_info['audio_path'] = None
    video_info['status'] = 'failed'
    return video_info

def get_video_details(youtube, video_id):
    """Récupère les détails d'une vidéo YouTube via l'API"""
    try:
        response = youtube.videos().list(
            part='snippet,contentDetails',
            id=video_id
        ).execute()

        if not response['items']:
            return None

        item = response['items'][0]
        title = item['snippet']['title']
        duration_iso = item['contentDetails']['duration']
        duration_minutes = isodate.parse_duration(duration_iso).total_seconds() / 60
        captions = "yes" if item['contentDetails'].get('caption') == "true" else "no"
        
        return title, duration_minutes, captions

    except Exception as e:
        logger.error(f"Erreur détails vidéo {video_id}: {str(e)}")
        return None

def get_videos_from_playlist(youtube, playlist_id):
    """Récupère les vidéos d'une playlist YouTube"""
    videos = []
    next_page_token = None

    try:
        while True:
            request = youtube.playlistItems().list(
                part='snippet',
                playlistId=playlist_id,
                maxResults=50,
                pageToken=next_page_token
            )
            response = request.execute()
            
            for item in response['items']:
                video_id = item['snippet']['resourceId']['videoId']
                details = get_video_details(youtube, video_id)
                
                if details is None:
                    continue
                    
                title, duration, captions = details
                videos.append({
                    'video_id': video_id,
                    'url': f'https://www.youtube.com/watch?v={video_id}',
                    'title': title,
                    'duration': duration,
                    'captions': captions
                })
                
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
                
        logger.info(f"Playlist {playlist_id}: {len(videos)} vidéos trouvées")
        return videos
        
    except Exception as e:
        logger.error(f"Erreur récupération playlist {playlist_id}: {str(e)}")
        return []

def afficher_stats(videos_info):
    """Affiche les statistiques actuelles des téléchargements"""
    total_videos = len(videos_info)
    success_count = sum(1 for v in videos_info if v.get('status') == 'success')
    failed_count = sum(1 for v in videos_info if v.get('status') == 'failed')
    pending_count = total_videos - success_count - failed_count
    
    # Calculer la durée totale en minutes
    total_duration = sum(v.get('duration', 0) for v in videos_info if v.get('status') == 'success')
    hours = int(total_duration // 60)
    minutes = int(total_duration % 60)
    
    print("\n" + "="*50)
    print(f"STATISTIQUES DU DATASET")
    print(f"Total vidéos: {total_videos}")
    print(f"Téléchargées: {success_count} ({success_count/total_videos*100:.1f}%)")
    print(f"En attente: {pending_count}")
    print(f"Échecs: {failed_count}")
    print(f"Durée totale téléchargée: {hours}h {minutes}m ({total_duration:.1f} minutes)")
    print("="*50 + "\n")

def main():
    print("🚀 DÉBUT DU TÉLÉCHARGEMENT MASSIF YOUTUBE")
    start_time = time.time()
    
    # 1. Initialisation de l'API YouTube
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    all_videos = []

    # 2. Récupération des vidéos depuis les playlists
    for playlist_id in PLAYLISTS:
        logger.info(f"Traitement playlist: {playlist_id}")
        vids = get_videos_from_playlist(youtube, playlist_id)
        all_videos.extend(vids)

    # 3. Suppression des doublons
    videos_unique = list({video['video_id']: video for video in all_videos}.values())
    print(f"Nombre total de vidéos à télécharger: {len(videos_unique)}")
    
    # 4. Calcul de la durée totale estimée
    total_estimated_duration = sum(video['duration'] for video in videos_unique)
    hours = int(total_estimated_duration // 60)
    minutes = int(total_estimated_duration % 60)
    print(f"Durée totale estimée du dataset: {hours}h {minutes}m ({total_estimated_duration:.1f} minutes)")

    # 5. Téléchargement parallèle des vidéos
    print(f"Démarrage du téléchargement parallèle avec {MAX_WORKERS} workers...")
    
    # Initialiser les statuts
    for video in videos_unique:
        video['status'] = 'pending'
    
    # Créer un pool de workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Soumettre les tâches
        future_to_video = {executor.submit(telecharger_video_savetube, video): video for video in videos_unique}
        
        # Traiter les résultats au fur et à mesure qu'ils arrivent
        completed = 0
        for future in concurrent.futures.as_completed(future_to_video):
            video = future_to_video[future]
            try:
                result = future.result()
                if result['status'] == 'success':
                    # Mettre à jour le statut dans la liste originale
                    video_index = videos_unique.index(video)
                    videos_unique[video_index] = result
                
                # Afficher les statistiques actualisées tous les 5 téléchargements
                completed += 1
                if completed % 5 == 0 or completed == len(videos_unique):
                    afficher_stats(videos_unique)
                
            except Exception as e:
                logger.error(f"Exception dans le worker pour {video['video_id']}: {str(e)}")
                video['status'] = 'failed'

    # 6. Exporter les informations dans un fichier CSV
    output_csv = "videos_downloaded_3.csv"
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # En-têtes
            writer.writerow(['ID', 'URL', 'Title', 'Duration (minutes)', 'Status', 'Audio Path'])
            for video in videos_unique:
                writer.writerow([
                    video['video_id'],
                    video['url'],
                    video['title'],
                    round(video['duration'], 2),
                    video.get('status', 'unknown'),
                    video.get('audio_path', '')
                ])
        print(f"📄 Fichier CSV généré: {output_csv}")
    except Exception as e:
        logger.error(f"Erreur génération CSV: {str(e)}")

    # 7. Statistiques finales
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed_time)))
    
    successful_downloads = sum(1 for v in videos_unique if v.get('status') == 'success')
    failed_downloads = sum(1 for v in videos_unique if v.get('status') == 'failed')
    
    print("\n" + "="*60)
    print("RÉSUMÉ FINAL")
    print(f"Téléchargements réussis: {successful_downloads}/{len(videos_unique)} ({successful_downloads/len(videos_unique)*100:.1f}%)")
    print(f"Téléchargements échoués: {failed_downloads}")
    
    # Durée totale téléchargée
    total_duration = sum(v.get('duration', 0) for v in videos_unique if v.get('status') == 'success')
    hours = int(total_duration // 60)
    minutes = int(total_duration % 60)
    print(f"Durée totale du dataset: {hours}h {minutes}m ({total_duration:.1f} minutes)")
    
    # Vitesse moyenne
    if successful_downloads > 0:
        avg_time_per_video = elapsed_time / successful_downloads
        print(f"Vitesse moyenne: {avg_time_per_video:.2f} secondes par vidéo")
    
    print(f"Temps total d'exécution: {elapsed_str}")
    print("="*60)
    print("\n🏁 TÉLÉCHARGEMENT TERMINÉ\n")

if __name__ == '__main__':
    main()