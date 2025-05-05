from datasets import load_from_disk, Audio
import os
import pandas as pd
import json
import logging
import argparse
from huggingface_hub import HfApi, create_repo, login
from tqdm import tqdm

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("huggingface_upload.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

def upload_to_huggingface(dataset_path, hf_token, repo_name, description):
    """
    Télécharger un dataset préparé vers HuggingFace
    
    Args:
        dataset_path: Chemin vers le dataset local (préparé avec dataset.save_to_disk())
        hf_token: Token d'accès HuggingFace
        repo_name: Nom du dépôt (format: "username/repo-name")
        description: Description du dataset
    """
    try:
        # Authentification avec HuggingFace
        login(token=hf_token)
        api = HfApi()
        
        # Créer le dépôt si il n'existe pas
        try:
            create_repo(repo_name, repo_type="dataset", exist_ok=True)
            logger.info(f"Dépôt créé ou déjà existant: {repo_name}")
        except Exception as e:
            logger.error(f"Erreur lors de la création du dépôt: {str(e)}")
            return False
        
        # Charger le dataset local
        dataset = load_from_disk(dataset_path)
        
        # Préparer les métadonnées
        metadata = {
            "language": "fon",
            "license": "cc-by-4.0",
            "dataset_info": {
                "description": description,
                "features": {
                    "audio": {
                        "sampling_rate": 16000
                    }
                },
                "homepage": f"https://huggingface.co/datasets/{repo_name}",
                "citation": "",
            }
        }
        
        # Enregistrer les métadonnées
        with open("dataset-metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Télécharger le fichier de métadonnées
        api.upload_file(
            path_or_fileobj="dataset-metadata.json",
            path_in_repo="dataset-metadata.json",
            repo_id=repo_name,
            repo_type="dataset"
        )
        
        # Télécharger les fichiers du dataset
        print("Téléchargement du dataset vers HuggingFace...")
        dataset.push_to_hub(
            repo_name,
            private=False,
            token=hf_token
        )
        
        logger.info(f"Dataset téléchargé avec succès: {repo_name}")
        print(f"✅ Dataset téléchargé avec succès: https://huggingface.co/datasets/{repo_name}")
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement vers HuggingFace: {str(e)}")
        print(f"❌ Erreur: {str(e)}")
        return False

def create_dataset_card(repo_name, original_hours, processed_hours, file_path="README.md"):
    """
    Créer une carte descriptive pour le dataset
    """
    try:
        # Charger les statistiques si disponibles
        stats = {}
        if os.path.exists("processing_statistics.json"):
            with open("processing_statistics.json", "r") as f:
                stats = json.load(f)
        
        # Template Markdown pour la carte du dataset
        card_content = f"""
            # Dataset Audio en langue Fon pour l'apprentissage automatique

            ## Description

            Ce dataset contient des enregistrements audio en langue Fon, principalement collectés à partir de YouTube. Les données ont été nettoyées, segmentées et préparées pour l'entraînement de modèles de reconnaissance vocale (STT) et de synthèse vocale (TTS).

            ## Statistiques du dataset

            - **Collection originale**: Environ {original_hours} heures d'audio
            - **Après traitement**: Environ {processed_hours} heures d'audio de haute qualité
            - **Langue**: Fon (parlée principalement au Bénin)
            - **Format audio**: WAV mono, 16kHz, 16-bit PCM
            """

                    # Ajouter des statistiques détaillées si disponibles
                    if stats:
                        card_content += f"""
            - **Nombre de segments**: {stats.get('kept_segments', 'N/A')}
            - **Durée moyenne des segments**: {stats.get('average_segment_duration', 'N/A'):.2f} secondes
            - **SNR moyen**: {stats.get('average_segment_snr', 'N/A'):.2f} dB
            """

                    card_content += """
            ## Prétraitement

            Les données audio ont subi plusieurs étapes de prétraitement:
            1. **Conversion et normalisation**: Conversion en WAV mono 16kHz et normalisation d'amplitude
            2. **Détection de la parole**: Extraction des segments contenant uniquement de la parole
            3. **Filtrage qualité**: Élimination des segments de mauvaise qualité (faible SNR, silence excessif)
            4. **Segmentation**: Découpage en segments de durée optimale pour l'apprentissage
            5. **Augmentation de données**: Application de techniques d'augmentation pour diversifier le dataset

            ## Utilisation

            Ce dataset est conçu pour entraîner:
            - Des modèles de reconnaissance vocale (Speech-to-Text) en Fon
            - Des modèles de synthèse vocale (Text-to-Speech) en Fon
            - Des systèmes de traduction vocale (Speech-to-Speech)

            ## Citation

            Si vous utilisez ce dataset dans vos travaux, veuillez le citer comme suit:

            ```
            @dataset{fon_audio_dataset,
            author    = {},
            title     = {Dataset Audio en langue Fon},
            year      = {2025},
            publisher = {HuggingFace},
            url       = {https://huggingface.co/datasets/%s}
            }
            ```

            ## Licence

            Ce dataset est partagé sous licence [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
            """ % repo_name

        # Écrire le contenu dans un fichier README.md
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(card_content)
        
        logger.info(f"Carte du dataset créée: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors de la création de la carte du dataset: {str(e)}")
        return False

def main():
    # Parser pour les arguments de ligne de commande
    parser = argparse.ArgumentParser(description="Télécharger un dataset audio vers HuggingFace")
    parser.add_argument("--dataset_path", type=str, default="huggingface_fon_dataset", 
                        help="Chemin vers le dataset local")
    parser.add_argument("--token", type=str, required=True,
                        help="Token d'accès HuggingFace")
    parser.add_argument("--repo_name", type=str, required=True,
                        help="Nom du dépôt sur HuggingFace (format: username/repo-name)")
    parser.add_argument("--original_hours", type=float, default=1016.0,
                        help="Nombre d'heures d'audio originales")
    parser.add_argument("--processed_hours", type=float, required=True,
                        help="Nombre d'heures d'audio après traitement")
    parser.add_argument("--description", type=str, default="Dataset audio en langue Fon pour l'entraînement de modèles STT et TTS",
                        help="Description courte du dataset")
    
    args = parser.parse_args()
    
    print("🚀 PRÉPARATION DU TÉLÉCHARGEMENT VERS HUGGINGFACE")
    
    # Créer la carte du dataset
    print("Création de la carte descriptive du dataset...")
    create_dataset_card(args.repo_name, args.original_hours, args.processed_hours)
    
    # Télécharger vers HuggingFace
    print(f"Téléchargement vers le dépôt: {args.repo_name}")
    upload_to_huggingface(args.dataset_path, args.token, args.repo_name, args.description)
    
    print("🏁 PROCESSUS DE TÉLÉCHARGEMENT TERMINÉ")

if __name__ == "__main__":
    main()
    