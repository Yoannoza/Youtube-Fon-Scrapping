
# 🎧 YouTube Fon Audio Scraper – Hackathon Isheero

Ce script a été développé dans le cadre du **Academics Pionners d'Isheero**, l'objectif de ce premier challenge est pour scraper **1000 heures de vidéos YouTube** (principalement en *fon*) et en extraire l’audio automatiquement.

---

## 🧾 Description

Le fichier principal est **`scrapp_v4.py`**. Il utilise l’**API YouTube** pour récupérer des vidéos à partir de playlists ou de recherches, puis automatise le téléchargement de l’audio via **Playwright** (site `yt.savetube.me`).

L'objectif est de créer une **base d’audios** en langue fon, utilisable pour des projets d’IA ou linguistiques.

---

## 🛠️ Pré-requis

- Python 3.9+
- [Playwright](https://playwright.dev/) (automatisation navigateur)
- Une clé API YouTube valide (YouTube Data API v3)
- Fichier `.env` avec ta clé :

```

GOOGLE\_API=ta\_clé\_api\_youtube

````

- Installer les dépendances :

```bash
pip install -r requirements.txt
playwright install
````

---

## 🧠 Utilisation

1. Ouvre le fichier `scrapp_v4.py`
2. Décommente **la playlist que tu veux scraper**
3. Modifie le nom du fichier de référence (par exemple : `videos_downloaders_fon_part1.csv`) pour chaque nouvelle session.
   👉 Ce fichier garde en mémoire ce qui a déjà été téléchargé.
4. Exécute le script :

```bash
python scrapp_v4.py
```

5. À chaque itération :

   * Change de playlist
   * Change de fichier `videos_downloaders_x.csv`
   * À la fin, tu pourras **fusionner tous les fichiers CSV**

---

## 📦 Résultat

* Audios enregistrés au format **MP3 (320 kbps)**
* Un fichier CSV contenant les métadonnées + lien vers l'audio :

  * Titre
  * URL vidéo
  * Durée
  * Langue
  * Lien MP3 direct

---

## 📂 Base d'audios

> ⚠️ La base d'audios complète n'est **pas dans ce repo**.
> Un accès sera fourni ultérieurement.

---

## ✅ Bonnes pratiques

* Ne lance pas plusieurs téléchargements en parallèle sur la même machine
* Garde un œil sur le fichier log (`youtube_download.log`)
* Utilise un VPN ou des proxies si tu rencontres des blocages

---

## 🤝 Contribution

* Tu veux ajouter une playlist ? Tu peux l’ajouter dans `scrapp_v4.py`, en commentaire.
* Tu peux aussi améliorer le script (robustesse, nouvelles méthodes de scrap, etc.)
