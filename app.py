import os
import logging
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Pour le serveur headless
import matplotlib.pyplot as plt
from utils import (
    parse_deputes,
    parse_votes,
    cluster_deputes,
    compute_top_3_similar_deputes,
    sanitize_law_title
)

# Configuration
DOSSIER_XML_DEPUTES = os.path.join("data", "acteur")
FOLDER_PATH_VOTES = os.path.join("data", "xml")

NAMESPACE = {"ns": "http://schemas.assemblee-nationale.fr/referentiel"}

VOTE_MAPPING = {
    'pour': 1,
    'contre': -1,
    'abstention': 0,
    'non_votant_volontaire': np.nan,
    'absent': np.nan
}

CLUSTER_THRESHOLD = 30  # Ajusté pour augmenter le nombre de groupes
PORT = 8000

# Initialisation de l'application Flask
app = Flask(__name__)

# Configuration du logging pour n'afficher que ERROR et CRITICAL
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

###############################################################################
#                       INITIALISATION DES DONNÉES
###############################################################################

# Variables globales
DEPUTES_DICT = {}
VOTES_DF = pd.DataFrame()
GROUPED_DEPUTES_FILTERED = {}

def load_data():
    global DEPUTES_DICT, VOTES_DF, GROUPED_DEPUTES_FILTERED

    logging.error("Initialisation : chargement des données...")
    DEPUTES_DICT = parse_deputes(DOSSIER_XML_DEPUTES, NAMESPACE)
    VOTES_DF = parse_votes(FOLDER_PATH_VOTES, NAMESPACE)

    if VOTES_DF.empty:
        logging.error("Le DataFrame des votes est vide. Vérifie les fichiers XML de votes.")
        GROUPED_DEPUTES_FILTERED = {}
        return

    # Remplacer les valeurs catégorielles par numériques
    votes_df_numeric = VOTES_DF.replace(VOTE_MAPPING)

    # Clustering initial
    grouped_deputes_initial = cluster_deputes(
        votes_df_numeric,
        DEPUTES_DICT,
        CLUSTER_THRESHOLD,
        "dendrogramme_initial.png"
    )

    # Filtrer les groupes avec >= 20 députés
    GROUPED_DEPUTES_FILTERED = {
        group: deputies for group, deputies in grouped_deputes_initial.items() if len(deputies) >= 20
    }
    logging.error(f"Nombre de groupes après filtrage (≥ 20 membres) : {len(GROUPED_DEPUTES_FILTERED)}")

    if not GROUPED_DEPUTES_FILTERED:
        logging.error("Aucun groupe n'a 20 membres ou plus. Aucune filtration appliquée.")
        GROUPED_DEPUTES_FILTERED = grouped_deputes_initial

    logging.error("Données chargées avec succès.")

# Charger les données au démarrage
load_data()

###############################################################################
#                       ROUTES FLASK
###############################################################################

@app.route('/')
def index():
    return render_template(
        'index.html',
        grouped_deputes=GROUPED_DEPUTES_FILTERED,
        votes_df=VOTES_DF
    )

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/submit', methods=['POST'])
def submit_votes():
    try:
        user_votes_sanitized = request.get_json()

        # Récupérer les titres originaux des lois
        sanitized_to_original = {
            sanitize_law_title(col): col for col in VOTES_DF.columns[:10]
        }
        original_user_votes = {
            sanitized_to_original.get(k, k): v for k, v in user_votes_sanitized.items()
        }

        # Calcul des 3 députés les plus similaires
        top_3 = compute_top_3_similar_deputes(original_user_votes, VOTES_DF, DEPUTES_DICT)

        results = []
        for deputy in top_3:
            results.append({
                "name": deputy['name'],
                "similarity": float(deputy['similarity']),
                "uid": deputy['uid']
            })

        return jsonify(results)

    except Exception as e:
        logging.error(f"Erreur lors du traitement de /submit : {e}")
        return jsonify({"error": "Une erreur est survenue lors du traitement des votes."}), 400

# Route pour servir les dendrogrammes
@app.route('/dendrogrammes/<filename>')
def serve_dendrogramme(filename):
    return send_from_directory(
        os.path.join(app.root_path, 'static', 'dendrogrammes'),
        filename
    )

###############################################################################
#                       UTILITIES POUR LES TEMPLATES
###############################################################################

@app.context_processor
def utility_processor():
    return dict(sanitize_law_title=sanitize_law_title)

###############################################################################
#                       EXÉCUTION DE L'APPLICATION
###############################################################################

if __name__ == "__main__":
    app.run(port=PORT, debug=False)  # Désactiver le mode debug
