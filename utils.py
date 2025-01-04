# utils.py 
import os
import logging
import xml.etree.ElementTree as ET
import re
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# Configuration des logs
logger = logging.getLogger(__name__)

VOTE_MAPPING = {
    'pour': 1,
    'contre': -1,
    'abstention': 0,
    'non_votant_volontaire': np.nan,
    'absent': np.nan
}

def sanitize_law_title(title: str) -> str:
    """
    Sanitize law titles to be used as HTML element IDs by replacing non-alphanumeric characters with underscores.

    Args:
        title (str): The original law title.

    Returns:
        str: Sanitized law title.
    """
    if not isinstance(title, str):
        raise ValueError("Le titre doit être une chaîne de caractères.")
    return re.sub(r'\W+', '_', title).strip()

def parse_deputes(dossier_xml_deputes: str, namespace: dict) -> dict:
    deputes_dict = {}
    try:
        xml_files = [f for f in os.listdir(dossier_xml_deputes) if f.endswith('.xml')]
        logger.info(f"Nombre de fichiers XML (députés) dans {dossier_xml_deputes} : {len(xml_files)}")

        for fichier in xml_files:
            chemin_fichier = os.path.join(dossier_xml_deputes, fichier)
            try:
                tree = ET.parse(chemin_fichier)
                root = tree.getroot()

                uid = root.find('.//ns:uid', namespace)
                nom = root.find('.//ns:nom', namespace)
                prenom = root.find('.//ns:prenom', namespace)

                if uid is not None and nom is not None and prenom is not None:
                    name = f"{prenom.text} {nom.text}"
                    deputes_dict[uid.text] = {
                        "name": name,
                        "uid": uid.text
                    }
                else:
                    logger.warning(f"Informations manquantes dans {chemin_fichier}")

            except ET.ParseError as e:
                logger.error(f"Erreur de parsing pour le fichier: {chemin_fichier} -> {e}")
            except Exception as e:
                logger.error(f"Erreur lors de la lecture de {chemin_fichier}: {e}")

    except FileNotFoundError:
        logger.error(f"Le dossier des députés n'existe pas : {dossier_xml_deputes}")
    except Exception as e:
        logger.error(f"Erreur lors de l'accès au dossier des députés : {e}")

    return deputes_dict

def parse_votes(folder_path_votes: str, namespace: dict) -> pd.DataFrame:
    """
    Parse les fichiers XML des votes et retourne un DataFrame des votes.

    Args:
        folder_path_votes (str): Chemin vers le dossier contenant les fichiers XML des votes.
        namespace (dict): Namespace XML utilisé pour parser les fichiers.

    Returns:
        pd.DataFrame: DataFrame des votes avec les députés en index et les lois en colonnes.
    """
    all_votes = {}
    try:
        xml_files = [f for f in os.listdir(folder_path_votes) if f.endswith('.xml')]
        logger.info(f"Nombre de fichiers XML (votes) dans {folder_path_votes} : {len(xml_files)}")

        for filename in xml_files:
            file_path = os.path.join(folder_path_votes, filename)
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()

                title_element = root.find('.//ns:titre', namespace)
                if title_element is None or not title_element.text:
                    logger.warning(f"Pas de titre trouvé dans {file_path}")
                    continue

                title = title_element.text.strip()

                decompte_nominatif = root.findall('.//ns:decompteNominatif', namespace)
                if not decompte_nominatif:
                    logger.warning(f"Pas de decompteNominatif dans {file_path}")
                    continue

                for section in decompte_nominatif:
                    for tag, value in [
                        ('ns:pours/ns:votant', 'pour'),
                        ('ns:contres/ns:votant', 'contre'),
                        ('ns:abstentions/ns:votant', 'abstention'),
                        ('ns:nonVotantsVolontaires/ns:votant', 'non_votant_volontaire')
                    ]:
                        for votant in section.findall(f'.//{tag}', namespace):
                            acteur_ref = votant.find('ns:acteurRef', namespace)
                            if acteur_ref is not None and acteur_ref.text:
                                all_votes.setdefault(acteur_ref.text, {})[title] = value

            except ET.ParseError as e:
                logger.error(f"Erreur de parsing pour le fichier: {file_path} -> {e}")
            except Exception as e:
                logger.error(f"Erreur lors de la lecture de {file_path}: {e}")

    except FileNotFoundError:
        logger.error(f"Le dossier des votes n'existe pas : {folder_path_votes}")
    except Exception as e:
        logger.error(f"Erreur lors de l'accès au dossier des votes : {e}")

    votes_df = pd.DataFrame.from_dict(all_votes, orient='index').fillna('absent')
    logger.info("DataFrame des votes généré avec succès.")
    return votes_df

# Remplacer cette liste de lois pour avoir une plus grande diversité de sujets
LAW_TITLES = [
    "Loi sur la protection de l'environnement",
    "Loi sur la réforme du système de santé",
    "Loi sur l'éducation et la formation professionnelle",
    "Loi sur la sécurité routière",
    "Loi sur la liberté d'expression",
    "Loi sur la fiscalité des entreprises",
    "Loi sur les droits des travailleurs",
    "Loi sur la sécurité intérieure",
    "Loi sur la protection des animaux",
    "Loi sur la lutte contre les inégalités sociales"
]

def cluster_deputes(votes_df_numeric_filtered: pd.DataFrame,
                    deputes_dict: dict,
                    cluster_threshold: int,
                    output_filename: str) -> dict:
    try:
        Z = linkage(votes_df_numeric_filtered.fillna(0), method='ward')
        clusters = fcluster(Z, cluster_threshold, criterion='maxclust')

        grouped_deputes = {}
        for idx, cluster_id in enumerate(clusters):
            uid = votes_df_numeric_filtered.index[idx]
            deputy_info = deputes_dict.get(uid, {})
            deputy_name = deputy_info.get("name", uid)
            grouped_deputes.setdefault(f"Groupe {cluster_id}", []).append(deputy_name)

        dendrogram_dir = os.path.join(os.getcwd(), 'static', 'dendrogrammes')
        os.makedirs(dendrogram_dir, exist_ok=True)

        plt.figure(figsize=(12, 8))
        dendrogram(
            Z,
            labels=[deputes_dict.get(uid, {}).get("name", uid) for uid in votes_df_numeric_filtered.index],
            leaf_rotation=90,
            leaf_font_size=10,
            color_threshold=cluster_threshold
        )
        plt.title("Dendrogramme des députés")
        plt.tight_layout()
        dendrogram_path = os.path.join(dendrogram_dir, output_filename)
        plt.savefig(dendrogram_path)
        plt.close()

        logger.info(f"Clustering effectué et dendrogramme sauvegardé ({dendrogram_path}).")
        return grouped_deputes

    except Exception as e:
        logger.error(f"Erreur lors du clustering des députés : {e}", exc_info=True)
        return {}

def compute_top_3_similar_deputes(user_votes: dict,
                                  votes_df: pd.DataFrame,
                                  deputes_dict: dict) -> list:
    """
    Calcule les 3 députés les plus similaires aux votes de l'utilisateur.

    Args:
        user_votes (dict): Dictionnaire des votes de l'utilisateur avec les lois comme clés.
        votes_df (pd.DataFrame): DataFrame des votes des députés.
        deputes_dict (dict): Dictionnaire des députés avec leurs informations.

    Returns:
        list: Liste des 3 députés les plus similaires avec leur nom, similarité et UID.
    """
    if not user_votes:
        logger.warning("Aucune réponse fournie par l'utilisateur.")
        return []

    laws_answered = list(user_votes.keys())
    user_vector = [
        VOTE_MAPPING.get(choice, 0) for choice in user_votes.values()
    ]
    user_vector = np.array(user_vector).reshape(1, -1)

    try:
        sub_df = votes_df[laws_answered].replace(VOTE_MAPPING)
    except KeyError as e:
        logger.error(f"Loi non trouvée dans les données de votes : {e}")
        return []

    sub_df = sub_df.fillna(0)

    if sub_df.empty:
        logger.warning("DataFrame filtré est vide après remplacement des valeurs.")
        return []

    similarity_scores = cosine_similarity(user_vector, sub_df.values)[0]

    uids = list(sub_df.index)
    deputy_scores = [
        {
            "name": deputes_dict.get(uid, {}).get("name", uid),
            "similarity": score,
            "uid": deputes_dict.get(uid, {}).get("uid", uid)
        }
        for uid, score in zip(uids, similarity_scores)
    ]

    deputy_scores_sorted = sorted(deputy_scores, key=lambda x: x['similarity'], reverse=True)

    logger.info("Top 3 députés similaires calculés avec succès.")
    return deputy_scores_sorted[:3]
