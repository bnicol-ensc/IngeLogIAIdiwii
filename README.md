# Projet d'ingénierie Logicielle pour l'IA
### Mené à contribution égales par G. Grosse, M. Mahaut, B. Nicol

## Problématique
Mettre à disposition sur DockerHub un service de classification d'intention (NLP) dans une image docker.

## Contexte
Projet : Ingénierie Logicielle pour le Machine learning
Spécialité de filière : Intelligence Artificielle
Ecoles d'ingénieurs associées : ENSC et ENSEIRB
Durée : 2 mois

## Premiers pas
Utiliser pip pour installer les requirements : `pip install -r requirements.txt`
Démarrer un serveur jupyter : `jupyter-notebook` ou `jupyer-lab`
Pour suivre l'évolution du projet, lire les différents notebooks et les analyses qu'ils comprennent
Pour tester le service, voir `api`.

## Contenu du projet
### api
Contient le serveur flask, le modèle et le Dockerfile.  
Le conteneur docker a été mis en ligne sur DockerHub (https://hub.docker.com/r/bnicolensc/intent_detection) et est récupérable au moyen de la commande `docker pull bnicolensc/intent_detection`
Pour lancer le conteneur : `docker run -d -p 5000:5000 bnicolensc/intent_detection`
Ensuite aller sur l'URL : http://0.0.0.0:5000/api/*Phrase_à_classifier*, en remplaçant *Phrase_à_classifier* par une phrase au choix.

### data
Contient les données traitées et non traitées utilisées pour l'entraînement et les tests de modèles.

### notebooks
Contient les programmes d'entraînement, de test et de comparaison des modèles (original VS généré).

### source
Fonctions outils nécessaires à l'exécution des notebooks.

### test
Tests unitaires des fonctions des fichiers `source`.
