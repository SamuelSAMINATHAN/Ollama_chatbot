# Projet Ollama Chatbot + Analyse PDF

Ce projet est une application graphique (Tkinter) qui permet d'interagir avec un modèle de langage Ollama pour un chat général et d'analyser des documents PDF en générant des réponses ou un résumé à partir du contenu du PDF.

## Fonctionnalités

- **Chat général** : Utilisation d'une LLMChain pour poser des questions en langage naturel.
- **Analyse PDF** : Extraction du texte d'un PDF et création d'une chaîne de QA (question / réponse) ou d'un résumé via RetrievalQA.

## Prérequis

- Python 3.10+
- installez [Ollama] (https://ollama.com/download/windows) 
- Installez des modèle LLM via le terminal (https://ollama.com/search) Ex:(ollama run deepseek-r1:8b)
- L'application utilise par défault "nomic-embed-text" pour l'embedding (ollama pull nomic-embed-text) 
    (Si vous voulez utiliser un autre modèle d'embendding, changez la ligne 50 du code)

## Installation

1. Clonez le dépôt :
   git clone https://github.com/votre_utilisateur/mon_projet.git
   cd mon_projet

2. Créez un environnement virtuel et activez-le :
    python3 -m venv venv
    source venv/bin/activate      # sous Linux/macOS
    python -m venv venv
    venv\Scripts\activate         # sous Windows

3. Installez les dépendances :
    pip install -r requirements.txt

4. Lancer l'application :
    python3 main.py           # sous Linux/macOS
    python3 main.py           # sous Windows

# Ollama_chatbot
