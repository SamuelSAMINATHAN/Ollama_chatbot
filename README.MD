# Ollama Chatbot + PDF Analysis Project


A project to familiarize myself with local LLMs.
This project is an application that allows interaction with an large langage model using Ollama for general chat and PDF document analysis by generating responses or a summary from the content of a PDF.
Tkinter is used for the graphical interface. 

## Features

- **Chat général** : Uses an LLMChain to ask questions in natural language.
- **Analyse PDF** : Extracts text from a PDF and creates a QA (question/answer) chain or a summary using RetrievalQA.

## Prerequisites

- Python 3.10+
- Install [Ollama] (https://ollama.com/download/windows) 
- Install LLM models with the terminal (https://ollama.com/search) Ex:(ollama run deepseek-r1:8b)
- The application uses "nomic-embed-text" by default for embedding (ollama pull nomic-embed-text) 
    (If you want to use another embedding model, change line 50 of the code)

## Installation

1. Clone the repository:
   git clone https://github.com/SamuelSAMINATHAN/Ollama_chatbot.git
   cd Ollama_chatbot

2. Create and activate a virtual environment:
    python3 -m venv venv
    source venv/bin/activate      # on Linux/macOS
    python -m venv venv
    venv\Scripts\activate         # on Windows

3. Install dependencies:
    pip install -r requirements.txt

4. Run the application:
    python3 main.py           # on Linux/macOS
    python3 main.py           # on Windows


