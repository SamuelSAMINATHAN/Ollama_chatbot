import os
import subprocess
import tkinter as tk
import pytest



from main import (
    get_ollama_models,
    PDFChatApplication
)

# --- Tests pour get_ollama_models ---

def test_get_ollama_models_success(monkeypatch):
    """Simule une exécution réussie de la commande ollama et vérifie l'extraction des modèles."""
    class DummyCompletedProcess:
        def __init__(self):
            self.returncode = 0
            self.stdout = "model1\nmodel2\n"
    def dummy_run(args, capture_output, text):
        return DummyCompletedProcess()
    monkeypatch.setattr(subprocess, "run", dummy_run)
    models = get_ollama_models()
    assert models == ["model1", "model2"]

def test_get_ollama_models_file_not_found(monkeypatch, capsys):
    """Vérifie le comportement en cas d'absence de la commande ollama."""
    def dummy_run(*args, **kwargs):
        raise FileNotFoundError()
    monkeypatch.setattr(subprocess, "run", dummy_run)
    models = get_ollama_models()
    captured = capsys.readouterr().out
    assert "Erreur : la commande 'ollama' n'est pas trouvée." in captured
    assert models == []


# --- Fixture pour créer une instance de l'application ---
@pytest.fixture
def app_instance(monkeypatch):
    """
    Crée une instance de PDFChatApplication en masquant la fenêtre Tkinter.
    Redéfinit la méthode _append_chat_message pour capturer les messages affichés.
    """
    root = tk.Tk()
    root.withdraw()  # Masquer la fenêtre principale pendant les tests

    messages = {"chat": [], "pdf": []}

    def fake_append(self, text, area="chat"):
        messages[area].append(text)
    monkeypatch.setattr(PDFChatApplication, "_append_chat_message", fake_append)
    app = PDFChatApplication(root)
    return app, messages, root


# --- Tests pour l'onglet Chat ---

def test_update_chat_model_no_model(app_instance):
    """Test de update_chat_model quand aucun modèle n'est trouvé."""
    app, messages, root = app_instance
    app.chat_model_var.set("Aucun modèle trouvé")
    app.update_chat_model()
    assert "Aucun modèle Ollama disponible." in messages["chat"]

def test_send_message_no_chain(app_instance):
    """Vérifie que send_message affiche une erreur si aucune chaîne de chat n'est configurée."""
    app, messages, root = app_instance
    app.chat_chain = None
    app.user_entry.insert("1.0", "Hello")
    app.send_message()
    assert "Aucun modèle de chat n'est sélectionné." in messages["chat"]

def test_send_message_with_chain(app_instance):
    """Simule l'envoi d'un message avec une chaîne LLM fictive."""
    app, messages, root = app_instance

    # Création d'une chaîne fictive qui renvoie une réponse prédéfinie
    class DummyChain:
        def run(self, inputs):
            return "Dummy response"
    app.chat_chain = DummyChain()

    test_message = "Test question"
    app.user_entry.delete("1.0", tk.END)
    app.user_entry.insert("1.0", test_message)
    app.send_message()
    chat_msgs = messages["chat"]
    # Vérification que le message utilisateur est affiché
    assert any("**User**:" in msg and test_message in msg for msg in chat_msgs)
    # Vérification que la réponse du bot est affichée
    assert any("**Bot**:" in msg and "Dummy response" in msg for msg in chat_msgs)


# --- Tests pour l'onglet PDF ---

def test_update_pdf_model(app_instance):
    """Vérifie que l'appel à update_pdf_model affiche le modèle sélectionné."""
    app, messages, root = app_instance
    test_model = "TestPDFModel"
    app.pdf_model_var.set(test_model)
    app.update_pdf_model()
    assert f"[INFO] Modèle PDF sélectionné : {test_model}" in messages["pdf"]

def test_ask_pdf_question_no_chain(app_instance):
    """Test de ask_pdf_question quand aucune QA chain n'est configurée."""
    app, messages, root = app_instance
    app.qa_chain = None
    app.pdf_question_entry.insert("1.0", "Question PDF?")
    app.ask_pdf_question()
    assert "Veuillez charger un PDF et sélectionner un modèle PDF." in messages["pdf"]

def test_ask_pdf_question_with_chain(app_instance):
    """Simule l'appel à ask_pdf_question avec une QA chain fictive."""
    app, messages, root = app_instance

    class DummyQAChain:
        def run(self, question):
            return "Dummy PDF answer"
    app.qa_chain = DummyQAChain()
    test_question = "What is PDF about?"
    app.pdf_question_entry.delete("1.0", tk.END)
    app.pdf_question_entry.insert("1.0", test_question)
    app.ask_pdf_question()
    pdf_msgs = messages["pdf"]
    assert any("**Question**:" in msg and test_question in msg for msg in pdf_msgs)
    assert any("**Réponse**:" in msg and "Dummy PDF answer" in msg for msg in pdf_msgs)

def test_summarize_pdf_not_ready(app_instance):
    """Vérifie que summarize_pdf affiche une erreur si le PDF ou le modèle n'est pas prêt."""
    app, messages, root = app_instance
    app.vectorstore = None
    app.qa_chain = None
    app.summarize_pdf()
    assert "PDF non prêt ou modèle PDF non sélectionné." in messages["pdf"]

def test_summarize_pdf_with_dummy(monkeypatch, app_instance):
    """
    Teste summarize_pdf en simulant un vectorstore existant et en remplaçant la création de la QA chain
    par une chaîne fictive qui renvoie un résumé prédéfini.
    """
    app, messages, root = app_instance

    # Création d'un vectorstore fictif avec la méthode as_retriever
    class DummyRetriever:
        pass
    class DummyVectorstore:
        def as_retriever(self, search_kwargs):
            return DummyRetriever()
    app.vectorstore = DummyVectorstore()

    # Création d'une QA chain fictive pour le résumé
    class DummySumChain:
        def run(self, question):
            return "Dummy summary"

    # Remplacement de RetrievalQA.from_chain_type par une fonction lambda retournant DummySumChain
    from langchain.chains import RetrievalQA
    monkeypatch.setattr(RetrievalQA, "from_chain_type", lambda **kwargs: DummySumChain())

    # S'assurer qu'un modèle PDF valide est sélectionné
    app.pdf_model_var.set("DummyModel")
    app.summarize_pdf()
    pdf_msgs = messages["pdf"]
    assert any("**Résumé demandé**:" in msg for msg in pdf_msgs)
    assert any("**Résumé**:" in msg and "Dummy summary" in msg for msg in pdf_msgs)

def test_load_pdf(monkeypatch, app_instance):
    """
    Teste load_pdf en simulant :
      - La sélection d'un fichier PDF via filedialog
      - L'extraction de texte d'un PDF via _extract_text_by_page
      - La création d'un vectorstore via Chroma.from_documents
    """
    app, messages, root = app_instance
    fake_pdf_path = "dummy.pdf"
    monkeypatch.setattr(tk.filedialog, "askopenfilename", lambda **kwargs: fake_pdf_path)

    # Remplacer _extract_text_by_page pour retourner une liste avec une page fictive
    def fake_extract_text_by_page(self, pdf_path):
        from langchain.docstore.document import Document
        return [Document(page_content="Dummy page text", metadata={"source": pdf_path, "page": 1})]
    monkeypatch.setattr(PDFChatApplication, "_extract_text_by_page", fake_extract_text_by_page)

    # Création d'un vectorstore fictif
    class DummyVectorstore:
        def persist(self):
            pass
        def as_retriever(self, search_kwargs):
            class DummyRetriever:
                pass
            return DummyRetriever()
    from langchain.vectorstores import Chroma
    monkeypatch.setattr(Chroma, "from_documents", lambda documents, embedding, collection_name, persist_directory: DummyVectorstore())

    # S'assurer qu'un modèle PDF valide est sélectionné pour que _create_pdf_qa_chain fonctionne
    app.pdf_model_var.set("DummyModel")
    app.load_pdf()
    pdf_msgs = messages["pdf"]
    assert any("Chargement du PDF" in msg and fake_pdf_path in msg for msg in pdf_msgs)
    assert any("PDF ingéré" in msg for msg in pdf_msgs)
    assert any("QA chain prête avec le modèle" in msg for msg in pdf_msgs)

