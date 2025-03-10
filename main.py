import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import subprocess
import fitz  # PyMuPDF
import os

from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.docstore.document import Document

# Ollama (LLM & Embeddings)
from langchain_ollama import OllamaLLM, OllamaEmbeddings

# Import supplémentaire pour construire la chaîne de chat
from langchain.chains import LLMChain

# -------------------------------
# Fonction pour lister les modèles Ollama
# -------------------------------
def get_ollama_models():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            models = []
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split()
                model_name = parts[0]
                models.append(model_name)
            return models
        else:
            return []
    except FileNotFoundError:
        print("Erreur : la commande 'ollama' n'est pas trouvée.")
        return []

class PDFChatApplication:
    def __init__(self, master):
        self.master = master
        self.master.title("Ollama Chatbot + Analyse PDF")

        # Liste des modèles Ollama
        self.models = get_ollama_models()

        # Embeddings
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text") 

        self.chroma_persist_dir = "chroma_db"
        if not os.path.exists(self.chroma_persist_dir):
            os.mkdir(self.chroma_persist_dir)

        self.vectorstore = None
        self.chat_chain = None  # Sera une LLMChain pour le chat général
        self.qa_chain = None    # Sera un RetrievalQA pour le PDF

        self.chat_model_var = tk.StringVar()
        self.pdf_model_var = tk.StringVar()
        
        if self.models:
            self.chat_model_var.set(self.models[0])
            self.pdf_model_var.set(self.models[0])
        else:
            self.chat_model_var.set("Aucun modèle trouvé")
            self.pdf_model_var.set("Aucun modèle trouvé")

        # ---- On construit l'interface
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Onglet Chat
        self.chat_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.chat_frame, text="Chat Général")
        self._build_chat_ui(self.chat_frame)

        # Onglet PDF
        self.pdf_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.pdf_frame, text="Analyse PDF")
        self._build_pdf_ui(self.pdf_frame)

    def _build_chat_ui(self, parent):
        # Zone de conversation
        self.conversation_area = scrolledtext.ScrolledText(parent, wrap=tk.WORD)
        self.conversation_area.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)
        self.conversation_area.config(state=tk.DISABLED)

        # Zone de saisie utilisateur
        self.user_entry = tk.Text(parent, width=70, height=3)
        self.user_entry.pack(pady=5)

        # Frame pour la sélection du modèle + bouton
        model_frame = tk.Frame(parent)
        model_frame.pack(pady=5)

        chat_model_combo = ttk.Combobox(
            model_frame, 
            textvariable=self.chat_model_var,
            values=self.models,
            state="readonly"
        )
        chat_model_combo.pack(side=tk.LEFT)

        use_model_button = tk.Button(
            model_frame, 
            text="Utiliser ce modèle", 
            command=self.update_chat_model
        )
        use_model_button.pack(side=tk.LEFT, padx=5)

        # Bouton d'envoi de message
        send_button = tk.Button(parent, text="Envoyer", command=self.send_message)
        send_button.pack(pady=5)

    def update_chat_model(self):
        """
        Met à jour le LLM pour le chat général, en créant une LLMChain avec un PromptTemplate.
        """
        selected_model = self.chat_model_var.get()
        if selected_model == "Aucun modèle trouvé":
            self._append_chat_message("Aucun modèle Ollama disponible.", area="chat")
            return

        # PromptTemplate pour gérer le "context" et la "question"
        chat_prompt_template = """{context}
Question utilisateur : {question}
Réponds en français en tenant compte du contexte ci-dessus :
"""

        prompt = PromptTemplate(
            template=chat_prompt_template,
            input_variables=["context", "question"]
        )

        # Construction de la LLMChain
        self.chat_chain = LLMChain(
            llm=OllamaLLM(model=selected_model),
            prompt=prompt
        )

        self._append_chat_message(f"[INFO] Modèle de chat sélectionné : {selected_model}", area="chat")

    def send_message(self):
        """
        Gestion du message utilisateur pour le chat général.
        """
        if not self.chat_chain:
            self._append_chat_message("Aucun modèle de chat n'est sélectionné.", area="chat")
            return

        user_input = self.user_entry.get("1.0", tk.END).strip()
        if not user_input:
            return

        # On affiche le message de l'utilisateur
        self._append_chat_message(f"**User**: {user_input}", area="chat")

        # Préparation des variables à passer au prompt
        final_inputs = {
            "context": "",
            "question": user_input
        }

        # Envoi au LLM via la LLMChain
        try:
            result = self.chat_chain.run(final_inputs)
        except Exception as e:
            result = f"Erreur lors de l'appel au modèle: {e}"

        # On affiche la réponse
        self._append_chat_message(f"**Bot**: {result}\n", area="chat")

        # Reset de la zone de saisie
        self.user_entry.delete("1.0", tk.END)

    # -------------------------------
    # Onglet PDF
    # -------------------------------
    def _build_pdf_ui(self, parent):
        # Frame de contrôle (haut)
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        self.load_pdf_button = tk.Button(control_frame, text="Charger un PDF", command=self.load_pdf)
        self.load_pdf_button.pack(side=tk.LEFT, padx=5)

        # Choix du modèle PDF + bouton
        self.pdf_model_combo = ttk.Combobox(control_frame, textvariable=self.pdf_model_var,
                                            values=self.models, state="readonly")
        self.pdf_model_combo.pack(side=tk.LEFT, padx=5)

        pdf_use_button = tk.Button(control_frame, text="Utiliser ce modèle", command=self.update_pdf_model)
        pdf_use_button.pack(side=tk.LEFT, padx=5)

        # Zone réponse PDF
        self.pdf_answer_area = scrolledtext.ScrolledText(parent, wrap=tk.WORD)
        self.pdf_answer_area.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)
        self.pdf_answer_area.config(state=tk.DISABLED)

        # Saisie question PDF
        self.pdf_question_entry = tk.Text(parent, width=70, height=3)
        self.pdf_question_entry.pack(pady=5)

        # Boutons
        self.ask_pdf_button = tk.Button(parent, text="Poser la question au PDF", command=self.ask_pdf_question)
        self.ask_pdf_button.pack(pady=5)

        self.summarize_pdf_button = tk.Button(parent, text="Demander un résumé du PDF", command=self.summarize_pdf)
        self.summarize_pdf_button.pack(pady=5)

    def update_pdf_model(self):
        """
        Sélection du modèle Ollama pour le PDF.
        """
        sel = self.pdf_model_var.get()
        self._append_chat_message(f"[INFO] Modèle PDF sélectionné : {sel}", area="pdf")

    def load_pdf(self):
        pdf_path = filedialog.askopenfilename(
            filetypes=[("PDF Files", "*.pdf")],
            title="Choisir un PDF"
        )
        if not pdf_path:
            return

        self._append_chat_message(f"[INFO] Chargement du PDF : {pdf_path}", area="pdf")
        docs = self._extract_text_by_page(pdf_path)
        collection_name = os.path.basename(pdf_path).replace(".pdf", "")

        self.vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=self.chroma_persist_dir
        )
        self.vectorstore.persist()

        self._append_chat_message(f"[INFO] PDF ingéré ({len(docs)} pages).", area="pdf")
        self._create_pdf_qa_chain()

    def _extract_text_by_page(self, pdf_path):
        docs = []
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc):
                text = page.get_text("text")
                metadata = {"source": pdf_path, "page": i+1}
                docs.append(Document(page_content=text, metadata=metadata))
        return docs

    def _create_pdf_qa_chain(self):
        if not self.vectorstore:
            return
        sel = self.pdf_model_var.get()
        if sel == "Aucun modèle trouvé":
            self._append_chat_message("Aucun modèle PDF sélectionné.", area="pdf")
            return

        llm_for_pdf = OllamaLLM(model=sel)
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        prompt_template = """Vous êtes un assistant qui s'appuie sur un document PDF.
Voici les extraits pertinents (pages) :

{context}

Question : {question}

Réponse en français :
"""
        pt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm_for_pdf,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": pt}
        )

        self._append_chat_message(f"[INFO] QA chain prête avec le modèle '{sel}'.", area="pdf")

    def ask_pdf_question(self):
        if not self.qa_chain:
            self._append_chat_message("Veuillez charger un PDF et sélectionner un modèle PDF.", area="pdf")
            return

        user_q = self.pdf_question_entry.get("1.0", tk.END).strip()
        if not user_q:
            return

        self._append_chat_message(f"**Question**: {user_q}", area="pdf")
        try:
            answer = self.qa_chain.run(user_q)
        except Exception as e:
            answer = f"Erreur : {e}"
        self._append_chat_message(f"**Réponse**: {answer}", area="pdf")
        self.pdf_question_entry.delete("1.0", tk.END)

    def summarize_pdf(self):
        if not self.vectorstore or not self.qa_chain:
            self._append_chat_message("PDF non prêt ou modèle PDF non sélectionné.", area="pdf")
            return

        sel = self.pdf_model_var.get()
        llm_for_pdf = OllamaLLM(model=sel)
        retriever = self.vectorstore.as_retriever(search_kwargs={"k":9999})

        summarize_prompt = PromptTemplate(
            template="""Vous êtes un assistant. 
On vous fournit des passages d'un document PDF. 
Faites-en un résumé synthétique.

{context}

Question: {question}

Résumé :
""",
            input_variables=["context","question"]
        )

        sum_chain = RetrievalQA.from_chain_type(
            llm=llm_for_pdf,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": summarize_prompt}
        )

        question = "Pouvez-vous me faire un résumé de ce PDF ?"
        self._append_chat_message(f"**Résumé demandé**: {question}", area="pdf")

        try:
            summary = sum_chain.run(question)
        except Exception as e:
            summary = f"Erreur : {e}"
        self._append_chat_message(f"**Résumé**: {summary}", area="pdf")

    def _append_chat_message(self, text, area="chat"):
        if area == "chat":
            self.conversation_area.config(state=tk.NORMAL)
            self.conversation_area.insert(tk.END, text + "\n")
            self.conversation_area.config(state=tk.DISABLED)
            self.conversation_area.see(tk.END)
        else:
            self.pdf_answer_area.config(state=tk.NORMAL)
            self.pdf_answer_area.insert(tk.END, text + "\n")
            self.pdf_answer_area.config(state=tk.DISABLED)
            self.pdf_answer_area.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = PDFChatApplication(root)
    root.mainloop()
