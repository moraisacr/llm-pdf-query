import os
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import gradio as gr
import numpy as np

# Inicializar o modelo de embeddings com BERT
model = SentenceTransformer("bert-base-nli-mean-tokens")

# Extrair texto de arquivos PDF e dividir em sentenças
def extract_text_from_pdf(pdf_file):
    texts = []
    try:
        with open(pdf_file, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text = page.extract_text()
                if text:
                    cleaned_text = ' '.join(text.strip().split())
                    texts.append(cleaned_text)
    except Exception as e:
        print(f"Error reading PDF file {pdf_file}: {e}")
    return texts

# Criar embeddings dos textos extraídos
def create_embeddings(texts, batch_size=16):
    if not texts:
        return np.array([])  # Retorna um array vazio se não houver texto

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch = [text for text in batch if len(text.strip()) > 0]  # Remover textos vazios
        batch_embeddings = model.encode(batch)
        embeddings.extend(batch_embeddings)

    embeddings = np.array(embeddings)
    return embeddings

# Indexar os embeddings utilizando FAISS (com Produto Interno)
def index_embeddings(embeddings):
    if embeddings.size == 0:
        return None  # Retorna None se não houver embeddings
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Produto Interno
    faiss.normalize_L2(embeddings)  # Normalizar embeddings para Produto Interno
    index.add(embeddings)
    return index

# Carregar o PDF e criar o índice
def load_pdfs_and_create_index(pdf_file):
    texts = extract_text_from_pdf(pdf_file)
    embeddings = create_embeddings(texts)
    if embeddings.size == 0:
        return texts, None  # Retorna None se não houver embeddings
    index = index_embeddings(embeddings)
    return texts, index

# Buscar resposta com FAISS
def answer_question(question, texts, index):
    if index is None:
        return "Erro: O índice de embeddings não está disponível."
    question_embedding = model.encode([question])
    faiss.normalize_L2(question_embedding)  # Normalizar o embedding da pergunta
    D, I = index.search(question_embedding, k=1)
    answer = texts[I[0][0]] if len(I[0]) > 0 else "Desculpe, não encontrei uma resposta."
    return answer

# Função principal para Gradio, permite o upload do PDF e processamento da pergunta
def process_pdf_and_answer_question(pdf_file, question):
    texts, index = load_pdfs_and_create_index(pdf_file)
    if index is None:
        return "Erro: Não foi possível processar o arquivo PDF."
    return answer_question(question, texts, index)

# Interface Gradio
iface = gr.Interface(
    fn=process_pdf_and_answer_question,
    inputs=[
        gr.File(type="filepath", label="Carregar Arquivo PDF"),
        gr.Textbox(lines=2, label="Pergunta")
    ],
    outputs="text",
    title="Assistente Conversacional"
)

print("Starting server...")
iface.launch()
