import os
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import gradio as gr
import requests
import numpy as np

# Inicializar o modelo de embeddings com BERT
model = SentenceTransformer("bert-base-nli-mean-tokens")

# Função para baixar arquivos do Google Drive
def download_file_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # Filtro para ignorar chunks vazios
                f.write(chunk)

# Extrair texto de arquivos PDF e dividir em sentenças, com tratamento de erros
def extract_text_from_pdf(pdf_file):
    print("Extracting text...")
    texts = []
    try:
        with open(pdf_file, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            for page_num in range(num_pages):
                try:
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    if text:
                        # Limpeza básica do texto
                        cleaned_text = ' '.join(text.strip().split())
                        texts.append(cleaned_text)
                        print(f"Page {page_num + 1}: {cleaned_text[:100]}...")  # Mostrar um trecho do texto
                except Exception as e:
                    print(f"Error extracting text from page {page_num}: {e}")
    except Exception as e:
        print(f"Error reading PDF file {pdf_file}: {e}")
    if not texts:
        print("Aviso: Nenhum texto foi extraído do PDF.")
    return texts

# Criar embeddings dos textos extraídos
def create_embeddings(texts, batch_size=16):
    print("Creating embeddings...")
    if not texts:
        print("Erro: Nenhum texto para criar embeddings.")
        return np.array([])  # Retorna um array vazio se não houver texto

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Limpeza básica do texto no lote
        batch = [text for text in batch if len(text.strip()) > 0]  # Remover textos vazios
        batch_embeddings = model.encode(batch)
        embeddings.extend(batch_embeddings)

    embeddings = np.array(embeddings)
    if embeddings.size == 0:
        print("Erro: Nenhum embedding foi criado.")
    else:
        print(f"Embeddings criados com sucesso. Forma final dos embeddings: {embeddings.shape}")
    return embeddings




# Indexar os embeddings utilizando FAISS (com Produto Interno)
def index_embeddings(embeddings):
    print("Indexing embeddings...")
    if embeddings.size == 0:
        print("Erro: Nenhum embedding disponível para indexação.")
        return None  # Retorna None se não houver embeddings
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Produto Interno
    faiss.normalize_L2(embeddings)  # Normalizar embeddings para Produto Interno
    index.add(embeddings)
    return index

# Baixar PDF e criar índice, com cache de embeddings
def load_pdfs_and_create_index(pdf_file, cache_file="embeddings_cache.npy"):
    # Verificar se o cache existe
    if os.path.exists(cache_file):
        print("Loading embeddings from cache...")
        embeddings = np.load(cache_file)
        texts = extract_text_from_pdf(pdf_file)
        index = index_embeddings(embeddings)
        return texts, index

    # Extrair texto do PDF e criar embeddings
    texts = extract_text_from_pdf(pdf_file)
    embeddings = create_embeddings(texts)
    if embeddings.size == 0:
        print("Erro: Nenhum embedding foi criado.")
        return texts, None  # Retorna None se não houver embeddings
    np.save(cache_file, embeddings)  # Salvar embeddings no cache
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

# ID do Google Drive e caminho do arquivo local
file_id = "1HduupYU4cM4iPuJPL5tTP60tA8RMMSqw"
pdf_file = "OAuth2-RFC.pdf"

# Baixar o PDF do Google Drive
download_file_from_google_drive(file_id, pdf_file)

# Carregar o PDF e criar o índice
texts, index = load_pdfs_and_create_index(pdf_file)

if index is not None:
    def qa_interface(question):
        return answer_question(question, texts, index)

    iface = gr.Interface(fn=qa_interface, inputs="text", outputs="text", title="Assistente Conversacional")

    print("Starting server...")
    iface.launch()
