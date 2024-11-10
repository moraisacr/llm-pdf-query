import os
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import gradio as gr
import requests
import numpy as np

# Inicializar o modelo de embeddings com BERT
model = SentenceTransformer("bert-base-nli-mean-tokens")

# Extrair texto de arquivos PDF e dividir em sentenças, com tratamento de erros
def extract_text_from_pdf(pdf_file):
    print("Extraindo texto...")
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
                except Exception as e:
                    print(f"Erro ao extrair texto da página {page_num}: {e}")
    except Exception as e:
        print(f"Erro ao ler o arquivo PDF {pdf_file}: {e}")
    return texts

# Criar embeddings dos textos extraídos
def create_embeddings(texts, batch_size=16):
    print("Criando embeddings...")
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_embeddings = model.encode(texts[i:i + batch_size])
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)

# Indexar os embeddings utilizando FAISS (com Produto Interno)
def index_embeddings(embeddings):
    print("Indexando embeddings...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Produto Interno
    faiss.normalize_L2(embeddings)  # Normalizar embeddings para Produto Interno
    index.add(embeddings)
    return index

# Baixar PDF e criar índice, com cache de embeddings
def load_pdfs_and_create_index(pdf_url, pdf_file, cache_file="embeddings_cache.npy"):
    # Verificar se o cache existe
    if os.path.exists(cache_file):
        print("Carregando embeddings do cache...")
        embeddings = np.load(cache_file)
        texts = extract_text_from_pdf(pdf_file)
        index = index_embeddings(embeddings)
        return texts, index

    response = requests.get(pdf_url)
    if response.status_code == 200:
        with open(pdf_file, "wb") as file:
            file.write(response.content)
        print(f"PDF baixado com sucesso: {pdf_file}")

        texts = extract_text_from_pdf(pdf_file)
        embeddings = create_embeddings(texts)
        np.save(cache_file, embeddings)  # Salvar embeddings no cache
        index = index_embeddings(embeddings)
        return texts, index
    else:
        print(f"Falha ao baixar o PDF. Código de status: {response.status_code}")
        return [], None

# Buscar resposta com FAISS
def answer_question(question, texts, index):
    question_embedding = model.encode([question])
    faiss.normalize_L2(question_embedding)  # Normalizar o embedding da pergunta
    D, I = index.search(question_embedding, k=1)
    answer = texts[I[0][0]] if len(I[0]) > 0 else "Desculpe, não encontrei uma resposta."
    return answer

pdf_url = "https://github.com/moraisacr/llm-pdf-query/blob/main/RFC%206749%20-%20The%20OAuth%202.0%20Authorization%20Framework.pdf"
pdf_file = "OAuth2-RFC.pdf"
texts, index = load_pdfs_and_create_index(pdf_url, pdf_file)

if index is not None:
    def qa_interface(question):
        return answer_question(question, texts, index)

    iface = gr.Interface(fn=qa_interface, inputs="text", outputs="text", title="Assistente Conversacional")

    print("Iniciando o servidor...")
    iface.launch()
