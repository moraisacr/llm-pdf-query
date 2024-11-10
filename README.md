
# Assistente Conversacional com Indexação de Documentos PDF usando LLM e Embeddings

## Descrição

Este projeto implementa um assistente conversacional que permite fazer perguntas sobre documentos PDF, baseando-se em modelos de linguagem (LLM) para criar e indexar embeddings textuais. Ao processar o conteúdo do PDF e armazenar representações vetoriais, o assistente pode responder perguntas sobre o conteúdo do documento com precisão. Essa abordagem é ideal para documentos extensos e técnicos, como especificações, normas e manuais, facilitando a navegação e o entendimento do conteúdo.

## Funcionalidades

- **Extração de Texto**: Extrai texto de PDFs, organizando-o por sentenças para maior precisão na indexação.
- **Criação de Embeddings**: Utiliza embeddings de frases para representar semanticamente o conteúdo do PDF.
- **Indexação e Busca com FAISS**: Indexa os embeddings para buscas rápidas e precisas, usando FAISS.
- **Interface Web**: Interface intuitiva criada com Gradio para que o usuário faça perguntas e receba respostas contextuais.
- **Cache de Embeddings**: Armazena embeddings para evitar reprocessamento de documentos já indexados.

## Tecnologias Utilizadas

- **Python**: Linguagem principal do projeto.
- **Sentence Transformers**: Biblioteca para criação de embeddings usando "bert-base-nli-mean-tokens".
- **FAISS (Facebook AI Similarity Search)**: Ferramenta para busca e indexação de alta performance.
- **PyPDF2**: Biblioteca para extração de texto de PDFs.
- **Gradio**: Biblioteca para criação de uma interface web interativa.
- **NumPy**: Utilizada para manipulação de arrays e processamento de embeddings.

## Como Usar

### Pré-requisitos

- Python 3.7 ou superior
- Instalar as dependências listadas no `requirements.txt`

### Instalação

1. Clone o repositório:

   ```bash
   git clone https://github.com/seu-usuario/nome-do-repositorio.git
   cd nome-do-repositorio
   ```

2. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

3. Baixe o PDF que deseja indexar ou defina o URL de um PDF.

### Executando o Projeto

Para iniciar o assistente, execute o seguinte comando:

```bash
python main.py
```

Isso iniciará o servidor Gradio. Acesse o endereço indicado no terminal para interagir com a interface web.

### Usando a Interface

1. **Carregar PDF**: O PDF será automaticamente processado e indexado.
2. **Fazer Perguntas**: Na interface, digite perguntas relacionadas ao conteúdo do PDF. O assistente responderá com base nas informações indexadas.

## Estrutura do Projeto

```
|-- main.py             # Script principal do assistente conversacional
|-- requirements.txt     # Lista de dependências do projeto
|-- README.md            # Descrição do projeto
```

## Exemplo de Perguntas

- "What is OAuth 2.0?"
- "What is an access token in OAuth 2.0?"
- "What is a refresh token in OAuth 2.0?"

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para enviar *pull requests* para melhorar o projeto.

## Licença

Este projeto é licenciado sob a [MIT License](LICENSE).
