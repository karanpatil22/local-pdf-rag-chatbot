# Local PDF RAG Chatbot (Ollama)

## Requirements

- Python 3.10+
- Ollama installed
- nomic-embed-text model pulled
- llama3.2 model pulled

## Setup

Clone repo:

```bash
git clone https://github.com/yourusername/local-pdf-rag-chatbot.git
cd local-pdf-rag-chatbot

**Create virtual environment**
python -m venv venv
venv\Scripts\activate  # Windows

**Install dependencies**
pip install -r requirements.txt

**Pull Ollama models**
ollama pull llama3.2
ollama pull nomic-embed-text

**Run the app**
streamlit run app.py

**Now If You Pull On Another System**
git clone <repo-url>
cd project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
ollama pull llama3.2
ollama pull nomic-embed-text
streamlit run app.py

Built using Python 3.11
