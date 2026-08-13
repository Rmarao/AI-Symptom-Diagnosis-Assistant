# 🏥 AI Symptom Diagnosis Assistant

A Retrieval-Augmented Generation (RAG) symptom checker. You describe your symptoms in plain English, the app finds the closest-matching diseases in a local symptom database using semantic search, and a Groq-hosted LLM turns those matches into a plain-language explanation.

> ⚠️ **Not medical advice.** This is an educational project. Always consult a healthcare professional for an actual diagnosis.

## How it works

1. **Retrieve** — your symptom text is embedded with `sentence-transformers/all-MiniLM-L6-v2` and compared against a FAISS index of ~9,500 disease/symptom entries (`disease_database.csv` / `disease_index.faiss`) using cosine similarity.
2. **Augment** — the top 5 matching diseases (with their catalogued symptoms and similarity scores) are formatted into a prompt.
3. **Generate** — that prompt is sent to a Groq-hosted LLM, which explains the likely diagnosis, confidence level, and differentiating symptoms — grounded in the retrieved matches rather than the model's own unchecked recall.

Everything runs through a single-page [Streamlit](https://streamlit.io/) UI (`app.py`).

## Setup

**1. Create a virtual environment and install dependencies:**

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Configure your Groq API key:**

```bash
cp .env.example .env
```

Edit `.env` and set `GROQ_API_KEY` to a key from [console.groq.com/keys](https://console.groq.com/keys).

**3. Run the app:**

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Rebuilding the search index

`disease_index.faiss` is a pre-built FAISS index checked into the repo so the app runs out of the box. If you edit `disease_database.csv` (add/remove diseases or symptoms), regenerate the index with:

```bash
python build_index.py
```

This re-embeds every row with the same model `app.py` uses at query time and writes a fresh `disease_index.faiss`.

## Project structure

```text
.
├── app.py                  Streamlit app: UI, FAISS search, Groq LLM call
├── build_index.py          Rebuilds disease_index.faiss from disease_database.csv
├── disease_database.csv    Disease -> symptoms reference data
├── disease_index.faiss     Pre-built FAISS index over disease_database.csv
├── requirements.txt        Python dependencies
├── Procfile                Process definition for Heroku-style deployment
└── .env.example             Template for the required GROQ_API_KEY
```

## Deployment

The included `Procfile` runs `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`, suitable for Heroku-style platforms. Set `GROQ_API_KEY` as an environment variable on whatever platform you deploy to — don't commit your real `.env` file.

## Disclaimer

This tool is for educational purposes only. It does not provide medical advice, diagnosis, or treatment, and its output should never be used as a substitute for consulting a qualified healthcare provider.
