# 🏨 Hotel Travel Assistant

An AI-powered travel assistant chatbot that helps users search for hotels, get personalized recommendations, check visa requirements, and query reviews — built as a university team project.

---

## 🧠 How It Works

1. **Input Preprocessing** — User query is cleaned and normalized using spaCy
2. **Intent Classification** — Groq LLM classifies the query into intents: hotel search, recommendation, visa inquiry, review query, and more
3. **Entity Extraction** — spaCy PhraseMatcher + regex extracts hotels, locations, traveler type, ratings
4. **Semantic Search** — FAISS vector index with sentence transformers (BGE, SBERT) retrieves relevant results
5. **Graph Queries** — Neo4j graph database handles relationship-based hotel and location queries
6. **Response Generation** — LLM generates a natural language response via Groq API
7. **Streamlit UI** — Clean web interface for interacting with the assistant

---

## 🛠 Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Neo4j](https://img.shields.io/badge/Neo4j-008CC1?style=flat&logo=neo4j&logoColor=white)

**Libraries:** Groq · spaCy · FAISS · SentenceTransformers · Pandas · SHAP · LIME

---

## 🏗 Project Structure

    pipeline/          # Core RAG and intent pipeline
    utils/             # Helper functions
    config/            # Configuration files
    streamlit_app.py   # Main Streamlit interface
    milestone_3.ipynb  # Full pipeline notebook

---

## ▶️ How to Run

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## 🎓 Course

Advanced Computer Lab 2 — German University in Cairo, Fall 2024  
Team: Omar Tamer · Jaydaa Roushdy · Aly Sherif
