# 📄 RAG Chatbot — Document Question Answering

This application allows you to upload a PDF document and chat with it using Google Gemini (Generative AI) + LangChain. It uses a Retrieval-Augmented Generation (RAG) approach to find answers from uploaded documents.

---

## 🚀 Features

- 📤 Upload a PDF file via web UI
- 🔍 Ask natural language questions about the document
- 🤖 Powered by Google Generative AI and LangChain
- 🧠 FAISS-powered local vector store
- ⚡ FastAPI backend + LangGraph orchestration
- 💬 Clean browser-based chatbot interface

---


## 🧰 Requirements

Install required packages:

```bash

python -m venv venv
source venv/bin/activate (mac )  
# Or 
venv\Scripts\activate on Windows
pip install -r requirements.txt

```
---

## 🧰 .env
create .env file

GOOGLE_API_KEY=your_google_generative_ai_key_here

## Run the Application
uvicorn main:app --reload


and run index.html in live server 