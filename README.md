# 📊 Text-to-SQL Chatbot

An AI-powered chatbot that converts natural language questions into SQL queries using LangChain and Groq LLM.

## Features
- 📂 Upload any CSV file — auto creates PostgreSQL table
- 💬 Ask questions in plain English — get SQL + Answer
- 📊 Auto chart generation (bar/line/pie)
- 💡 AI-generated dataset insights
- 🕘 Persistent query history per dataset
- 🔗 Multiple CSV + JOIN query support

## Tech Stack
Python | LangChain | Groq LLaMA 3.3 | PostgreSQL | Streamlit | Matplotlib

## How to Run
1. Clone the repo
2. Install requirements: `pip install -r requirements.txt`
3. Add Groq API key in `.streamlit/secrets.toml`
4. Run: `streamlit run app.py`
