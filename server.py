from flask import Flask, request, jsonify
import os
from pathlib import Path
from dotenv import load_dotenv
from waitress import serve
import psycopg2
from psycopg2.extras import Json
import io
import pandas as pd
from PyPDF2 import PdfReader
import docx
import numpy as np
import uuid
from psycopg2.extras import Json


from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp

load_dotenv()
app = Flask(__name__)

# ====== KONFIG ======
LLM_MODEL_PATH = os.getenv("LLM_MODEL1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
USE_OPENAI = os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"
INDEX_DIR = os.getenv("INDEX_DIR", "./index")
CTX_NUMBER = 2048
MAX_TOKENS = 512
TOP_K = 3

DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT"))
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")

print(f"🔄 Loading LLaMA 2 model (CPU Windows safe): {LLM_MODEL_PATH}")
llm = LlamaCpp(
    model_path=LLM_MODEL_PATH,
    n_ctx=CTX_NUMBER,
    temperature=0.4,
    max_tokens=MAX_TOKENS,
    n_threads=1,
    n_batch=4,
    verbose=True
)

if USE_OPENAI:
    embeddings = OpenAIEmbeddings()
else:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

prompt_template = (
    "Kamu adalah asisten yang membantu. Jawablah singkat hanya berdasarkan konteks.\n\n"
    "KONTEKS:\n{context}\n\nPERTANYAAN:\n{question}\n\nJawaban:"
)
prompt_chain = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
chain = LLMChain(llm=llm, prompt=prompt_chain)

@app.route("/query", methods=["POST"])
def query():
    body = request.get_json(force=True)
    question = body.get("question", "").strip()

    if not question:
        return jsonify({"error": "Pertanyaan kosong"}), 400

    try:
        q_vector = embeddings.embed_query(question)
        q_vector = np.array(q_vector)
        print("🟢 Embedding pertanyaan:", q_vector[:10], "...")
    except Exception as e:
        print("❌ Gagal membuat embedding pertanyaan:", str(e))
        return jsonify({"error": f"Gagal membuat embedding pertanyaan: {str(e)}"}), 500

    try:
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
            user=DB_USER, password=DB_PASS
        )
        cur = conn.cursor()
        cur.execute("SELECT file_name, content, vector FROM embeddings")
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as e:
        return jsonify({"error": f"Gagal ambil embeddings dari DB: {str(e)}"}), 500

    contexts = []
    vectors = []
    for file_name, content, vector_json in rows:
        vec = np.array(vector_json)
        vectors.append((file_name, content, vec))

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

    sims = [(file_name, content, cosine_sim(q_vector, vec)) for file_name, content, vec in vectors]
    sims = sorted(sims, key=lambda x: x[2], reverse=True)[:TOP_K]

    context = "\n\n".join([f"Source: {f}\n{c}" for f, c, s in sims])
    if len(context) > CTX_NUMBER:
        context = context[-CTX_NUMBER:]

    result = chain.invoke({"context": context, "question": question})
    answer = result["text"] if isinstance(result, dict) else str(result)

    return jsonify({
        "question": question,
        "answer": answer,
        "context_used": context if context else None
    })

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang diunggah"}), 400

    file = request.files["file"]
    filename = file.filename
    file_bytes = file.read()
    content = ""

    try:
        mime_type = file.mimetype

        if mime_type == "application/pdf":
            reader = PdfReader(io.BytesIO(file_bytes))
            content = "\n".join([page.extract_text() or "" for page in reader.pages])

        elif mime_type in [
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword"
        ]:
            doc = docx.Document(io.BytesIO(file_bytes))
            content = "\n".join([p.text for p in doc.paragraphs])

        elif mime_type == "text/csv":
            df = pd.read_csv(io.BytesIO(file_bytes))
            content = df.to_string(index=False)

        elif mime_type in [
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ]:
            df = pd.read_excel(io.BytesIO(file_bytes))
            content = df.to_string(index=False)

        else:
            return jsonify({"error": f"Tipe file {mime_type} tidak didukung"}), 400

    except Exception as e:
        return jsonify({"error": f"Gagal membaca file: {str(e)}"}), 500

    try:
        vector = embeddings.embed_query(content)
    except Exception as e:
        return jsonify({"error": f"Gagal membuat embedding: {str(e)}"}), 500

    try:
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
            user=DB_USER, password=DB_PASS
        )
        cur = conn.cursor()
        user_id = uuid.UUID("fc639c93-c233-47be-8215-7536d7b9b2ef")
        cur.execute("""
            INSERT INTO embeddings (user_id, file_name, content, vector)
            VALUES (%s, %s, %s, %s)
        """, (str(user_id), filename, content, Json(vector)))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        return jsonify({"error": f"Gagal menyimpan ke DB: {str(e)}"}), 500

    return jsonify({"message": f"File {filename} berhasil diunggah dan disimpan"})


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=6000)
