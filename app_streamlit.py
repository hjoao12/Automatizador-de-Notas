# =========================
# IMPORTAÇÕES
# =========================
import streamlit as st
import os
import io
import time
import json
import zipfile
import uuid
import shutil
import unicodedata
import re
import hashlib
import pickle
import base64
import gc
import pandas as pd
from pathlib import Path
from pypdf import PdfReader, PdfWriter
import google.generativeai as genai
from pdf2image import convert_from_bytes
from PIL import Image
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# CONFIGURAÇÃO INICIAL
# =========================
st.set_page_config(
    page_title="Automatizador de Notas Fiscais",
    page_icon="📄",
    layout="wide"
)

load_dotenv()

# =========================
# PASTA TEMPORÁRIA (CORREÇÃO CRÍTICA)
# =========================
TEMP_FOLDER = Path("./temp_sessions")
TEMP_FOLDER.mkdir(exist_ok=True)

# =========================
# SUPABASE (SEGURO)
# =========================
try:
    from supabase import create_client

    @st.cache_resource
    def init_supabase():
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        return create_client(url, key) if url and key else None

    supabase = init_supabase()
except Exception:
    supabase = None

# =========================
# CACHE DE DOCUMENTOS
# =========================
class DocumentCache:
    def __init__(self, cache_dir="./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def get_cache_key(self, img_bytes, prompt):
        return hashlib.md5(img_bytes + prompt.encode()).hexdigest()

    def get(self, key):
        p = self.cache_dir / f"{key}.pkl"
        if p.exists():
            try:
                with open(p, "rb") as f:
                    return pickle.load(f)
            except:
                return None
        return None

    def set(self, key, value):
        try:
            with open(self.cache_dir / f"{key}.pkl", "wb") as f:
                pickle.dump(value, f)
        except:
            pass

    def clear(self):
        for f in self.cache_dir.glob("*.pkl"):
            try:
                f.unlink()
            except:
                pass

document_cache = DocumentCache()

# =========================
# GEMINI
# =========================
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    st.error("GOOGLE_API_KEY não encontrada")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-2.5-flash")

# =========================
# UTILIDADES
# =========================
def normalizar(txt):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKD", txt or "").encode("ASCII","ignore").decode().upper()).strip()

def limpar_numero(n):
    n = re.sub(r"\D", "", str(n))
    return n.lstrip("0") or "000"

def limpar_emitente(e):
    return re.sub(r"_+", "_", re.sub(r"[^A-Z0-9]", "_", normalizar(e))) or "SEM_NOME"

# =========================
# OCR / GEMINI
# =========================
def extrair_pagina_imagem(pdf_bytes):
    imgs = convert_from_bytes(pdf_bytes, dpi=200)
    buf = io.BytesIO()
    imgs[0].save(buf, format="JPEG", quality=85)
    return buf.getvalue()

def processar_pagina_worker(job):
    gc.collect()

    img = extrair_pagina_imagem(job["bytes"])
    cache_key = document_cache.get_cache_key(img, job["prompt"])

    if job["use_cache"]:
        cached = document_cache.get(cache_key)
        if cached:
            return {**cached, "status": "CACHE"}

    image = Image.open(io.BytesIO(img))
    response = model.generate_content(
        [job["prompt"], image],
        generation_config=genai.types.GenerationConfig(
            temperature=0.1,
            response_mime_type="application/json"
        )
    )

    data = json.loads(response.text.replace("```json","").replace("```",""))
    result = {
        "status": "OK",
        "dados": data,
        "tempo": 0,
        "provider": "Gemini",
        "name": job["name"],
        "page_idx": job["page_idx"],
        "pdf_bytes": job["bytes"]
    }

    document_cache.set(cache_key, result)
    return result

# =========================
# PROMPT (ROBUSTO)
# =========================
PROMPT = """
Documento: NOTA FISCAL BRASILEIRA (PDF ESCANEADO).

Use APENAS OCR visual.
Extraia:
- emitente
- numero_nota (somente dígitos)
- cidade

Retorne SOMENTE:
{ "emitente": "", "numero_nota": "", "cidade": "" }
"""

# =========================
# UI
# =========================
st.title("Automatizador de Notas Fiscais PDF")

uploaded = st.file_uploader("Envie PDFs", type="pdf", accept_multiple_files=True)
processar = st.button("Processar")

if uploaded and processar:
    session_id = uuid.uuid4().hex
    session_folder = TEMP_FOLDER / session_id
    session_folder.mkdir()

    jobs = []
    for f in uploaded:
        r = PdfReader(io.BytesIO(f.read()))
        for i, p in enumerate(r.pages):
            w = PdfWriter()
            w.add_page(p)
            b = io.BytesIO()
            w.write(b)
            jobs.append({
                "bytes": b.getvalue(),
                "prompt": PROMPT,
                "name": f.name,
                "page_idx": i,
                "use_cache": True
            })

    agrupados = {}
    with ThreadPoolExecutor(max_workers=2) as ex:
        for fut in as_completed([ex.submit(processar_pagina_worker, j) for j in jobs]):
            r = fut.result()
            d = r["dados"]

            num = limpar_numero(d.get("numero_nota"))
            emit = limpar_emitente(d.get("emitente"))

            # 🔒 AGRUPAMENTO SEGURO (CORREÇÃO)
            key = (num, emit, r["name"])
            agrupados.setdefault(key, []).append(r["pdf_bytes"])

    resultados = []
    for (num, emit, origem), pages in agrupados.items():
        w = PdfWriter()
        for p in pages:
            rd = PdfReader(io.BytesIO(p))
            for pg in rd.pages:
                w.add_page(pg)

        nome = f"DOC {num}_{emit}.pdf"
        with open(session_folder / nome, "wb") as f:
            w.write(f)

        resultados.append(nome)

    st.success("Processamento concluído")
    zip_mem = io.BytesIO()
    with zipfile.ZipFile(zip_mem, "w") as z:
        for r in resultados:
            z.write(session_folder / r, arcname=r)
    zip_mem.seek(0)

    st.download_button("Baixar ZIP", zip_mem, "notas_processadas.zip", "application/zip")
