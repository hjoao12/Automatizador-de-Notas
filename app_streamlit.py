import os
import io
import time
import json
import uuid
import zipfile
import shutil
import unicodedata
import re
import requests
from pathlib import Path
from PyPDF2 import PdfReader, PdfWriter
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

# ======================================================================
# CONFIGURAÇÃO INICIAL
# ======================================================================
load_dotenv()

TEMP_FOLDER = Path("./temp")
os.makedirs(TEMP_FOLDER, exist_ok=True)

MAX_RETRIES = 1
MIN_RETRY_DELAY = 3
MAX_RETRY_DELAY = 10
MAX_TOTAL_PAGES = 50

# IA Keys
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Config Gemini
genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = os.getenv("MODEL_NAME", "models/gemini-2.0-flash")

# ======================================================================
# FUNÇÕES DE NORMALIZAÇÃO
# ======================================================================
def _normalizar_texto(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII")
    s = re.sub(r"[^A-Z0-9 ]+", " ", s.upper())
    return re.sub(r"\s+", " ", s).strip()

def limpar_emitente(nome: str) -> str:
    if not nome:
        return "SEM_NOME"
    nome = unicodedata.normalize("NFKD", nome).encode("ASCII", "ignore").decode("ASCII")
    nome = "".join(c if c.isalnum() else "_" for c in nome)
    return re.sub(r"_+", "_", nome).strip("_")

def limpar_numero(numero: str) -> str:
    if not numero:
        return "0"
    numero = re.sub(r"[.\-,/]", "", numero)
    return numero.lstrip("0") or "0"

def substituir_nome_emitente(nome_raw: str, cidade_raw: str = None) -> str:
    SUBSTITUICOES_NOMES = {
        "COMPANHIA DE AGUA E ESGOTOS DA PARAIBA": "CAGEPA",
        "COMPANHIA DE AGUA E ESGOTOS DA PARAÍBA": "CAGEPA",
        "COMPANHIA DE AGUA E ESGOTO DA PARAIBA": "CAGEPA",
        "CIA DE AGUA E ESGOTO DO CEARA": "CAGECE",
        "COMPANHIA DE AGUA E ESGOTO DO CEARA": "CAGECE",
        "CAGECE": "CAGECE",
        "SABARA QUIMICOS E INGREDIENTES SA": "SABARA",
        "SABARA QUIMICOS E INGREDIENTES LTDA": "SABARA",
        "SABARÁ QUIMICOS E INGREDIENTES SA": "SABARA",
        "SABARÁ QUIMICOS E INGREDIENTES LTDA": "SABARA",
    }

    nome_norm = _normalizar_texto(nome_raw)
    for padrao, subst in SUBSTITUICOES_NOMES.items():
        if _normalizar_texto(padrao) in nome_norm:
            return subst
    return nome_norm

# ======================================================================
# FALLBACK AUTOMÁTICO DE IAs
# ======================================================================
def chamar_ia_fallback(pdf_bytes: io.BytesIO, prompt_instrucao: str):
    """
    Tenta usar Gemini -> DeepSeek -> OpenAI automaticamente.
    """
    pdf_bytes.seek(0)
    pdf_data = pdf_bytes.getvalue()

    # 1️⃣ GEMINI
    try:
        start = time.time()
        model = genai.GenerativeModel(MODEL_NAME)
        resp = model.generate_content(
            [prompt_instrucao, {"mime_type": "application/pdf", "data": pdf_data}],
            generation_config={"response_mime_type": "application/json"},
            request_options={"timeout": 30}
        )
        texto = resp.text.strip().lstrip("```json").rstrip("```").strip()
        dados = json.loads(texto)
        tempo = round(time.time() - start, 2)
        print(f"✅ Gemini OK ({tempo}s)")
        return dados, "Gemini", tempo
    except Exception as e:
        print(f"⚠️ Falha Gemini: {e}")

    # 2️⃣ DEEPSEEK (fallback rápido via API HTTP)
    if DEEPSEEK_API_KEY:
        try:
            start = time.time()
            headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
            files = {"file": ("nota.pdf", pdf_data, "application/pdf")}
            data = {
                "model": "deepseek-chat",
                "prompt": prompt_instrucao,
                "max_tokens": 200
            }
            resp = requests.post("https://api.deepseek.com/v1/files/analyze", headers=headers, files=files, data=data, timeout=20)
            if resp.status_code == 200:
                dados = resp.json()
                tempo = round(time.time() - start, 2)
                print(f"✅ DeepSeek OK ({tempo}s)")
                return dados, "DeepSeek", tempo
            else:
                print(f"⚠️ DeepSeek erro {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"⚠️ Falha DeepSeek: {e}")

    # 3️⃣ OPENAI (fallback final)
    if OPENAI_API_KEY:
        try:
            import openai
            openai.api_key = OPENAI_API_KEY
            start = time.time()
            completion = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Você extrai dados de notas fiscais de PDFs e responde apenas JSON."},
                    {"role": "user", "content": prompt_instrucao}
                ],
                timeout=20
            )
            texto = completion.choices[0].message["content"]
            dados = json.loads(texto.strip().lstrip("```json").rstrip("```").strip())
            tempo = round(time.time() - start, 2)
            print(f"✅ OpenAI OK ({tempo}s)")
            return dados, "OpenAI", tempo
        except Exception as e:
            print(f"⚠️ Falha OpenAI: {e}")

    return {"error": "Nenhuma IA respondeu"}, "Nenhuma", 0

# ======================================================================
# INTERFACE STREAMLIT
# ======================================================================
st.set_page_config(page_title="Automatizador de Notas com Fallback IA", layout="wide")
st.title("📄 Automatizador de Notas com Fallback IA")

uploaded_files = st.file_uploader("Envie seus PDFs de notas fiscais", type=["pdf"], accept_multiple_files=True)
if st.button("Processar Notas") and uploaded_files:
    session_id = str(uuid.uuid4())
    session_folder = TEMP_FOLDER / session_id
    os.makedirs(session_folder, exist_ok=True)

    progress_bar = st.progress(0.0)
    logs = st.empty()
    total_files = len(uploaded_files)
    progresso = 0
    resultados = []

    prompt_instrucao = (
        "Analise a nota fiscal em PDF. Extraia emitente, número da nota e cidade. "
        "Responda SOMENTE em JSON: {\"emitente\":\"NOME\",\"numero_nota\":\"NUMERO\",\"cidade\":\"CIDADE\"}"
    )

    for idx, file in enumerate(uploaded_files):
        try:
            pdf_bytes = io.BytesIO(file.read())
            leitor = PdfReader(pdf_bytes)

            for i, page in enumerate(leitor.pages):
                writer = PdfWriter()
                page_stream = io.BytesIO()
                writer.add_page(page)
                writer.write(page_stream)
                page_stream.seek(0)

                dados, modelo_usado, tempo = chamar_ia_fallback(page_stream, prompt_instrucao)
                sucesso = False
                novo_nome = f"ERRO_{file.name}_p{i+1}.pdf"

                if "error" not in dados and dados.get("emitente") and dados.get("numero_nota"):
                    emitente = dados.get("emitente", "")
                    numero = limpar_numero(dados.get("numero_nota", ""))
                    cidade = dados.get("cidade", "")
                    emitente_final = limpar_emitente(substituir_nome_emitente(emitente, cidade))
                    novo_nome = f"DOC {numero}_{emitente_final}.pdf"
                    sucesso = True

                with open(session_folder / novo_nome, "wb") as f:
                    page_stream.seek(0)
                    f.write(page_stream.read())

                resultados.append({
                    "original": f"{file.name} (pág {i+1})",
                    "novo": novo_nome,
                    "modelo": modelo_usado,
                    "tempo": tempo,
                    "status": "Sucesso" if sucesso else "Falha"
                })

                progresso += 1
                progress_bar.progress(min(progresso / (total_files * len(leitor.pages)), 1.0))
                logs.write(f"📄 {file.name} pág {i+1} → {novo_nome} ({modelo_usado})")

        except Exception as e:
            st.error(f"Erro ao processar {file.name}: {e}")

    st.success("✅ Processamento concluído!")

    # Download ZIP
    zip_path = session_folder / "notas_processadas.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in session_folder.glob("*.pdf"):
            zf.write(f, arcname=f.name)

    with open(zip_path, "rb") as f:
        st.download_button("⬇️ Baixar ZIP com notas processadas", f, file_name="notas_processadas.zip")

    st.subheader("📊 Resultados")
    st.dataframe(resultados)
