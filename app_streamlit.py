import os
import io
import time
import json
import zipfile
import shutil
import unicodedata
import re
import uuid
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader, PdfWriter
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

# ==============================================================================
# CONFIGURAÇÃO DO APP
# ==============================================================================
load_dotenv()
st.set_page_config(page_title="Automatizador de Notas", layout="wide")

TEMP_FOLDER = Path(os.environ.get("TEMP_DIR", "./temp"))
os.makedirs(TEMP_FOLDER, exist_ok=True)

MAX_TOTAL_PAGES = int(os.getenv("MAX_TOTAL_PAGES", "50"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
MIN_RETRY_DELAY = int(os.getenv("MIN_RETRY_DELAY", "5"))
MAX_RETRY_DELAY = int(os.getenv("MAX_RETRY_DELAY", "30"))
MODEL_NAME = os.getenv("MODEL_NAME", "models/gemini-2.0-flash-exp")
DEBUG_SUBSTITUICOES = os.getenv("DEBUG_SUBSTITUICOES", "False").lower() in ("true", "1", "yes")

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ Chave GOOGLE_API_KEY não encontrada no arquivo .env")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)
st.success("✅ Gemini configurado com sucesso!")

# ==============================================================================
# FUNÇÕES DE NORMALIZAÇÃO
# ==============================================================================
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

def _normalizar_texto(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII")
    s = re.sub(r"[^A-Z0-9 ]+", " ", s.upper())
    s = re.sub(r"\s+", " ", s).strip()
    return s

def limpar_emitente(nome: str) -> str:
    if not nome: return "SEM_NOME"
    nome = unicodedata.normalize("NFKD", nome).encode("ASCII","ignore").decode("ASCII")
    nome = "".join(c if c.isalnum() else "_" for c in nome)
    while "__" in nome:
        nome = nome.replace("__","_")
    return nome.strip("_")

def limpar_numero(numero: str) -> str:
    if not numero: return "0"
    numero = re.sub(r'[.\-,/]','',numero)
    return numero.lstrip('0') or "0"

def substituir_nome_emitente(nome_raw: str, cidade_raw: str = None) -> str:
    nome_norm = _normalizar_texto(nome_raw)
    cidade_norm = _normalizar_texto(cidade_raw) if cidade_raw else None
    if DEBUG_SUBSTITUICOES:
        st.write(f"[DEBUG] Emitente: '{nome_raw}' -> '{nome_norm}', Cidade: '{cidade_raw}' -> '{cidade_norm}'")
    if "SABARA" in nome_norm and cidade_norm:
        return f"SABARA_{limpar_emitente(cidade_norm)}"
    for padrao_raw, substituto in SUBSTITUICOES_NOMES.items():
        if _normalizar_texto(padrao_raw) in nome_norm:
            return substituto
    return nome_norm

# ==============================================================================
# FUNÇÕES DE ESTADO E RETRY
# ==============================================================================
def chamar_gemini_retry(model, prompt_instrucao, page_stream):
    for tentativa in range(MAX_RETRIES + 1):
        try:
            start = time.time()
            resp = model.generate_content(
                [prompt_instrucao, {"mime_type": "application/pdf", "data": page_stream.getvalue()}],
                generation_config={"response_mime_type": "application/json"},
                request_options={'timeout': 30}
            )
            tempo = round(time.time() - start, 2)
            texto = resp.text.strip().lstrip("```json").rstrip("```").strip()
            dados = json.loads(texto)
            return dados, True, tempo
        except ResourceExhausted as e:
            delay = min(MIN_RETRY_DELAY * (tentativa + 1), MAX_RETRY_DELAY)
            st.warning(f"Quota excedida, aguardando {delay}s...")
            time.sleep(delay)
        except Exception as e:
            if tentativa < MAX_RETRIES:
                time.sleep(MIN_RETRY_DELAY)
            else:
                return {"error": str(e)}, False, 0
    return {"error": "Falha máxima de tentativas"}, False, 0

# ==============================================================================
# INTERFACE STREAMLIT
# ==============================================================================
st.title("🧾 Automatizador de Notas - IA Gemini")
st.markdown("Faça upload de **arquivos PDF de notas fiscais** e deixe a IA extrair e renomear automaticamente.")

uploaded_files = st.file_uploader("Selecione um ou mais PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files and st.button("🚀 Processar PDFs"):
    session_id = str(uuid.uuid4())
    session_folder = TEMP_FOLDER / session_id
    os.makedirs(session_folder, exist_ok=True)

    resultados = []
    total_paginas = 0
    start_global = time.time()

    prompt = (
        "Analise a nota fiscal. Extraia emitente, número da nota e cidade. "
        "Responda SOMENTE em JSON: {\"emitente\":\"NOME\",\"numero_nota\":\"NUMERO\",\"cidade\":\"CIDADE\"}"
    )

    progress_bar = st.progress(0)
    progresso = 0
    total_files = len(uploaded_files)

    for file_index, file in enumerate(uploaded_files):
        file_name = file.name
        pdf_bytes = io.BytesIO(file.read())

        try:
            leitor = PdfReader(pdf_bytes)
        except Exception as e:
            st.error(f"Erro ao ler {file_name}: {e}")
            continue

        for i, page in enumerate(leitor.pages):
            page_stream = io.BytesIO()
            writer = PdfWriter()
            writer.add_page(page)
            writer.write(page_stream)
            page_stream.seek(0)

            dados, ok, tempo = chamar_gemini_retry(model, prompt, page_stream)
            if ok and "error" not in dados:
                emitente = dados.get("emitente","")
                numero = dados.get("numero_nota","")
                cidade = dados.get("cidade","")
                numero_limpo = limpar_numero(numero)
                nome_map = substituir_nome_emitente(emitente, cidade)
                emitente_limpo = limpar_emitente(nome_map)
                novo_nome = f"DOC {numero_limpo}_{emitente_limpo}.pdf"
                destino = session_folder / novo_nome
                with open(destino, "wb") as f_out:
                    f_out.write(page_stream.read())
                resultados.append({"original": file_name, "novo": novo_nome, "tempo": tempo, "status": "✅ Sucesso"})
            else:
                resultados.append({"original": file_name, "novo": "-", "tempo": 0, "status": f"❌ {dados.get('error','Erro desconhecido')}"})

            progresso += 1
            progress_bar.progress(progresso / (total_files * len(leitor.pages)))

    tempo_total = round(time.time() - start_global, 2)
    st.success(f"🏁 Processamento concluído em {tempo_total}s!")

    # Tabela de resultados
    st.dataframe(resultados, use_container_width=True)

    # ZIP download
    memory_zip = io.BytesIO()
    with zipfile.ZipFile(memory_zip, "w") as zf:
        for f in os.listdir(session_folder):
            zf.write(session_folder / f, arcname=f)
    memory_zip.seek(0)

    st.download_button(
        label="📦 Baixar Notas Renomeadas (.zip)",
        data=memory_zip,
        file_name="notas_processadas.zip",
        mime="application/zip"
    )
