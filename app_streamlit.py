import os
import io
import time
import json
import zipfile
import uuid
import shutil
import unicodedata
import re
from pathlib import Path
from PyPDF2 import PdfReader, PdfWriter
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
import streamlit as st
from dotenv import load_dotenv

# =====================================================================
# CONFIGURAÇÃO INICIAL
# =====================================================================
load_dotenv()
st.set_page_config(page_title="Automatizador de Notas", page_icon="🧾", layout="wide")
st.title("🧠 Automatizador de Notas Fiscais PDF")

TEMP_FOLDER = Path("./temp")
os.makedirs(TEMP_FOLDER, exist_ok=True)

MAX_TOTAL_PAGES = int(os.getenv("MAX_TOTAL_PAGES", "50"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
MIN_RETRY_DELAY = int(os.getenv("MIN_RETRY_DELAY", "5"))
MAX_RETRY_DELAY = int(os.getenv("MAX_RETRY_DELAY", "30"))
MODEL_NAME = os.getenv("MODEL_NAME", "models/gemini-2.0-flash")

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ Chave GOOGLE_API_KEY não encontrada.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)
st.success("✅ Google Gemini configurado com sucesso!")

# =====================================================================
# NORMALIZAÇÃO E SUBSTITUIÇÕES
# =====================================================================
SUBSTITUICOES_FIXAS = {
    "COMPANHIA DE AGUA E ESGOTOS DA PARAIBA": "CAGEPA",
    "COMPANHIA DE AGUA E ESGOTOS DA PARAÍBA": "CAGEPA",
    "COMPANHIA DE AGUA E ESGOTO DA PARAIBA": "CAGEPA",
    "CIA DE AGUA E ESGOTO DO CEARA": "CAGECE",
    "COMPANHIA DE AGUA E ESGOTO DO CEARA": "CAGECE",
    "CAGECE": "CAGECE",
    "TRANSPORTE LIDA": "TRANSPORTE_LIDA",
    "TRANSPORTE LIDA LTDA": "TRANSPORTE_LIDA",
    "TRANSPORTELIDA": "TRANSPORTE_LIDA",
    "UNIPAR CARBOCLORO": "UNIPAR_CARBLOCLORO",
    "UNIPAR CARBOCLORO LTDA": "UNIPAR_CARBLOCLORO",
    "UNIPAR_CARBLOCLORO LTDA": "UNIPAR_CARBLOCLORO",
    "EXPRESS TCM": "EXPRESS_TCM",
    "EXPRESS TCM LTDA": "EXPRESS_TCM",
}

def _normalizar_texto(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII")
    s = re.sub(r"[^A-Z0-9 ]+", " ", s.upper())
    return re.sub(r"\s+", " ", s).strip()

def substituir_nome_emitente(nome_raw: str, cidade_raw: str = None) -> str:
    nome_norm = _normalizar_texto(nome_raw)
    cidade_norm = _normalizar_texto(cidade_raw) if cidade_raw else None

    # Sabará com cidade abreviada
    if "SABARA" in nome_norm:
        if cidade_norm:
            return f"SB_{cidade_norm}"
        return "SB"

    # Substituições fixas
    for padrao, substituto in SUBSTITUICOES_FIXAS.items():
        if _normalizar_texto(padrao) in nome_norm:
            return substituto

    return nome_norm

def limpar_emitente(nome: str) -> str:
    if not nome:
        return "SEM_NOME"
    nome = unicodedata.normalize("NFKD", nome).encode("ASCII", "ignore").decode("ASCII")
    nome = re.sub(r"[^A-Z0-9_]+", "_", nome.upper())
    while "__" in nome:
        nome = nome.replace("__", "_")
    return nome.strip("_")

def limpar_numero(numero: str) -> str:
    if not numero:
        return "0"
    numero = re.sub(r"[.\-,/ ]", "", numero)
    return numero.lstrip("0") or "0"

# =====================================================================
# RETRY GEMINI
# =====================================================================
def calcular_delay(tentativa, error_msg):
    if "retry in" in error_msg.lower():
        try:
            return min(float(re.search(r"retry in (\d+\.?\d*)s", error_msg.lower()).group(1)) + 2, MAX_RETRY_DELAY)
        except:
            pass
    return min(MIN_RETRY_DELAY * (tentativa + 1), MAX_RETRY_DELAY)

def chamar_gemini_retry(model, prompt_instrucao, page_stream):
    for tentativa in range(MAX_RETRIES + 1):
        try:
            start = time.time()
            resp = model.generate_content(
                [prompt_instrucao, {"mime_type": "application/pdf", "data": page_stream.getvalue()}],
                generation_config={"response_mime_type": "application/json"},
                request_options={'timeout': 60}
            )
            tempo = round(time.time() - start, 2)
            texto = resp.text.strip().lstrip("```json").rstrip("```").strip()
            dados = json.loads(texto)
            return dados, True, tempo
        except ResourceExhausted as e:
            delay = calcular_delay(tentativa, str(e))
            st.warning(f"⚠️ Quota excedida (tentativa {tentativa + 1}/{MAX_RETRIES}). Aguardando {delay}s...")
            time.sleep(delay)
        except Exception as e:
            if tentativa < MAX_RETRIES:
                time.sleep(MIN_RETRY_DELAY)
            else:
                return {"error": str(e)}, False, 0
    return {"error": "Falha máxima de tentativas"}, False, 0

# =====================================================================
# INTERFACE STREAMLIT
# =====================================================================
st.subheader("📎 Faça upload de um ou mais arquivos PDF")
uploaded_files = st.file_uploader("Selecione arquivos PDF", type=["pdf"], accept_multiple_files=True)

if uploaded_files and st.button("🚀 Processar PDFs"):
    session_id = str(uuid.uuid4())
    session_folder = TEMP_FOLDER / session_id
    os.makedirs(session_folder, exist_ok=True)

    resultados = []
    start_global = time.time()
    prompt = (
        "Analise a nota fiscal e extraia emitente, número da nota e cidade. "
        "Responda SOMENTE em JSON no formato: "
        "{\"emitente\":\"NOME\",\"numero_nota\":\"NUMERO\",\"cidade\":\"CIDADE\"}"
    )

    total_paginas = 0
    for f in uploaded_files:
        f_bytes = io.BytesIO(f.read())
        try:
            leitor = PdfReader(f_bytes)
            total_paginas += len(leitor.pages)
        except:
            continue

    progress_bar = st.progress(0.0)
    progresso_texto = st.empty()
    progresso = 0

    agrupados = {}

    for file_index, file in enumerate(uploaded_files):
        file.seek(0)
        pdf_bytes = io.BytesIO(file.read())
        try:
            leitor = PdfReader(pdf_bytes)
        except:
            continue

        for i, page in enumerate(leitor.pages):
            start_page_time = time.time()
            page_stream = io.BytesIO()
            writer = PdfWriter()
            writer.add_page(page)
            writer.write(page_stream)
            page_stream.seek(0)

            dados, ok, tempo_pagina = chamar_gemini_retry(model, prompt, page_stream)
            if not ok or "error" in dados:
                progresso += 1
                continue

            emitente_raw = dados.get("emitente", "") or ""
            numero_raw = dados.get("numero_nota", "") or ""
            cidade_raw = dados.get("cidade", "") or ""

            numero_limpo = limpar_numero(numero_raw)
            nome_map = substituir_nome_emitente(emitente_raw, cidade_raw)
            emitente_limpo = limpar_emitente(nome_map)
            chave = (numero_limpo, emitente_limpo)

            if chave not in agrupados:
                agrupados[chave] = []
            agrupados[chave].append(page_stream.getvalue())

            progresso += 1
            progress_bar.progress(min(progresso / total_paginas, 1.0))
            progresso_texto.markdown(f"⏱ Página {progresso}/{total_paginas} processada em {tempo_pagina:.2f}s")

    # Juntar PDFs por grupo
    for (numero, emitente), paginas in agrupados.items():
        writer = PdfWriter()
        for p_bytes in paginas:
            r = PdfReader(io.BytesIO(p_bytes))
            writer.add_page(r.pages[0])
        nome_pdf = f"DOC {numero}_{emitente}.pdf"
        with open(session_folder / nome_pdf, "wb") as f_out:
            writer.write(f_out)
        resultados.append({"novo": nome_pdf, "numero": numero, "emitente": emitente, "paginas": len(paginas)})

    # Editor de renomeação antes do download
    st.subheader("✏️ Renomeie antes de baixar")
    novos_nomes = {}
    for res in resultados:
        novo_nome = st.text_input(f"{res['emitente']} (DOC {res['numero']})", res['novo'])
        novos_nomes[res['novo']] = novo_nome

    # Compactar para download
    memory_zip = io.BytesIO()
    with zipfile.ZipFile(memory_zip, "w") as zf:
        for f in os.listdir(session_folder):
            nome_final = novos_nomes.get(f, f)
            zf.write(session_folder / f, arcname=nome_final)
    memory_zip.seek(0)

    tempo_total = round(time.time() - start_global, 2)
    st.success(f"🏁 Processamento concluído em {tempo_total}s.")

    st.download_button(
        "⬇️ Baixar arquivos processados",
        data=memory_zip,
        file_name="notas_processadas.zip",
        mime="application/zip"
    )
