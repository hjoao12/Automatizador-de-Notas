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

# ==============================================================================
# Configuração inicial
# ==============================================================================
load_dotenv()
st.set_page_config(page_title="Automatizador de Notas", page_icon="🧾", layout="wide")
st.title("🧠 Automatizador de Notas Fiscais PDF")

TEMP_FOLDER = Path("./temp")
os.makedirs(TEMP_FOLDER, exist_ok=True)

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

# ==============================================================================
# Padronização de nomes
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
    "UNIPAR CARBOCLORO LTDA": "UNIPAR_CARBOCLORO",
    "UNIPAR CARBOCLORO SA": "UNIPAR_CARBOCLORO",
    "EXPRESS TCM LTDA": "EXPRESS_TCM",
    "TRANSPORTE LIDA LTDA": "TRANSPORTE_LIDA",
    "TRANSPORTELIDA LTDA": "TRANSPORTE_LIDA",
    "TRANSPORTELIDA": "TRANSPORTE_LIDA",
}

def _normalizar_texto(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII")
    s = re.sub(r"[^A-Z0-9 ]+", " ", s.upper())
    s = re.sub(r"\s+", " ", s).strip()
    return s

def substituir_nome_emitente(nome_raw: str, cidade_raw: str = None) -> str:
    nome_norm = _normalizar_texto(nome_raw)
    cidade_norm = _normalizar_texto(cidade_raw) if cidade_raw else None

    # Sabará - único caso que usa cidade
    if "SABARA" in nome_norm and cidade_norm:
        return f"SB_{cidade_norm}"

    for padrao_raw, substituto in SUBSTITUICOES_NOMES.items():
        if _normalizar_texto(padrao_raw) in nome_norm:
            return substituto

    return re.sub(r"\s+", "_", nome_norm.strip())

def limpar_emitente(nome: str) -> str:
    if not nome:
        return "SEM_NOME"
    nome = unicodedata.normalize("NFKD", nome).encode("ASCII", "ignore").decode("ASCII")
    nome = "".join(c if c.isalnum() else "_" for c in nome)
    while "__" in nome:
        nome = nome.replace("__", "_")
    return nome.strip("_")

def limpar_numero(numero: str) -> str:
    if not numero:
        return "0"
    numero = re.sub(r"[.\-,/]", "", numero)
    return numero.lstrip("0") or "0"

# ==============================================================================
# Retry do Gemini
# ==============================================================================
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
            st.warning(f"⚠️ Quota excedida. Tentando novamente em {delay}s...")
            time.sleep(delay)
        except Exception as e:
            if tentativa < MAX_RETRIES:
                st.warning(f"Tentativa {tentativa + 1} falhou. Repetindo...")
                time.sleep(MIN_RETRY_DELAY)
            else:
                return {"error": str(e)}, False, 0
    return {"error": "Falha máxima de tentativas"}, False, 0

# ==============================================================================
# Interface
# ==============================================================================
st.subheader("📎 Faça upload de um ou mais PDFs")
uploaded_files = st.file_uploader("Selecione arquivos PDF", type=["pdf"], accept_multiple_files=True)

if uploaded_files and st.button("🚀 Processar PDFs"):
    session_id = str(uuid.uuid4())
    session_folder = TEMP_FOLDER / session_id
    os.makedirs(session_folder, exist_ok=True)

    resultados = {}
    prompt = (
        "Analise a nota fiscal. Extraia emitente, número da nota e cidade. "
        "Responda SOMENTE em JSON: {\"emitente\":\"NOME\",\"numero_nota\":\"NUMERO\",\"cidade\":\"CIDADE\"}"
    )

    total_paginas = sum(len(PdfReader(io.BytesIO(f.read())).pages) for f in uploaded_files)
    st.info(f"📄 Total de páginas: {total_paginas}")

    progress_bar = st.progress(0.0)
    progresso_texto = st.empty()
    progresso = 0

    for file in uploaded_files:
        pdf_bytes = io.BytesIO(file.read())
        leitor = PdfReader(pdf_bytes)

        for i, page in enumerate(leitor.pages):
            start_time = time.time()
            page_stream = io.BytesIO()
            writer = PdfWriter()
            writer.add_page(page)
            writer.write(page_stream)
            page_stream.seek(0)

            dados, ok, tempo_pagina = chamar_gemini_retry(model, prompt, page_stream)
            if ok and "error" not in dados:
                emitente = dados.get("emitente", "")
                numero = dados.get("numero_nota", "")
                cidade = dados.get("cidade", "")
                numero_limpo = limpar_numero(numero)
                nome_map = substituir_nome_emitente(emitente, cidade)
                emitente_limpo = limpar_emitente(nome_map)
                novo_nome = f"DOC {numero_limpo}_{emitente_limpo}.pdf"

                if numero_limpo not in resultados:
                    resultados[numero_limpo] = {"emitente": emitente_limpo, "arquivos": []}
                resultados[numero_limpo]["arquivos"].append(page_stream.getvalue())

                status = "✅ Sucesso"
            else:
                status = f"❌ {dados.get('error', 'Erro desconhecido')}"

            progresso += 1
            progress_bar.progress(min(progresso / total_paginas, 1.0))
            progresso_texto.text(f"⏱ Página {progresso}/{total_paginas} — {status} ({tempo_pagina:.2f}s)")

    # Agrupar e salvar PDFs
    tabelas = []
    for numero, dados in resultados.items():
        saida = session_folder / f"DOC {numero}_{dados['emitente']}.pdf"
        writer = PdfWriter()
        for conteudo in dados["arquivos"]:
            leitor = PdfReader(io.BytesIO(conteudo))
            for p in leitor.pages:
                writer.add_page(p)
        with open(saida, "wb") as f_out:
            writer.write(f_out)
        tabelas.append({"numero": numero, "emitente": dados["emitente"], "novo_nome": saida.name})

    # Edição manual antes de baixar
    st.subheader("✏️ Revise e edite nomes antes de baixar")
    edited_df = st.data_editor(tabelas, num_rows="dynamic", key="edicoes")

    # Compactar
    memory_zip = io.BytesIO()
    with zipfile.ZipFile(memory_zip, 'w') as zf:
        for row in edited_df:
            origem = session_folder / row["novo_nome"]
            zf.write(origem, arcname=row["novo_nome"])
    memory_zip.seek(0)

    st.download_button(
        "⬇️ Baixar arquivos processados",
        data=memory_zip,
        file_name="notas_processadas.zip",
        mime="application/zip"
    )
