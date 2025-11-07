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

# Diretórios temporários
TEMP_FOLDER = Path("./temp")
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Limites e modelo
MAX_TOTAL_PAGES = int(os.getenv("MAX_TOTAL_PAGES", "50"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
MIN_RETRY_DELAY = int(os.getenv("MIN_RETRY_DELAY", "5"))
MAX_RETRY_DELAY = int(os.getenv("MAX_RETRY_DELAY", "30"))
MODEL_NAME = os.getenv("MODEL_NAME", "models/gemini-2.0-flash")

# Configuração do Gemini
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ Chave GOOGLE_API_KEY não encontrada no .env ou nos segredos do Streamlit.")
    st.stop()
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)
st.success("✅ Google Gemini configurado com sucesso!")

# ==============================================================================
# Substituições e normalização
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
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII")
    s = re.sub(r"[^A-Z0-9 ]+", " ", s.upper())
    s = re.sub(r"\s+", " ", s).strip()
    return s

def substituir_nome_emitente(nome_raw: str, cidade_raw: str = None) -> str:
    nome_norm = _normalizar_texto(nome_raw)
    cidade_norm = _normalizar_texto(cidade_raw) if cidade_raw else None
    if "SABARA" in nome_norm and cidade_norm:
        return f"SABARA_{cidade_norm}"
    for padrao_raw, substituto in SUBSTITUICOES_NOMES.items():
        if _normalizar_texto(padrao_raw) in nome_norm:
            return substituto
    return nome_norm

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
# Função de retry do Gemini
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
            st.warning(f"⚠️ Quota excedida (tentativa {tentativa + 1}/{MAX_RETRIES}). Aguardando {delay}s...")
            time.sleep(delay)
        except Exception as e:
            if tentativa < MAX_RETRIES:
                st.warning(f"Tentativa {tentativa + 1} falhou, tentando novamente...")
                time.sleep(MIN_RETRY_DELAY)
            else:
                return {"error": str(e)}, False, 0
    return {"error": "Falha máxima de tentativas"}, False, 0

# ==============================================================================
# Interface Streamlit
# ==============================================================================
st.subheader("📎 Faça upload de um ou mais arquivos PDF")
uploaded_files = st.file_uploader("Selecione arquivos PDF", type=["pdf"], accept_multiple_files=True)

if uploaded_files and st.button("🚀 Processar PDFs"):
    session_id = str(uuid.uuid4())
    session_folder = TEMP_FOLDER / session_id
    os.makedirs(session_folder, exist_ok=True)

    resultados = []
    start_global = time.time()
    prompt = (
        "Analise a nota fiscal. Extraia emitente, número da nota e cidade. "
        "Responda SOMENTE em JSON: {\"emitente\":\"NOME\",\"numero_nota\":\"NUMERO\",\"cidade\":\"CIDADE\"}"
    )

    total_paginas = 0
    for f in uploaded_files:
        leitor = PdfReader(io.BytesIO(f.read()))
        total_paginas += len(leitor.pages)
        f.seek(0)

    st.info(f"📄 Total de páginas a processar: {total_paginas}")
    progress_bar = st.progress(0.0)
    progresso_texto = st.empty()
    progresso = 0

    for file_index, file in enumerate(uploaded_files):
        file_name = file.name
        pdf_bytes = io.BytesIO(file.read())
        leitor = PdfReader(pdf_bytes)

        for i, page in enumerate(leitor.pages):
            start_page_time = time.time()
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
                destino = session_folder / novo_nome
                with open(destino, "wb") as f_out:
                    f_out.write(page_stream.read())
                status_msg = "✅ Sucesso"
            else:
                status_msg = f"❌ {dados.get('error', 'Erro desconhecido')}"
                novo_nome = "-"

            progresso += 1
            progresso_atual = min(progresso / total_paginas, 1.0)
            progress_bar.progress(progresso_atual)
            progresso_texto.markdown(
                f"⏱ Página {progresso}/{total_paginas} — **{file_name} (pág {i+1})** → {status_msg} ({tempo_pagina:.2f}s)"
            )

            resultados.append({
                "original": file_name,
                "novo": novo_nome,
                "status": status_msg,
                "tempo": round(time.time() - start_page_time, 2)
            })

    tempo_total = round(time.time() - start_global, 2)
    st.success(f"🏁 Processamento concluído em {tempo_total}s ({len(resultados)} páginas).")

    # 📦 Compactar para download
    memory_zip = io.BytesIO()
    with zipfile.ZipFile(memory_zip, 'w') as zf:
        for f in os.listdir(session_folder):
            zf.write(session_folder / f, arcname=f)
    memory_zip.seek(0)

    st.download_button(
        "⬇️ Baixar arquivos processados",
        data=memory_zip,
        file_name="notas_processadas.zip",
        mime="application/zip"
    )

    st.subheader("📋 Resultados detalhados")
    st.dataframe(resultados)
