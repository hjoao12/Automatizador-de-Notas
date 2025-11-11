# app_streamlit.py
import os
import io
import time
import json
import zipfile
import uuid
import unicodedata
import re
from pathlib import Path
from PyPDF2 import PdfReader, PdfWriter
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
import streamlit as st
from dotenv import load_dotenv

# ----------------------------
# Config
# ----------------------------
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
    st.error("❌ GOOGLE_API_KEY não encontrada.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

# ----------------------------
# Padronizações
# ----------------------------
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
    "TRANSPORTE LIDA LTDA": "TRANSPORTE_LIDA",
    "TRANSPORTE LIDA": "TRANSPORTE_LIDA",
    "TRANSPORTELIDA": "TRANSPORTE_LIDA",
    "UNIPAR CARBLOCLORO LTDA": "UNIPAR_CARBOCLORO",
    "UNIPAR CARBLOCLORO S A": "UNIPAR_CARBOCLORO",
    "UNIPAR CARBOCLORO": "UNIPAR_CARBOCLORO",
    "EXPRESS TCM LTDA": "EXPRESS_TCM",
    "EXPRESS TCM": "EXPRESS_TCM",
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
    cidade_norm = _normalizar_texto(cidade_raw) if cidade_raw else ""
    # Sabará uses city abbreviation SB_CIDADE
    if "SABARA" in nome_norm and cidade_norm:
        # only first token of city
        return f"SB_{cidade_norm.split()[0]}"
    for padrao, sub in SUBSTITUICOES_NOMES.items():
        if _normalizar_texto(padrao) in nome_norm:
            return sub
    # default: replace spaces by underscore
    return re.sub(r"\s+", "_", nome_norm)

def limpar_emitente(nome: str) -> str:
    if not nome:
        return "SEM_NOME"
    nome = unicodedata.normalize("NFKD", nome).encode("ASCII", "ignore").decode("ASCII")
    nome = "".join(c if c.isalnum() else "_" for c in nome)
    return re.sub(r"_+", "_", nome).strip("_")

def limpar_numero(num_raw: str) -> str:
    if not num_raw:
        return "0"
    # extract digits sequence that is the invoice number (first long digits group)
    # keep fallback to remove punctuation
    m = re.search(r"(\d{3,})", str(num_raw))
    if m:
        return m.group(1).lstrip("0") or "0"
    cleaned = re.sub(r"[^\d]", "", str(num_raw))
    return cleaned.lstrip("0") or "0"

# ----------------------------
# Extrair parte/total de strings (1/4, 1 de 4, 1-4, (1/4), nº 123/1)
# returns (part:int, total:int) or None
# ----------------------------
def parse_sequencia(text: str):
    if not text:
        return None
    s = str(text).lower()
    # common patterns: 1/4, 1 de 4, 1 - 4, (1/4)
    m = re.search(r"(\d{1,3})\s*/\s*(\d{1,3})", s)
    if not m:
        m = re.search(r"(\d{1,3})\s+de\s+(\d{1,3})", s)
    if not m:
        m = re.search(r"(\d{1,3})\s*-\s*(\d{1,3})", s)
    if m:
        try:
            part = int(m.group(1))
            total = int(m.group(2))
            if total >= part >= 1:
                return (part, total)
        except:
            return None
    # sometimes appears like '1/4 NF 123' or 'NF 123 - 1/4' handled by above
    return None

# ----------------------------
# Gemini retry helper
# ----------------------------
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
            st.warning(f"⚠️ Quota excedida. Tentativa {tentativa+1}. Aguardando {delay}s...")
            time.sleep(delay)
        except Exception as e:
            if tentativa < MAX_RETRIES:
                time.sleep(MIN_RETRY_DELAY)
            else:
                return {"error": str(e)}, False, 0
    return {"error": "Máximo de tentativas"}, False, 0

# ----------------------------
# UI: upload and processing
# ----------------------------
st.subheader("📎 Faça upload dos PDFs (várias páginas/partes são aceitas)")
uploaded_files = st.file_uploader("Selecione arquivos PDF", type=["pdf"], accept_multiple_files=True)

if uploaded_files and st.button("🚀 Processar PDFs"):
    session_id = str(uuid.uuid4())
    session_folder = TEMP_FOLDER / session_id
    os.makedirs(session_folder, exist_ok=True)

    # read files once
    arquivos = []
    for f in uploaded_files:
        content = f.read()
        arquivos.append({"name": f.name, "bytes": content})

    # count pages safely
    total_paginas = 0
    for a in arquivos:
        try:
            r = PdfReader(io.BytesIO(a["bytes"]))
            total_paginas += len(r.pages)
        except Exception:
            st.warning(f"Arquivo {a['name']} inválido/oculto — será ignorado.")
    st.info(f"📄 Total de páginas a processar: {total_paginas}")

    prompt = ("Analise a nota fiscal (DANFE). Extraia emitente, número da nota e, se presente, sequência (ex: 1/4). "
              "Responda SOMENTE em JSON: {\"emitente\":\"NOME\",\"numero_nota\":\"NUMERO\",\"cidade\":\"CIDADE\",\"sequencia\":\"x/y\"}")

    progresso = 0
    progress_bar = st.progress(0.0)
    progresso_text = st.empty()

    # agrupamento: key = (numero, emitente) -> dict with list of page bytes and optional parts info
    agrup = {}

    resultados = []  # lines for table

    start_time_total = time.time()
    for a in arquivos:
        name = a["name"]
        try:
            reader = PdfReader(io.BytesIO(a["bytes"]))
        except Exception:
            st.warning(f"Não foi possível ler {name}, pulando.")
            continue

        for idx, page in enumerate(reader.pages):
            # write page to bytes
            b = io.BytesIO()
            w = PdfWriter()
            w.add_page(page)
            w.write(b)
            b.seek(0)

            dados, ok, tempo = chamar_gemini_retry(model, prompt, b)
            seq = None
            numero_raw = ""
            emitente_raw = ""
            cidade_raw = ""

            if ok and "error" not in dados:
                emitente_raw = dados.get("emitente", "") or ""
                numero_raw = dados.get("numero_nota", "") or ""
                cidade_raw = dados.get("cidade", "") or ""
                seq_field = dados.get("sequencia") or dados.get("sequencia_nota") or ""
                seq = parse_sequencia(seq_field) or parse_sequencia(numero_raw) or None
                numero = limpar_numero(numero_raw)
                nome_mapeado = substituir_nome_emitente(emitente_raw, cidade_raw)
                emitente_limpo = limpar_emitente(nome_mapeado)
                key = (numero, emitente_limpo)
                if key not in agrup:
                    agrup[key] = {"pages": [], "parts": set()}
                agrup[key]["pages"].append(b.getvalue())
                if seq:
                    agrup[key]["parts"].add(seq)
                status = "✅ OK"
            else:
                numero = "-"
                emitente_limpo = "-"
                status = f"❌ {dados.get('error','erro')}"

            progresso += 1
            progress_bar.progress(min(progresso / total_paginas, 1.0))
            progresso_text.text(f"{name} pág {idx+1} — {status} ({tempo:.2f}s)")

            resultados.append({
                "arquivo_origem": name,
                "pagina": idx+1,
                "emitente_detectado": emitente_raw if ok else "-",
                "numero_detectado": numero_raw if ok else "-",
                "sequencia": f"{seq[0]}/{seq[1]}" if seq else "-",
                "status": status,
                "tempo_s": round(tempo,2)
            })

    # build final files from agrup
    arquivos_finais = []
    for (numero, emitente), info in agrup.items():
        if not numero or numero == "0":
            # skip invalid
            continue
        writer = PdfWriter()
        for page_bytes in info["pages"]:
            try:
                r = PdfReader(io.BytesIO(page_bytes))
                for p in r.pages:
                    writer.add_page(p)
            except Exception:
                # ignore page if can't be read
                continue
        nome_final = f"DOC {numero}_{emitente}.pdf"
        path_final = session_folder / nome_final
        with open(path_final, "wb") as f_out:
            writer.write(f_out)
        arquivos_finais.append({
            "Novo Nome": nome_final,
            "Emitente": emitente,
            "Número": numero,
            "Total Páginas": len(info["pages"]),
            "Sequências Detectadas": len(info["parts"]) if info["parts"] else 0
        })

    # Show summary and allow manual edits
    st.success(f"Processamento terminado em {round(time.time() - start_time_total,2)}s — {len(arquivos_finais)} notas geradas.")
    st.subheader("🔎 Resultados — verifique e edite os nomes finais antes de baixar")
    if arquivos_finais:
        df_edit = st.data_editor(arquivos_finais, num_rows="dynamic", use_container_width=True, key="editor_final")
        # prepare zip using edited names
        if st.button("📦 Gerar ZIP com nomes editados"):
            memory_zip = io.BytesIO()
            with zipfile.ZipFile(memory_zip, "w") as zf:
                for row in df_edit:
                    nome = row.get("Novo Nome")
                    src = session_folder / row.get("Novo Nome")
                    # if user changed the name, find the original file by matching number+emitente fallback
                    if not src.exists():
                        # attempt match by number+emitente
                        numero = row.get("Número")
                        emitente = row.get("Emitente")
                        candidate = session_folder / f"DOC {numero}_{emitente}.pdf"
                        if candidate.exists():
                            src = candidate
                        else:
                            st.warning(f"Arquivo {nome} não encontrado — pulando.")
                            continue
                    zf.write(src, arcname=nome)
            memory_zip.seek(0)
            st.download_button("⬇️ Baixar ZIP final", data=memory_zip, file_name="notas_processadas.zip", mime="application/zip")
    else:
        st.info("Nenhuma nota processada.")

    # optional: cleanup session folder after download manually or here
    # shutil.rmtree(session_folder)  # uncomment if you want auto-cleanup
    st.dataframe(resultados)
