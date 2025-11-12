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
st.set_page_config(page_title="Automatizador de Notas Fiscais", page_icon="🧾", layout="wide")

# CSS corporativo claro
st.markdown("""
<style>
body { background-color: #f8f9fa; color: #212529; font-family: 'Segoe UI', Roboto, Arial, sans-serif; }
[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e9ecef; }
h1,h2,h3,h4 { color: #0f4c81; }
div.stButton > button { background-color: #0f4c81; color: white; border-radius: 8px; border: none; font-weight: 500; }
div.stButton > button:hover { background-color: #0b3a5a; }
.stProgress > div > div > div > div { background-color: #28a745 !important; }
.success-log { color: #155724; background-color: #d4edda; padding: 6px 10px; border-radius: 6px; }
.warning-log { color: #856404; background-color: #fff3cd; padding: 6px 10px; border-radius: 6px; }
.error-log { color: #721c24; background-color: #f8d7da; padding: 6px 10px; border-radius: 6px; }
.block-container { padding-top: 2rem; }
.small-note { font-size:13px; color:#6b7280; }
.card { background: #fff; padding: 12px; border-radius:8px; box-shadow: 0 6px 18px rgba(15,76,129,0.04); margin-bottom:12px; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Automatizador de Notas Fiscais PDF")

# Configs básicas
TEMP_FOLDER = Path("./temp")
os.makedirs(TEMP_FOLDER, exist_ok=True)
MAX_TOTAL_PAGES = int(os.getenv("MAX_TOTAL_PAGES", "50"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
MIN_RETRY_DELAY = int(os.getenv("MIN_RETRY_DELAY", "5"))
MAX_RETRY_DELAY = int(os.getenv("MAX_RETRY_DELAY", "30"))
MODEL_NAME = os.getenv("MODEL_NAME", "models/gemini-2.5-flash")

# Configura Gemini
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ Chave GOOGLE_API_KEY não encontrada.")
    st.stop()
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

try:
    _ = model.name
    st.success("✅ Google Gemini configurado.")
except Exception:
    st.warning("⚠️ Problema ao conectar com Gemini — verifique a variável de ambiente GOOGLE_API_KEY.")

# =====================================================================
# Normalização e substituições
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
    if not s: return ""
    s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII")
    s = re.sub(r"[^A-Z0-9 ]+", " ", s.upper())
    return re.sub(r"\s+", " ", s).strip()

def substituir_nome_emitente(nome_raw: str, cidade_raw: str = None) -> str:
    nome_norm = _normalizar_texto(nome_raw)
    cidade_norm = _normalizar_texto(cidade_raw) if cidade_raw else None
    if "SABARA" in nome_norm:
        return f"SB_{cidade_norm.split()[0]}" if cidade_norm else "SB"
    for padrao, substituto in SUBSTITUICOES_FIXAS.items():
        if _normalizar_texto(padrao) in nome_norm:
            return substituto
    return re.sub(r"\s+", "_", nome_norm)

def limpar_emitente(nome: str) -> str:
    if not nome: return "SEM_NOME"
    nome = unicodedata.normalize("NFKD", nome).encode("ASCII", "ignore").decode("ASCII")
    nome = re.sub(r"[^A-Z0-9_]+", "_", nome.upper())
    return re.sub(r"_+", "_", nome).strip("_")

def limpar_numero(numero: str) -> str:
    if not numero: return "0"
    numero = re.sub(r"[^\d]", "", str(numero))
    return numero.lstrip("0") or "0"

# =====================================================================
# Retry Gemini
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
            try:
                dados = json.loads(texto)
            except Exception:
                dados = {"error": "Resposta da IA não era JSON", "_raw": texto}
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
# Upload e Processamento
# =====================================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 📎 Enviar PDFs e processar (uma vez)")
uploaded_files = st.file_uploader(
    "Selecione arquivos PDF",
    type=["pdf"],
    accept_multiple_files=True,
    key="uploader"
)
col_up_a, col_up_b = st.columns([1,1])
with col_up_a:
    process_btn = st.button("🚀 Processar PDFs")
with col_up_b:
    clear_session = st.button("♻️ Limpar sessão (apagar temporários)")
st.markdown("</div>", unsafe_allow_html=True)

# Limpar sessão
if clear_session:
    if "session_folder" in st.session_state:
        try:
            shutil.rmtree(st.session_state["session_folder"])
        except Exception:
            pass
    for k in ["resultados", "session_folder", "novos_nomes", "processed_logs", "files_meta", "selected_files", "_manage_target"]:
        st.session_state.pop(k, None)
    st.success("Sessão limpa.")
    st.stop()

# Inicialização de session_state
for key in ["resultados", "processed_logs", "files_meta", "novos_nomes", "selected_files"]:
    if key not in st.session_state:
        st.session_state[key] = []

# Processamento
if uploaded_files and process_btn:
    session_id = str(uuid.uuid4())
    session_folder = TEMP_FOLDER / session_id
    os.makedirs(session_folder, exist_ok=True)
    st.session_state["session_folder"] = str(session_folder)

    arquivos = [{"name": f.name, "bytes": f.read()} for f in uploaded_files]

    total_paginas = sum(len(PdfReader(io.BytesIO(a["bytes"])).pages) for a in arquivos)
    st.info(f"📄 Total de páginas a processar: {total_paginas}")

    agrupados_bytes = {}
    progresso = 0
    progress_bar = st.progress(0.0)
    progresso_text = st.empty()
    start_all = time.time()

    prompt = (
        "Analise a nota fiscal (DANFE). Extraia emitente, número da nota e cidade. "
        "Responda SOMENTE em JSON: {\"emitente\":\"NOME\",\"numero_nota\":\"NUMERO\",\"cidade\":\"CIDADE\"}"
    )

    for a in arquivos:
        name = a["name"]
        try:
            reader = PdfReader(io.BytesIO(a["bytes"]))
        except Exception:
            st.session_state["processed_logs"].append((name, 0, "ERRO_LEITURA"))
            continue

        for idx, page in enumerate(reader.pages):
            b = io.BytesIO()
            w = PdfWriter()
            w.add_page(page)
            w.write(b)
            b.seek(0)

            dados, ok, tempo = chamar_gemini_retry(model, prompt, b)
            page_label = f"{name} (pág {idx+1})"

            if not ok or "error" in dados:
                st.session_state["processed_logs"].append((page_label, tempo, "ERRO_IA", dados.get("error", str(dados))))
                progresso += 1
                progress_bar.progress(min(progresso/total_paginas, 1.0))
                progresso_text.markdown(f"<span class='warning-log'>⚠️ {page_label} — ERRO IA</span>", unsafe_allow_html=True)
                st.session_state["resultados"].append({
                    "arquivo_origem": name,
                    "pagina": idx+1,
                    "emitente_detectado": dados.get("emitente") if isinstance(dados, dict) else "-",
                    "numero_detectado": dados.get("numero_nota") if isinstance(dados, dict) else "-",
                    "status": "ERRO"
                })
                continue

            emitente_raw = dados.get("emitente", "") or ""
            numero_raw = dados.get("numero_nota", "") or ""
            cidade_raw = dados.get("cidade", "") or ""
            numero = limpar_numero(numero_raw)
            nome_map = substituir_nome_emitente(emitente_raw, cidade_raw)
            emitente = limpar_emitente(nome_map)

            key = (numero, emitente)
            agrupados_bytes.setdefault(key, []).append(b.getvalue())

            st.session_state["processed_logs"].append((page_label, tempo, "OK", f"{numero} / {emitente}"))
            st.session_state["resultados"].append({
                "arquivo_origem": name,
                "pagina": idx+1,
                "emitente_detectado": emitente_raw,
                "numero_detectado": numero_raw,
                "status": "OK",
                "tempo_s": round(tempo, 2)
            })

            progresso += 1
            progress_bar.progress(min(progresso/total_paginas, 1.0))
            progresso_text.markdown(f"<span class='success-log'>✅ {page_label} — OK ({tempo:.2f}s)</span>", unsafe_allow_html=True)

    # gerar PDFs finais
    for (numero, emitente), pages_bytes in agrupados_bytes.items():
        if not numero or numero == "0": continue
        writer = PdfWriter()
        for pb in pages_bytes:
            r = PdfReader(io.BytesIO(pb))
            for p in r.pages:
                writer.add_page(p)
        nome_pdf = f"DOC {numero}_{emitente}.pdf"
        caminho = session_folder / nome_pdf
        with open(caminho, "wb") as f_out:
            writer.write(f_out)
        st.session_state["files_meta"][nome_pdf] = {"numero": numero, "emitente": emitente, "pages": len(pages_bytes)}
        st.session_state["novos_nomes"][nome_pdf] = nome_pdf

    st.success(f"✅ Processamento concluído em {round(time.time() - start_all, 2)}s — {len(agrupados_bytes)} arquivos gerados.")
