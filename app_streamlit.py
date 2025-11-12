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

TEMP_FOLDER = Path("./temp")
os.makedirs(TEMP_FOLDER, exist_ok=True)

MAX_TOTAL_PAGES = int(os.getenv("MAX_TOTAL_PAGES", "50"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
MIN_RETRY_DELAY = int(os.getenv("MIN_RETRY_DELAY", "5"))
MAX_RETRY_DELAY = int(os.getenv("MAX_RETRY_DELAY", "30"))
REQUEST_DELAY = int(os.getenv("REQUEST_DELAY", "2"))
MODEL_NAME = os.getenv("MODEL_NAME", "models/gemini-2.0-flash")

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ Chave GOOGLE_API_KEY não encontrada.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

st.title("🧠 Automatizador de Notas Fiscais PDF")
st.markdown("<div class='muted'>Conectando ao modelo...</div>", unsafe_allow_html=True)
try:
    _ = model.name
    st.success("✅ Google Gemini configurado.")
except Exception:
    st.warning("⚠️ Problema ao conectar com Gemini — verifique a variável de ambiente GOOGLE_API_KEY.")

# =====================================================================
# CSS CORPORATIVO
# =====================================================================
st.markdown("""
<style>
body { background-color: #f8f9fa; color: #212529; font-family: 'Segoe UI', Roboto, Arial, sans-serif; }
[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e9ecef; }
h1,h2,h3,h4 { color: #0f4c81; }
div.stButton > button { background-color: #0f4c81; color: white; border-radius: 8px; border: none; font-weight: 500; }
div.stButton > button:hover { background-color: #0b3a5a; }
.stProgress > div > div > div > div { background-color: #28a745 !important; }
.success-log { color:#155724;background-color:#d4edda;padding:6px 10px;border-radius:6px; }
.warning-log { color:#856404;background-color:#fff3cd;padding:6px 10px;border-radius:6px; }
.error-log { color:#721c24;background-color:#f8d7da;padding:6px 10px;border-radius:6px; }
.top-actions { display:flex; gap:10px; align-items:center; }
.block-container { padding-top:2rem; }
.small-note { font-size:13px;color:#6b7280; }
.card { background:#fff;padding:12px;border-radius:8px;box-shadow:0 6px 18px rgba(15,76,129,0.04); margin-bottom:12px; }
</style>
""", unsafe_allow_html=True)

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
    if not s: return ""
    s = unicodedata.normalize("NFKD", s).encode("ASCII","ignore").decode("ASCII")
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
    nome = unicodedata.normalize("NFKD", nome).encode("ASCII","ignore").decode("ASCII")
    nome = re.sub(r"[^A-Z0-9_]+", "_", nome.upper())
    return re.sub(r"_+", "_", nome).strip("_")

def limpar_numero(numero: str) -> str:
    if not numero: return "0"
    numero = re.sub(r"[^\d]", "", str(numero))
    return numero.lstrip("0") or "0"

# =====================================================================
# FUNÇÕES GEMINI + FALLBACK
# =====================================================================
def calcular_delay(tentativa, error_msg):
    if "retry in" in error_msg.lower():
        try:
            return min(float(re.search(r"retry in (\d+\.?\d*)s", error_msg.lower()).group(1)) + 2, MAX_RETRY_DELAY)
        except: pass
    return min(MIN_RETRY_DELAY * (tentativa + 1), MAX_RETRY_DELAY)

# Simulação de fallback (substitua pelas chamadas reais das APIs)
def chamar_deepseek(page_stream, prompt):
    time.sleep(1)
    return {"emitente":"DEEPSEEK_FAKE","numero_nota":"123","cidade":"CIDADE_DS"}, 1.0

def chamar_chatgpt(page_stream, prompt):
    time.sleep(1)
    return {"emitente":"CHATGPT_FAKE","numero_nota":"456","cidade":"CIDADE_CG"}, 1.0

def chamar_gemini_retry(model, prompt_instrucao, page_stream):
    for tentativa in range(MAX_RETRIES + 1):
        try:
            start = time.time()
            resp = model.generate_content(
                [prompt_instrucao, {"mime_type": "application/pdf", "data": page_stream.getvalue()}],
                generation_config={"response_mime_type": "application/json", "temperature":0.1},
                request_options={'timeout':120}
            )
            tempo = round(time.time() - start, 2)
            texto = resp.text.strip().lstrip("```json").rstrip("```").strip()
            try:
                dados = json.loads(texto)
                if not isinstance(dados, dict): raise ValueError("Resposta não é JSON válido")
                return dados, True, tempo
            except Exception:
                if tentativa < MAX_RETRIES:
                    st.warning(f"⚠️ Resposta da IA não era JSON (tentativa {tentativa+1}). Retentando...")
                    time.sleep(MIN_RETRY_DELAY)
                    continue
                else:
                    break
        except ResourceExhausted as e:
            delay = calcular_delay(tentativa, str(e))
            st.warning(f"⚠️ Quota excedida Gemini (tentativa {tentativa+1}/{MAX_RETRIES}). Aguardando {int(delay)}s...")
            time.sleep(delay)
        except Exception as e:
            if tentativa < MAX_RETRIES:
                time.sleep(MIN_RETRY_DELAY)
            else:
                break

    # fallback Deepseek
    try:
        st.info("🔄 Tentando fallback Deepseek...")
        dados_ds, tempo_ds = chamar_deepseek(page_stream, prompt_instrucao)
        if dados_ds: return dados_ds, True, tempo_ds
    except Exception as e:
        st.warning(f"⚠️ Deepseek falhou: {str(e)[:100]}")

    # fallback ChatGPT
    try:
        st.info("🔄 Tentando fallback ChatGPT...")
        dados_cg, tempo_cg = chamar_chatgpt(page_stream, prompt_instrucao)
        if dados_cg: return dados_cg, True, tempo_cg
    except Exception as e:
        st.warning(f"⚠️ ChatGPT falhou: {str(e)[:100]}")

    st.error("❌ Todos os modelos falharam. Retornando vazio.")
    return {"emitente": "", "numero_nota": "", "cidade": ""}, False, 0

# =====================================================================
# FUNÇÕES DE ARQUIVOS
# =====================================================================
def salvar_pdf(path: Path, pages_bytes: list):
    writer = PdfWriter()
    for pb in pages_bytes:
        try:
            r = PdfReader(io.BytesIO(pb))
            for p in r.pages: writer.add_page(p)
        except Exception: continue
    with open(path, "wb") as f_out: writer.write(f_out)

def criar_zip(files, folder: Path, nomes_map: dict):
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w") as zf:
        for f in files:
            src = folder / f
            if src.exists(): zf.write(src, arcname=nomes_map.get(f,f))
    mem.seek(0)
    return mem

# =====================================================================
# PROCESSAMENTO DE PDF
# =====================================================================
def processar_pdfs(uploaded_files):
    if not uploaded_files: return
    session_id = str(uuid.uuid4())
    session_folder = TEMP_FOLDER / session_id
    os.makedirs(session_folder, exist_ok=True)

    arquivos = []
    for f in uploaded_files:
        try: arquivos.append({"name":f.name, "bytes":f.read()})
        except Exception: st.warning(f"Erro ao ler {f.name}, ignorado.")

    total_paginas = 0
    for a in arquivos:
        try: total_paginas += len(PdfReader(io.BytesIO(a["bytes"])).pages)
        except Exception: st.warning(f"Arquivo inválido: {a['name']}")
    st.info(f"📄 Total de páginas a processar: {total_paginas}")

    agrupados_bytes = {}
    resultados_meta = []
    processed_logs = []
    progresso = 0
    progress_bar = st.progress(0.0)
    progresso_text = st.empty()
    start_all = time.time()

    prompt = "Analise a nota fiscal (DANFE). Extraia emitente, número da nota e cidade. Responda SOMENTE em JSON: {\"emitente\":\"NOME\",\"numero_nota\":\"NUMERO\",\"cidade\":\"CIDADE\"}"

    for a in arquivos:
        name = a["name"]
        try: reader = PdfReader(io.BytesIO(a["bytes"]))
        except Exception:
            processed_logs.append((name, 0, "ERRO_LEITURA"))
            continue
        for idx, page in enumerate(reader.pages):
            if progresso > 0: time.sleep(REQUEST_DELAY)
            b = io.BytesIO(); w = PdfWriter(); w.add_page(page); w.write(b); b.seek(0)

            dados, ok, tempo = chamar_gemini_retry(model, prompt, b)
            page_label = f"{name} (pág {idx+1})"
            if not ok or "error" in dados:
                processed_logs.append((page_label, tempo, "ERRO_IA", dados.get("error","")))
                resultados_meta.append({"arquivo_origem":name,"pagina":idx+1,"emitente_detectado":dados.get("emitente","-"),"numero_detectado":dados.get("numero_nota","-"),"status":"ERRO"})
                progresso +=1
                progress_bar.progress(min(progresso/total_paginas,1.0))
                progresso_text.markdown(f"<span class='log-warn'>⚠️ {page_label} — ERRO IA</span>", unsafe_allow_html=True)
                continue

            numero = limpar_numero(dados.get("numero_nota",""))
            emitente = limpar_emitente(substituir_nome_emitente(dados.get("emitente",""), dados.get("cidade","")))
            key = (numero, emitente)
            agrupados_bytes.setdefault(key, []).append(b.getvalue())

            processed_logs.append((page_label, tempo, "OK", f"{numero} / {emitente}"))
            resultados_meta.append({"arquivo_origem":name,"pagina":idx+1,"emitente_detectado":dados.get("emitente",""),"numero_detectado":dados.get("numero_nota",""),"status":"OK","tempo_s":round(tempo,2)})

            progresso +=1
            progress_bar.progress(min(progresso/total_paginas,1.0))
            progresso_text.markdown(f"<span class='log-ok'>✅ {page_label} — OK ({tempo:.2f}s)</span>", unsafe_allow_html=True)

    resultados, files_meta, nomes_map = [], {}, {}
    for (numero, emitente), pages_bytes in agrupados_bytes.items():
        if not numero or numero=="0": continue
        nome_pdf = f"DOC {numero}_{emitente}.pdf"
        caminho = session_folder / nome_pdf
        salvar_pdf(caminho, pages_bytes)
        resultados.append({"file":nome_pdf,"numero":numero,"emitente":emitente,"pages":len(pages_bytes)})
        files_meta[nome_pdf] = {"numero":numero,"emitente":emitente,"pages":len(pages_bytes)}
        nomes_map[nome_pdf] = nome_pdf

    # Salvar na sessão
    st.session_state["resultados"] = resultados
    st.session_state["session_folder"] = str(session_folder)
    st.session_state["novos_nomes"] = nomes_map
    st.session_state["processed_logs"] = processed_logs
    st.session_state["files_meta"] = files_meta

    # Mostra tabela de logs
    st.subheader("📋 Resumo de processamento")
    if processed_logs:
        st.table([{"Página":l[0],"Tempo(s)":l[1],"Status":l[2],"Detalhes":l[3]} for l in processed_logs])

    # Botão download ZIP
    if resultados:
        zip_mem = criar_zip([r["file"] for r in resultados], session_folder, nomes_map)
        st.download_button("📥 Baixar todos PDFs (ZIP)", data=zip_mem, file_name=f"Notas_{uuid.uuid4().hex}.zip", mime="application/zip")

    st.success(f"✅ Processamento concluído em {round(time.time()-start_all,2)}s — {len(resultados)} arquivos gerados.")
    st.rerun()

# =====================================================================
# UPLOAD E BOTÕES DE AÇÃO
# =====================================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 📎 Enviar PDFs e processar (uma vez)")
uploaded_files = st.file_uploader("Selecione arquivos PDF", type=["pdf"], accept_multiple_files=True, key="uploader")
col_up_a, col_up_b = st.columns([1,1])
with col_up_a: process_btn = st.button("🚀 Processar PDFs")
with col_up_b: clear_session = st.button("♻️ Limpar sessão (apagar temporários)")
st.markdown("</div>", unsafe_allow_html=True)

if clear_session:
    if "session_folder" in st.session_state:
        shutil.rmtree(st.session_state["session_folder"], ignore_errors=True)
    for k in ["resultados","session_folder","novos_nomes","processed_logs","files_meta","selected_files"]:
        st.session_state.pop(k, None)
    st.success("Sessão limpa.")
    st.experimental_rerun()

if uploaded_files and process_btn:
    processar_pdfs(uploaded_files)
