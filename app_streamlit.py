# turbo_v5_corrected_sections.py
# Versão: Turbo v5 — Corrigida (seções e indentação)

# =====================================================================
# PARTE 1/6 — IMPORTS E CONFIGURAÇÃO INICIAL
# =====================================================================
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
import fitz
import base64
import openai
from pathlib import Path
from PyPDF2 import PdfReader, PdfWriter
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
import streamlit as st
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# =====================================================================
# CONFIGURAÇÃO INICIAL
# =====================================================================
load_dotenv()
st.set_page_config(page_title="Automatizador de Notas Fiscais", page_icon="icone.ico")

# ======= CSS Corporativo Claro =======
st.markdown("""
<style>
body {
  background-color: #f8f9fa;
  color: #212529;
  font-family: 'Segoe UI', Roboto, Arial, sans-serif;
}
[data-testid="stSidebar"] {
  background-color: #ffffff;
  border-right: 1px solid #e9ecef;
}
h1, h2, h3, h4 {
  color: #0f4c81;
}
div.stButton > button {
  background-color: #0f4c81;
  color: white;
  border-radius: 8px;
  border: none;
  font-weight: 500;
}
div.stButton > button:hover {
  background-color: #0b3a5a;
}
.stProgress > div > div > div > div {
  background-color: #28a745 !important;
}
.success-log {
  color: #155724;
  background-color: #d4edda;
  padding: 6px 10px;
  border-radius: 6px;
}
.warning-log {
  color: #856404;
  background-color: #fff3cd;
  padding: 6px 10px;
  border-radius: 6px;
}
.error-log {
  color: #721c24;
  background-color: #f8d7da;
  padding: 6px 10px;
  border-radius: 6px;
}
.top-actions {
  display: flex;
  gap: 10px;
  align-items: center;
}
.block-container {
  padding-top: 2rem;
}
.small-note {
  font-size:13px;
  color:#6b7280;
}
.card { background: #fff; padding: 12px; border-radius:8px; box-shadow: 0 6px 18px rgba(15,76,129,0.04); margin-bottom:12px; }
.metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 10px; }
.manage-panel { background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #0f4c81; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

st.title("Automatizador de Notas Fiscais PDF")

# =====================================================================
# PARTE 2/6 — CACHE INTELIGENTE E VARIÁVEIS GLOBAIS
# =====================================================================
class DocumentCache:
    def __init__(self, cache_dir="./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def get_cache_key(self, pdf_bytes, prompt):
        content_hash = hashlib.md5(pdf_bytes).hexdigest()
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        return f"{content_hash}_{prompt_hash}"

    def _cache_path(self, key):
        safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", key)[:240]
        return self.cache_dir / f"{safe}.pkl"

    def get(self, key, ttl_seconds: int = 3600):
        cache_file = self._cache_path(key)
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    obj = pickle.load(f)
                ts = obj.get("ts", 0)
                if ttl_seconds <= 0 or (time.time() - ts) < ttl_seconds:
                    return obj.get("data")
                else:
                    try:
                        cache_file.unlink()
                    except:
                        pass
                    return None
            except Exception:
                return None
        return None

    def set(self, key, data, ttl_seconds: int = 86400):
        cache_file = self._cache_path(key)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({"data": data, "ts": time.time(), "ttl": ttl_seconds}, f)
        except Exception:
            pass

    def clear(self):
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except:
                pass


document_cache = DocumentCache()

# =====================================================================
# PARTE 3/6 — CONFIG, PADRÕES E NORMALIZAÇÃO
# =====================================================================
TEMP_FOLDER = Path("./temp")
TEMP_FOLDER.mkdir(exist_ok=True)

MAX_TOTAL_PAGES = int(os.getenv("MAX_TOTAL_PAGES", "50"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
MIN_RETRY_DELAY = int(os.getenv("MIN_RETRY_DELAY", "5"))
MAX_RETRY_DELAY = int(os.getenv("MAX_RETRY_DELAY", "30"))

PATTERNS_FILE = "patterns.json"

def load_patterns():
    default_patterns = {
        "COMPANHIA DE AGUA E ESGOTOS DA PARAIBA": "CAGEPA",
        "COMPANHIA DE AGUA E ESGOTOS DA PARAÍBA": "CAGEPA",
        "CIA DE AGUA E ESGOTO DO CEARA": "CAGECE",
        "COMPANHIA DE AGUAS E ESGOTOS DO RN": "CAERN",
        "PETRÓLEO BRASILEIRO S.A": "PETROBRAS",
        "PETROLEO BRASILEIRO S.A": "PETROBRAS",
        "NEOENERGIA": "NEOENERGIA",
        "EQUATORIAL": "EQUATORIAL"
    }
    if not os.path.exists(PATTERNS_FILE):
        save_patterns(default_patterns)
        return default_patterns
    try:
        with open(PATTERNS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default_patterns

def save_patterns(patterns):
    try:
        with open(PATTERNS_FILE, 'w', encoding='utf-8') as f:
            json.dump(patterns, f, ensure_ascii=False, indent=4)
        return True
    except Exception:
        return False

SUBSTITUICOES_FIXAS = load_patterns()


def _normalizar_texto(s: str) -> str:
    if not s:
        return ""
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
    if not nome:
        return "SEM_NOME"
    nome = unicodedata.normalize("NFKD", nome).encode("ASCII", "ignore").decode("ASCII")
    nome = re.sub(r"[^A-Z0-9_]+", "_", nome.upper())
    return re.sub(r"_+", "_", nome).strip("_")


def limpar_numero(numero: str) -> str:
    if not numero:
        return "0"
    numero = re.sub(r"[^\d]", "", str(numero))
    return numero.lstrip("0") or "0"


def validar_e_corrigir_dados(dados):
    if not isinstance(dados, dict):
        return {"emitente": "NÃO_IDENTIFICADO", "numero_nota": "000000", "cidade": "NÃO_IDENTIFICADO"}
    required_fields = ['emitente', 'numero_nota', 'cidade']
    for field in required_fields:
        if field not in dados or not dados[field]:
            dados[field] = "NÃO_IDENTIFICADO"
    if 'numero_nota' in dados:
        numero_limpo = re.sub(r"[^\d]", "", str(dados['numero_nota']))
        dados['numero_nota'] = numero_limpo if numero_limpo else "000000"
    return dados

# =====================================================================
# PARTE 4/6 — GEMINI / OPENAI CLIENTS E FUNÇÕES DE PROCESSAMENTO
# =====================================================================
# Gemini
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ Chave GOOGLE_API_KEY não encontrada.")
    st.stop()
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(os.getenv("MODEL_NAME", "models/gemini-2.5-flash"))
    st.sidebar.success("✅ Gemini configurado")
except Exception as e:
    st.error(f"❌ Erro ao configurar Gemini: {str(e)}")
    st.stop()

# OpenAI client (opcional)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = None
if OPENAI_API_KEY:
    try:
        openai.api_key = OPENAI_API_KEY
        openai_client = openai
    except Exception:
        openai_client = None


def extrair_json_seguro(texto: str):
    texto = (texto or "").strip()
    match_array = re.search(r"\[.*\]", texto, re.DOTALL)
    if match_array:
        try:
            return json.loads(match_array.group())
        except Exception:
            pass
    match_obj = re.search(r"\{.*\}", texto, re.DOTALL)
    if match_obj:
        try:
            return [json.loads(match_obj.group())]
        except Exception:
            pass
    clean = texto.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean)
    except Exception:
        return []


def calcular_delay(tentativa, error_msg):
    try:
        em = (error_msg or "").lower()
    except:
        em = ""
    if "retry in" in em:
        try:
            seg = float(re.search(r"retry in (\d+\.?\d*)s", em).group(1))
            return min(seg + 2, MAX_RETRY_DELAY)
        except:
            pass
    return min(MIN_RETRY_DELAY * (tentativa + 1), MAX_RETRY_DELAY)


def processar_pagina_gemini(prompt_instrucao, page_stream):
    for tentativa in range(MAX_RETRIES + 1):
        try:
            start = time.time()
            resp = model.generate_content(
                [prompt_instrucao, {"mime_type": "application/pdf", "data": page_stream.getvalue()}],
                generation_config={"response_mime_type": "application/json"},
                request_options={'timeout': 60}
            )
            tempo = round(time.time() - start, 2)
            texto = (resp.text or "").strip()
            if texto.startswith("```"):
                texto = texto.replace("```json", "").replace("```", "").strip()
            try:
                dados = json.loads(texto)
            except Exception:
                dados = {"error": "Resposta não era JSON válido", "_raw": texto[:600]}
            return dados, True, tempo, "Gemini"
        except ResourceExhausted as e:
            delay = calcular_delay(tentativa, str(e))
            time.sleep(delay)
        except Exception as e:
            if tentativa < MAX_RETRIES:
                time.sleep(MIN_RETRY_DELAY)
            else:
                return {"error": str(e)}, False, 0, "Gemini"
    return {"error": "Falha máxima de tentativas"}, False, 0, "Gemini"


def processar_pagina_openai_fallback(prompt_sistema, pdf_bytes):
    if not openai_client:
        return {"error": "OpenAI não configurada (.env)"}, False, 0, "OpenAI"
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if len(doc) < 1:
            return {"error": "PDF vazio"}, False, 0, "OpenAI"
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_bytes = pix.tobytes("png")
        base64_image = base64.b64encode(img_bytes).decode('utf-8')
        # Enviar imagem como texto (data URL). Implementação pode variar conforme sua conta OpenAI
        messages = [
            {"role": "system", "content": "Você é um assistente OCR. Extraia emitente, numero_nota e cidade e retorne JSON."},
            {"role": "user", "content": prompt_sistema + "\nIMAGEM: data:image/png;base64," + base64_image}
        ]
        resp = openai_client.ChatCompletion.create(model="gpt-4o", messages=messages, temperature=0)
        conteudo = resp.choices[0].message['content'] if 'choices' in resp and resp.choices else resp['choices'][0]['message']['content']
        try:
            dados = json.loads(conteudo)
            return dados, True, 0, "GPT-Vision"
        except Exception:
            return {"error": "JSON Inválido OpenAI", "_raw": conteudo[:600]}, False, 0, "GPT-Vision"
    except Exception as e:
        return {"error": f"Erro Vision: {str(e)}"}, False, 0, "GPT-Vision"

# =====================================================================
# PARTE 5/6 — WORKER E LOOP PRINCIPAL
# =====================================================================

def processar_pagina_worker(job_data):
    pdf_bytes = job_data["bytes"]
    prompt = job_data["prompt"]
    name = job_data["name"]
    page_idx = job_data["page_idx"]
    mode = job_data.get("mode", "GEMINI")
    use_cache = job_data.get("use_cache", True)

    cache_key = document_cache.get_cache_key(pdf_bytes, prompt)
    if use_cache:
        cached_result = document_cache.get(cache_key, ttl_seconds=86400)
        if cached_result:
            # cached_result expected to be dict with 'dados','tempo','provider'
            return {"status": "CACHE", "dados": cached_result.get('dados'), "tempo": cached_result.get('tempo', 0), "provider": cached_result.get('provider','CACHE'), "name": name, "page_idx": page_idx, "pdf_bytes": pdf_bytes}

    if mode == "OPENAI":
        dados, ok, tempo, provider = processar_pagina_openai_fallback(prompt, pdf_bytes)
    else:
        page_stream = io.BytesIO(pdf_bytes)
        dados, ok, tempo, provider = processar_pagina_gemini(prompt, page_stream)
        if (not ok or 'error' in dados) and openai_client:
            # Retry with OpenAI Vision as fallback
            dados2, ok2, tempo2, prov2 = processar_pagina_openai_fallback(prompt, pdf_bytes)
            if ok2 and 'error' not in dados2:
                dados, ok, tempo, provider = dados2, ok2, tempo2, prov2

    if ok and isinstance(dados, dict) and 'error' not in dados:
        document_cache.set(cache_key, {'dados': dados, 'tempo': tempo, 'provider': provider}, ttl_seconds=86400)
        return {"status": "OK", "dados": dados, "tempo": tempo, "provider": provider, "name": name, "page_idx": page_idx, "pdf_bytes": pdf_bytes}
    else:
        err = dados.get('error') if isinstance(dados, dict) else str(dados)
        return {"status": "ERRO", "dados": dados, "tempo": tempo, "provider": provider, "name": name, "page_idx": page_idx, "error_msg": err, "pdf_bytes": pdf_bytes}

# =====================================================================
# PARTE 6/6 — UI, SIDEBAR, LOOP DE PROCESSAMENTO E GERAÇÃO DE ARQUIVOS
# =====================================================================
# Sidebar: provider selection and cache/workers
with st.sidebar:
    st.header("⚙️ Configuração")
    provider_label = st.radio("Modelo Principal", ["Gemini (Google)", "GPT-4o (OpenAI)"], index=0)
    mode_sel = "GEMINI" if "Gemini" in provider_label else "OPENAI"
    st.markdown("---")
    use_cache = st.checkbox("Usar Cache", value=True, key="use_cache")
    workers = st.slider("Workers", 1, 8, 4)
    if st.button("🔄 Limpar Cache"):
        document_cache.clear()
        st.success("Cache limpo!")

# UPLOAD UI
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 📎 Enviar PDFs e processar")
uploaded_files = st.file_uploader("Selecione arquivos PDF", type=["pdf"], accept_multiple_files=True, key="uploader")
col_up_a, col_up_b = st.columns([1,1])
with col_up_a:
    process_btn = st.button("🚀 Processar PDFs")
with col_up_b:
    clear_session = st.button("♻️ Limpar sessão")
st.markdown("</div>", unsafe_allow_html=True)

if clear_session:
    if "session_folder" in st.session_state:
        try:
            shutil.rmtree(st.session_state["session_folder"])
        except Exception:
            pass
    for k in ["resultados", "session_folder", "novos_nomes", "processed_logs", "files_meta", "selected_files", "_manage_target"]:
        st.session_state.pop(k, None)
    st.success("Sessão limpa.")
    st.rerun()

if uploaded_files and process_btn:
    session_id = str(uuid.uuid4())
    session_folder = TEMP_FOLDER / session_id
    os.makedirs(session_folder, exist_ok=True)

    arquivos = []
    for f in uploaded_files:
        try:
            b = f.read()
            arquivos.append({"name": f.name, "bytes": b})
        except Exception:
            st.warning(f"Erro ao ler {f.name}, ignorado.")

    total_paginas = 0
    for a in arquivos:
        try:
            r = PdfReader(io.BytesIO(a["bytes"]))
            total_paginas += len(r.pages)
        except Exception:
            st.warning(f"Arquivo inválido: {a['name']}")

    st.info(f"📄 Total de páginas a processar: {total_paginas}")

    # preparar jobs por página
    jobs = []
    for a in arquivos:
        name = a["name"]
        try:
            reader = PdfReader(io.BytesIO(a["bytes"]))
            for idx, page in enumerate(reader.pages):
                b = io.BytesIO()
                w = PdfWriter()
                w.add_page(page)
                w.write(b)
                page_bytes = b.getvalue()
                jobs.append({
                    "bytes": page_bytes,
                    "prompt": (
                        "Analise a nota fiscal (DANFE). Extraia emitente, número da nota e cidade. "
                        "Responda SOMENTE em JSON: {\"emitente\":\"NOME\",\"numero_nota\":\"NUMERO\",\"cidade\":\"CIDADE\"}"
                    ),
                    "name": name,
                    "page_idx": idx,
                    "use_cache": use_cache,
                    "mode": mode_sel
                })
        except Exception as e:
            st.warning(f"Erro ao ler páginas de {name}: {e}")

    if not jobs:
        st.error("Nenhuma página válida encontrada nos PDFs enviados.")
        st.stop()

    # executar em paralelo
    agrupados_bytes = {} # Declarado apenas uma vez agora
    processed_logs = []
    resultados_meta = []
    processed_count = 0
    total_jobs = len(jobs) if jobs else 1
    progress_bar = st.progress(0.0)
    progresso_text = st.empty()

    start_all = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_job = {executor.submit(processar_pagina_worker, job): job for job in jobs}
        
        # CORREÇÃO DE INDENTAÇÃO ABAIXO
        for future in as_completed(future_to_job):
            processed_count += 1
            try:
                result = future.result()
                name = result.get("name")
                idx = result.get("page_idx")
                page_label = f"{name} (pág {idx+1})"
                if result.get("status") == "ERRO":
                    processed_logs.append((page_label, result.get("tempo",0), "ERRO_IA", result.get("error_msg","erro"), result.get("provider","")))
                    progresso_text.markdown(f"<span class='warning-log'>⚠️ {page_label} — ERRO</span>", unsafe_allow_html=True)
                else:
                    dados = result.get("dados", {})
                    tempo = result.get("tempo", 0)
                    provider_used = result.get("provider", "")
                    dados = validar_e_corrigir_dados(dados)
                    emitente_raw = dados.get("emitente","")
                    numero_raw = dados.get("numero_nota","")
                    cidade_raw = dados.get("cidade","")
                    numero = limpar_numero(numero_raw)
                    nome_map = substituir_nome_emitente(emitente_raw, cidade_raw)
                    emitente = limpar_emitente(nome_map)
                    key = (numero, emitente)
                    
                    # Nota: agrupados_bytes é thread-safe aqui pois estamos na thread principal consumindo o iterator
                    agrupados_bytes.setdefault(key, []).append(result.get("pdf_bytes"))
                    
                    status_lbl = result.get("status")
                    processed_logs.append((page_label, tempo, status_lbl, f"{numero} / {emitente}", provider_used))
                    resultados_meta.append({"arquivo_origem": name, "pagina": idx+1, "emitente_detectado": emitente_raw, "numero_detectado": numero_raw, "status": status_lbl, "tempo_s": round(tempo,2), "provider": provider_used})
                    progresso_text.markdown(f"<span class='success-log'>✅ {page_label} — {status_lbl} ({tempo:.2f}s)</span>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Erro crítico no worker: {e}")
            
            progress_bar.progress(min(processed_count/total_jobs, 1.0))

    # gerar arquivos finais
    resultados = []
    files_meta = {}
    for (numero, emitente), pages_bytes in agrupados_bytes.items():
        if not numero or numero == "0":
            continue
        writer = PdfWriter()
        count = 0
        for pb in pages_bytes:
            try:
                r = PdfReader(io.BytesIO(pb))
                for p in r.pages:
                    writer.add_page(p)
                    count += 1
            except Exception:
                continue
        if count == 0:
            continue
        nome_pdf = f"DOC {numero}_{emitente}.pdf"
        caminho = session_folder / nome_pdf
        with open(caminho, "wb") as f_out:
            writer.write(f_out)
        resultados.append({"file": nome_pdf, "numero": numero, "emitente": emitente, "pages": count})
        files_meta[nome_pdf] = {"numero": numero, "emitente": emitente, "pages": count}

    st.session_state["resultados"] = resultados
    st.session_state["session_folder"] = str(session_folder)
    st.session_state["novos_nomes"] = {r["file"]: r["file"] for r in resultados}
    st.session_state["processed_logs"] = processed_logs
    st.session_state["files_meta"] = files_meta

    st.success(f"✅ Processamento concluído em {round(time.time() - start_all, 2)}s — {len(resultados)} arquivos gerados.")
    st.rerun()
# FIM