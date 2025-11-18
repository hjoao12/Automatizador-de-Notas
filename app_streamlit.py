# arquivo otimizado modo B - turbofast (substitua o seu arquivo por este)
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
st.set_page_config(
    page_title="Automatizador de Notas Fiscais",
    page_icon="icone.ico"
)

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

st.title("Automatizador de Notas Fiscais PDF — Modo B (Turbo)")

# =====================================================================
# CONFIGURAÇÕES GERAIS E ARQUIVOS DE CONFIG
# =====================================================================
TEMP_FOLDER = Path("./temp")
os.makedirs(TEMP_FOLDER, exist_ok=True)

CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)

CONFIG_DIR = Path("./config")
CONFIG_DIR.mkdir(exist_ok=True)
PATTERNS_FILE = CONFIG_DIR / "patterns.json"

# threads para executor (MODIFICAÇÃO)
MAX_WORKERS_DEFAULT = min(6, (os.cpu_count() or 2))

MAX_TOTAL_PAGES = int(os.getenv("MAX_TOTAL_PAGES", "500"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
MIN_RETRY_DELAY = int(os.getenv("MIN_RETRY_DELAY", "5"))
MAX_RETRY_DELAY = int(os.getenv("MAX_RETRY_DELAY", "30"))

# =====================================================================
# SISTEMA DE CACHE INTELIGENTE (ATUALIZADO)
# =====================================================================
class DocumentCache:
    def __init__(self, cache_dir=CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _cache_path(self, key: str):
        safe = re.sub(r'[^a-zA-Z0-9_\-]', '_', key)
        return self.cache_dir / f"{safe}.pkl"

    def get_cache_key_file(self, file_bytes: bytes, prompt: str):
        """Chave por arquivo inteiro + prompt (MODIFICAÇÃO)"""
        content_hash = hashlib.md5(file_bytes).hexdigest()
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        return f"{content_hash}_{prompt_hash}"

    def get(self, key):
        cache_file = self._cache_path(key)
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except (EOFError, pickle.UnpicklingError):
                # cache corrompido -> remover
                try:
                    cache_file.unlink()
                except Exception:
                    pass
                return None
            except Exception as e:
                st.sidebar.error(f"Cache read error: {e}")
                return None
        return None

    def set(self, key, data):
        cache_file = self._cache_path(key)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            st.sidebar.error(f"Cache write error: {e}")

    def clear(self):
        """Limpa todo o cache"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except Exception:
                pass

document_cache = DocumentCache()

# =====================================================================
# PADRÕES DE RENOMEAÇÃO PERSISTENTES (MODIFICAÇÃO)
# - Permite adicionar/editar/excluir padrões via UI do Streamlit
# - Previne conflitos básicos (mesma chave normalize)
# =====================================================================

# Inicializa arquivo de patterns com as fixas se não existir
def load_patterns():
    if not PATTERNS_FILE.exists():
        # salva SUBSTITUICOES_FIXAS abaixo quando definidas
        save_patterns(SUBSTITUICOES_FIXAS)
        return dict(SUBSTITUICOES_FIXAS)
    try:
        with open(PATTERNS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # garantir keys are strings
            return {str(k): str(v) for k, v in data.items()}
    except Exception as e:
        st.sidebar.error(f"Erro ao carregar padrões: {e}")
        return dict(SUBSTITUICOES_FIXAS)

def save_patterns(pats: dict):
    try:
        with open(PATTERNS_FILE, "w", encoding="utf-8") as f:
            json.dump(pats, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.sidebar.error(f"Erro ao salvar padrões: {e}")

# Carregar o padrão fixo inicial como fallback (usado na primeira criação)
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
    "UNIPAR CARBOCLORO": "UNIPAR_CARBOCLORO",
    "UNIPAR CARBOCLORO LTDA": "UNIPAR_CARBOCLORO",
    "UNIPAR_CARBLOCLORO LTDA": "UNIPAR_CARBOCLORO",
    "EXPRESS TCM": "EXPRESS_TCM",
    "EXPRESS TCM LTDA": "EXPRESS_TCM",
    "MDM RENOVADORA DE PNEUS LTDA ME": "MDM_RENOVADORA",
    "MDM RENOVADORA DE PNEUS": "MDM_RENOVADORA",
    "COMPANHIA DE AGUAS E ESGOTOS DO RN": "CAERN",
    "EKIPE TEC DE SEG E INCENDIO LTDA ME": "EKIPE",
    "PETRÓLEO BRASILEIRO S.A": "PETROBRAS",
    "PETROLEO BRASILEIRO S.A": "PETROBRAS",
    "PETRÓLEO BRASILEIRO S A": "PETROBRAS",
    "PETROLEO BRASILEIRO S A": "PETROBRAS",
    "INNOVATIVE WATER CARE IND E COM DE PROD QUIM BRASIL LTDA": "SIGURA",
    "COMERCIAL E IMPORTADORA DE PNEUS": "CAMPNEUS",
    "URP CARGAS E LOGISTICA LTDA": "URP",
    "U.R.P CARGAS & LOGISTICA LTDA": "URP",
    "U.R.P CARGAS E LOGISTICA LTDA": "URP",
    "MF DE MELO FILHO-ME": "MF_DE_MELO",
    "M.F DE MELO FILHO": "MF_DE_MELO",
    "M.F DE MELO FILHO-ME": "MF_DE_MELO",
}

# Carrega padrões persistentes (inclui as FIXAS inicialmente)
PATTERNS = load_patterns()

def normalize_pattern_key(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII")
    s = re.sub(r"[^A-Z0-9 ]+", " ", s.upper())
    return re.sub(r"\s+", " ", s).strip()

def add_pattern(raw_pattern: str, substitute: str) -> (bool, str):
    """Adiciona novo padrão. Retorna (ok, mensagem). Previne conflitos básicos."""
    key_norm = normalize_pattern_key(raw_pattern)
    if not key_norm:
        return False, "Padrão vazio."
    # prevenção de conflito: mesma chave normalizada
    for existing in PATTERNS.keys():
        if normalize_pattern_key(existing) == key_norm:
            return False, "Conflito: padrão já existe (mesma normalização)."
    PATTERNS[raw_pattern] = substitute
    save_patterns(PATTERNS)
    return True, "Padrão adicionado."

def edit_pattern(old_raw: str, new_raw: str, new_sub: str) -> (bool, str):
    if old_raw not in PATTERNS:
        return False, "Padrão não encontrado."
    key_norm = normalize_pattern_key(new_raw)
    for k in PATTERNS.keys():
        if k != old_raw and normalize_pattern_key(k) == key_norm:
            return False, "Conflito: outro padrão com mesma normalização."
    # Aplica edição
    PATTERNS.pop(old_raw)
    PATTERNS[new_raw] = new_sub
    save_patterns(PATTERNS)
    return True, "Padrão editado."

def remove_pattern(raw_pattern: str) -> (bool, str):
    if raw_pattern in PATTERNS:
        PATTERNS.pop(raw_pattern)
        save_patterns(PATTERNS)
        return True, "Removido"
    return False, "Não achou padrão"

# =====================================================================
# NORMALIZAÇÃO E SUBSTITUIÇÕES (USANDO PATTERNS persistente)
# =====================================================================
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
    # usa PATTERNS (persistente)
    for padrao, substituto in PATTERNS.items():
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
    """Valida e corrige dados extraídos da IA"""
    if not isinstance(dados, dict):
        dados = {}

    required_fields = ['emitente', 'numero_nota', 'cidade']

    # Verifica campos obrigatórios
    for field in required_fields:
        if field not in dados or not dados[field]:
            dados[field] = "NÃO_IDENTIFICADO"

    # Correções comuns (pode ser expandido)
    correcoes = {
        'emitente': {
            'CPFL ENERGIA': 'CPFL',
            'COMPANHIA PAULISTA DE FORCA E LUZ': 'CPFL',
            'SABARA': 'SABARA'
        }
    }

    for field, correcoes_field in correcoes.items():
        if field in dados:
            for incorreto, correto in correcoes_field.items():
                if incorreto in dados[field].upper():
                    dados[field] = correto
                    break

    # Validação de número da nota
    if 'numero_nota' in dados:
        numero_limpo = re.sub(r'[^\d]', '', str(dados['numero_nota']))
        dados['numero_nota'] = numero_limpo if numero_limpo else "000000"

    return dados

# =====================================================================
# CONFIGURAÇÃO GEMINI (usar st.secrets quando disponível) - (MODIFICAÇÃO)
# =====================================================================
# Preferir st.secrets (Streamlit Cloud) com fallback para env var
if hasattr(st, "secrets") and st.secrets.get("GOOGLE_API_KEY"):
    GEMINI_API_KEY = st.secrets["GOOGLE_API_KEY"]
else:
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    st.error("❌ Chave GOOGLE_API_KEY não encontrada. Configure st.secrets['GOOGLE_API_KEY'] ou variável de ambiente.")
    st.stop()

try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(os.getenv("MODEL_NAME", "models/gemini-2.5-flash"))
    st.sidebar.success("✅ Gemini configurado")
except Exception as e:
    st.error(f"❌ Erro ao configurar Gemini: {str(e)}")
    st.stop()

# =====================================================================
# PROCESSAMENTO GEMINI (MULTITHREADED POR PÁGINA) - (MODIFICAÇÃO)
# - processa páginas em paralelo com ThreadPoolExecutor
# - usa cache por arquivo para pular todo o processamento se já existir
# =====================================================================
def calcular_delay(tentativa, error_msg):
    if not error_msg:
        return min(MIN_RETRY_DELAY * (tentativa + 1), MAX_RETRY_DELAY)
    if "retry in" in error_msg.lower():
        try:
            return min(float(re.search(r"retry in (\d+\.?\d*)s", error_msg.lower()).group(1)) + 2, MAX_RETRY_DELAY)
        except:
            pass
    return min(MIN_RETRY_DELAY * (tentativa + 1), MAX_RETRY_DELAY)

def processar_pagina_gemini_single(prompt_instrucao: str, page_bytes: bytes, timeout: int = 60):
    """Processa uma página (em bytes) com retry. Retorna (dados, ok, tempo, provider)."""
    for tentativa in range(MAX_RETRIES + 1):
        try:
            start = time.time()
            resp = model.generate_content(
                [prompt_instrucao, {"mime_type": "application/pdf", "data": page_bytes}],
                generation_config={"response_mime_type": "application/json"},
                request_options={'timeout': timeout}
            )
            tempo = round(time.time() - start, 2)
            texto = resp.text.strip()
            # limpeza rápida
            if texto.startswith("```"):
                texto = texto.replace("```json", "").replace("```", "").strip()
            try:
                dados = json.loads(texto)
            except Exception as e:
                dados = {"error": f"Resposta não era JSON válido: {str(e)}", "_raw": texto[:500]}
            return dados, True, tempo, "Gemini"
        except ResourceExhausted as e:
            delay = calcular_delay(tentativa, str(e))
            # não use st.sidebar.warning em threads; escrevemos aviso via retornos
            time.sleep(delay)
        except Exception as e:
            if tentativa < MAX_RETRIES:
                time.sleep(MIN_RETRY_DELAY)
            else:
                return {"error": str(e)}, False, 0, "Gemini"
    return {"error": "Falha máxima de tentativas"}, False, 0, "Gemini"

# =====================================================================
# SIDEBAR: Configurações + UI de Gestão de Padrões (MODIFICAÇÃO)
# =====================================================================
with st.sidebar:
    st.markdown("### 🔧 Configurações")
    st.markdown("#### Otimizações")
    use_cache = st.checkbox("Usar Cache", value=True, key="use_cache")
    worker_count = st.number_input("Threads (Worker pool)", min_value=1, max_value=16, value=MAX_WORKERS_DEFAULT, step=1, key="worker_count")
    st.markdown("---")
    st.markdown("### 🧩 Padrões de Renomeação (persistentes)")
    # Mostra lista de padrões
    st.markdown("**Padrões existentes:**")
    for k, v in PATTERNS.items():
        st.markdown(f"- `{k}` → `{v}`")
    st.markdown("**Adicionar novo padrão**")
    new_pat_raw = st.text_input("Texto a reconhecer (ex: 'EMPRESA XYZ LTDA')", key="new_pat_raw")
    new_pat_sub = st.text_input("Substituto (ex: 'EMPRESA_XYZ')", key="new_pat_sub")
    if st.button("➕ Adicionar padrão"):
        ok, msg = add_pattern(new_pat_raw.strip(), new_pat_sub.strip())
        if ok:
            st.success(msg)
            st.rerun()
        else:
            st.warning(msg)
    st.markdown("**Editar / Excluir**")
    edit_sel = st.selectbox("Selecionar padrão para editar/excluir", options=[""] + list(PATTERNS.keys()), index=0, key="edit_sel")
    if edit_sel:
        col_e1, col_e2 = st.columns([2,1])
        with col_e1:
            edit_raw = st.text_input("Padrão", value=edit_sel, key="edit_raw")
            edit_sub = st.text_input("Substituto", value=PATTERNS.get(edit_sel, ""), key="edit_sub")
        with col_e2:
            if st.button("✏️ Salvar edição"):
                ok, msg = edit_pattern(edit_sel, edit_raw.strip(), edit_sub.strip())
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.warning(msg)
            if st.button("🗑️ Excluir padrão"):
                ok, msg = remove_pattern(edit_sel)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.warning(msg)
    st.markdown("---")
    if st.button("🔄 Limpar Cache"):
        document_cache.clear()
        st.success("Cache limpo!")
        st.rerun()

# =====================================================================
# DASHBOARD ANALÍTICO (mantive a sua lógica)
# =====================================================================
def criar_dashboard_analitico():
    if "resultados" not in st.session_state:
        return
    st.markdown("---")
    st.markdown("### 📊 Dashboard Analítico")
    resultados = st.session_state["resultados"]
    logs = st.session_state.get("processed_logs", [])
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_arquivos = len(resultados)
        st.metric("📁 Arquivos Processados", total_arquivos)
    with col2:
        total_paginas = sum(r.get('pages', 1) for r in resultados)
        st.metric("📄 Total de Páginas", total_paginas)
    with col3:
        sucessos = len([log for log in logs if log[2] == "OK"])
        st.metric("✅ Sucessos", sucessos)
    with col4:
        erros = len([log for log in logs if log[2] != "OK"])
        st.metric("❌ Erros", erros)
    if resultados:
        st.markdown("#### 📈 Emitentes Mais Frequentes")
        emitentes = {}
        for r in resultados:
            emitente = r.get('emitente', 'Desconhecido')
            emitentes[emitente] = emitentes.get(emitente, 0) + 1
        for emitente, count in sorted(emitentes.items(), key=lambda x: x[1], reverse=True)[:5]:
            st.write(f"`{emitente}`: {count} documento(s)")

# =====================================================================
# UPLOAD E PROCESSAMENTO (MODE B - HÍBRIDO MULTITHREAD)
# - Processa cada página com ThreadPoolExecutor, mas agrupa por (numero, emitente)
# - Usa cache por arquivo (se cache existir, pula todo o processamento do arquivo)
# =====================================================================

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 📎 Enviar PDFs e processar ")
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
        if k in st.session_state:
            del st.session_state[k]
    st.success("Sessão limpa.")
    st.rerun()

if uploaded_files and process_btn:
    session_id = str(uuid.uuid4())
    session_folder = TEMP_FOLDER / session_id
    os.makedirs(session_folder, exist_ok=True)

    # Construir lista de arquivos (bytes) e total de páginas
    arquivos = []
    total_paginas = 0
    for f in uploaded_files:
        try:
            b = f.read()
            reader = PdfReader(io.BytesIO(b))
            n_pages = len(reader.pages)
            total_paginas += n_pages
            arquivos.append({"name": f.name, "bytes": b, "pages": n_pages})
        except Exception:
            st.warning(f"Erro ao ler {f.name}, ignorado.")

    st.info(f"📄 Total de páginas a processar: {total_paginas}")

    # Estruturas de resultado
    agrupados_bytes = {}
    resultados_meta = []
    processed_logs = []
    progresso = 0
    progress_bar = st.progress(0.0)
    progresso_text = st.empty()
    start_all = time.time()

    prompt = (
        "Analise a nota fiscal (DANFE). Extraia emitente, número da nota e cidade. "
        "Responda SOMENTE em JSON: {\"emitente\":\"NOME\",\"numero_nota\":\"NUMERO\",\"cidade\":\"CIDADE\"}"
    )

    # Executor config (MODIFICAÇÃO)
    worker_count = int(st.session_state.get("worker_count", MAX_WORKERS_DEFAULT))

    for a in arquivos:
        name = a["name"]
        file_bytes = a["bytes"]
        n_pages = a["pages"]

        # Cache por arquivo inteiro
        cache_key = document_cache.get_cache_key_file(file_bytes, prompt)
        cached_result = document_cache.get(cache_key) if use_cache else None
        if cached_result:
            st.sidebar.info(f"💾 Usando cache para {name}")
            # cached_result expected to be {'pages': [...], 'meta': [...], 'grouped': {...}}
            # Reutiliza cached grouped PDFs: armazenamos os 'dados' extraidos por página
            page_results = cached_result.get("page_results", [])
        else:
            # precisamos gerar page bytes e processá-los em paralelo
            try:
                reader = PdfReader(io.BytesIO(file_bytes))
            except Exception as e:
                processed_logs.append((name, 0, "ERRO_LEITURA", str(e), "Gemini"))
                continue

            # montar lista de page-bytes (em memória) sem escrever arquivos
            page_buffers = []
            for idx, page in enumerate(reader.pages):
                b = io.BytesIO()
                w = PdfWriter()
                w.add_page(page)
                w.write(b)
                page_bytes = b.getvalue()
                page_buffers.append((idx, page_bytes))

            # Processar com ThreadPoolExecutor
            page_results = [None] * len(page_buffers)
            futures = []
            with ThreadPoolExecutor(max_workers=worker_count) as ex:
                for idx, page_bytes in page_buffers:
                    futures.append(ex.submit(processar_pagina_gemini_single, prompt, page_bytes))
                # coletar resultados conforme terminam
                for i, fut in enumerate(as_completed(futures)):
                    try:
                        dados, ok, tempo, provider = fut.result()
                    except Exception as e:
                        dados, ok, tempo, provider = {"error": str(e)}, False, 0, "Gemini"
                    # encontrar posição correspondente: usamos index i (não é ideal) -> em vez disso, vamos mapear pela ordem de submissão
                    # para manter simplicidade, recolhemos results em append e depois ordenamos pela página idx original
                    page_results[i] = (dados, ok, tempo, provider)

            # NOTE: above as_completed doesn't preserve order of pages.
            # Re-run in ordered collection to ensure no mix: to be robust, process using map to keep order:
            # (we used as_completed for earlier fast path; but to ensure order we recompute ordered below)
            # To be safe, do ordered processing with map if futures length small:
            if len(page_buffers) > 0:
                # try ordered map to keep page alignment
                try:
                    with ThreadPoolExecutor(max_workers=worker_count) as ex2:
                        ordered_results = list(ex2.map(lambda pb: processar_pagina_gemini_single(prompt, pb[1]), page_buffers))
                        page_results = ordered_results
                except Exception:
                    # fallback mantido
                    pass

            # salvar cache parcial por arquivo
            if use_cache:
                document_cache.set(cache_key, {
                    "page_results": page_results,
                    "generated_at": time.time()
                })

        # agora iterar por resultados e agrupar
        # page_results is list of tuples (dados, ok, tempo, provider) in page order
        for page_idx, res in enumerate(page_results):
            if res is None:
                # falha silenciosa
                processed_logs.append((f"{name} (pág {page_idx+1})", 0, "ERRO_IA", "Sem resposta", "Gemini"))
                progresso += 1
                if progresso % 3 == 0:
                    progress_bar.progress(min(progresso/total_paginas, 1.0))
                continue
            dados, ok, tempo, provider = res
            page_label = f"{name} (pág {page_idx+1})"
            if not ok or "error" in dados:
                processed_logs.append((page_label, tempo, "ERRO_IA", dados.get("error", str(dados)), provider))
                progresso += 1
                if progresso % 3 == 0:
                    progress_bar.progress(min(progresso/total_paginas, 1.0))
                progresso_text.markdown(f"<span class='warning-log'>⚠️ {page_label} — ERRO IA</span>", unsafe_allow_html=True)
                resultados_meta.append({
                    "arquivo_origem": name,
                    "pagina": page_idx+1,
                    "emitente_detectado": dados.get("emitente") if isinstance(dados, dict) else "-",
                    "numero_detectado": dados.get("numero_nota") if isinstance(dados, dict) else "-",
                    "status": "ERRO",
                    "provider": provider
                })
                continue

            # Validar e corrigir dados
            dados = validar_e_corrigir_dados(dados)
            emitente_raw = dados.get("emitente", "") or ""
            numero_raw = dados.get("numero_nota", "") or ""
            cidade_raw = dados.get("cidade", "") or ""

            numero = limpar_numero(numero_raw)
            nome_map = substituir_nome_emitente(emitente_raw, cidade_raw)
            emitente = limpar_emitente(nome_map)

            key = (numero, emitente)
            agrupados_bytes.setdefault(key, []).append({
                "arquivo_origem": name,
                "pagina": page_idx + 1,
                # armazenamos bytes comprimidos por página para reconstituir depois
                "bytes": None  # we'll reconstruct from original file later to avoid extra memory
            })

            # Para reconstrução dos bytes sem reler, vamos armazenar o PDF completo e pagina indices
            # mas para simplicidade, vamos re-abrir o arquivo original quando montar PDFs finais

            processed_logs.append((page_label, tempo, "OK", f"{numero} / {emitente}", provider))
            resultados_meta.append({
                "arquivo_origem": name,
                "pagina": page_idx+1,
                "emitente_detectado": emitente_raw,
                "numero_detectado": numero_raw,
                "status": "OK",
                "tempo_s": round(tempo, 2),
                "provider": provider
            })

            progresso += 1
            if progresso % 3 == 0 or progresso == total_paginas:
                progress_bar.progress(min(progresso/total_paginas, 1.0))
            progresso_text.markdown(f"<span class='success-log'>✅ {page_label} — OK ({tempo:.2f}s)</span>", unsafe_allow_html=True)

    # Montar arquivos finais: precisamos re-ler os PDFs originais para pegar as páginas corretas.
    # Para isso, primeiro criamos um mapa de (arquivo_origem -> bytes) para reabertura
    arquivos_map = {a["name"]: a["bytes"] for a in arquivos}

    resultados = []
    files_meta = {}

    for (numero, emitente), pages_list in agrupados_bytes.items():
        if not numero or numero == "0":
            continue
        writer = PdfWriter()
        total_pages_added = 0
        # pages_list items have arquivo_origem and pagina
        for entry in pages_list:
            origem = entry["arquivo_origem"]
            pagina_idx = entry["pagina"] - 1
            file_bytes = arquivos_map.get(origem)
            if not file_bytes:
                continue
            try:
                r = PdfReader(io.BytesIO(file_bytes))
                if 0 <= pagina_idx < len(r.pages):
                    writer.add_page(r.pages[pagina_idx])
                    total_pages_added += 1
            except Exception:
                continue
        if total_pages_added == 0:
            continue
        nome_pdf = f"DOC {numero}_{emitente}.pdf"
        caminho = session_folder / nome_pdf
        with open(caminho, "wb") as f_out:
            writer.write(f_out)
        resultados.append({
            "file": nome_pdf,
            "numero": numero,
            "emitente": emitente,
            "pages": total_pages_added
        })
        files_meta[nome_pdf] = {"numero": numero, "emitente": emitente, "pages": total_pages_added}

    st.session_state["resultados"] = resultados
    st.session_state["session_folder"] = str(session_folder)
    st.session_state["novos_nomes"] = {r["file"]: r["file"] for r in resultados}
    st.session_state["processed_logs"] = processed_logs
    st.session_state["files_meta"] = files_meta

    st.success(f"✅ Processamento concluído em {round(time.time() - start_all, 2)}s — {len(resultados)} arquivos gerados.")

    criar_dashboard_analitico()
    st.rerun()

# =====================================================================
# PAINEL CORPORATIVO - GERENCIAMENTO (mantive sua lógica, pequenas correções)
# =====================================================================
if "resultados" in st.session_state:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Gerenciamento — selecione e aplique ações")
    resultados = st.session_state["resultados"]
    session_folder = Path(st.session_state["session_folder"])
    novos_nomes = st.session_state.get("novos_nomes", {r["file"]: r["file"] for r in resultados})
    files_meta = st.session_state.get("files_meta", {})

    col1, col2, col3, col4 = st.columns([3,2,2,2])
    with col1:
        q = st.text_input("🔎 Buscar arquivo ou emitente", value="", placeholder="parte do nome, emitente ou número")
    with col2:
        sort_by = st.selectbox("Ordenar por", ["Nome (A-Z)", "Nome (Z-A)", "Número (asc)", "Número (desc)"], index=0)
    with col3:
        show_logs = st.checkbox("Mostrar logs detalhados", value=False)
    with col4:
        if st.button("⬇️ Baixar Selecionadas"):
            sel = st.session_state.get("selected_files", [])
            if not sel:
                st.warning("Nenhuma nota selecionada para download.")
            else:
                mem = io.BytesIO()
                with zipfile.ZipFile(mem, "w") as zf:
                    for f in sel:
                        src = session_folder / f
                        if src.exists():
                            arcname = novos_nomes.get(f, f)
                            zf.write(src, arcname=arcname)
                mem.seek(0)
                st.download_button("⬇️ Clique novamente para confirmar download", data=mem, file_name="selecionadas.zip", mime="application/zip")
        if st.button("🗑️ Excluir Selecionadas"):
            sel = st.session_state.get("selected_files", [])
            if not sel:
                st.warning("Nenhuma nota selecionada para exclusão.")
            else:
                count = 0
                for f in sel:
                    src = session_folder / f
                    try:
                        if src.exists():
                            src.unlink()
                    except Exception:
                        pass
                    st.session_state["resultados"] = [r for r in st.session_state["resultados"] if r["file"] != f]
                    if f in st.session_state.get("novos_nomes", {}):
                        st.session_state["novos_nomes"].pop(f, None)
                    if f in st.session_state.get("files_meta", {}):
                        st.session_state["files_meta"].pop(f, None)
                    count += 1
                st.success(f"{count} arquivo(s) excluído(s).")
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    visible = resultados.copy()
    if q:
        q_up = q.strip().upper()
        visible = [r for r in visible if q_up in r["file"].upper() or q_up in r["emitente"].upper() or q_up in r["numero"]]
    if sort_by == "Nome (A-Z)":
        visible.sort(key=lambda x: x["file"])
    elif sort_by == "Nome (Z-A)":
        visible.sort(key=lambda x: x["file"], reverse=True)
    elif sort_by == "Número (asc)":
        visible.sort(key=lambda x: int(x["numero"]) if x["numero"].isdigit() else 0)
    else:
        visible.sort(key=lambda x: int(x["numero"]) if x["numero"].isdigit() else 0, reverse=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📁 Notas processadas")

    if "selected_files" not in st.session_state:
        st.session_state["selected_files"] = []

    for r in visible:
        fname = r["file"]
        meta = files_meta.get(fname, {})
        cols = st.columns([0.06, 0.48, 0.28, 0.18])

        checked = fname in st.session_state.get("selected_files", [])
        cb = cols[0].checkbox("", value=checked, key=f"cb_{fname}")

        if cb and fname not in st.session_state["selected_files"]:
            st.session_state["selected_files"].append(fname)
        if (not cb) and fname in st.session_state["selected_files"]:
            st.session_state["selected_files"].remove(fname)

        novos_nomes[fname] = cols[1].text_input(label=fname, value=novos_nomes.get(fname, fname), key=f"rename_input_{fname}")

        emit = meta.get("emitente", r.get("emitente", "-"))
        num = meta.get("numero", r.get("numero", "-"))
        cols[2].markdown(f"<div class='small-note'>{emit}  •  Nº {num}  •  {r.get('pages',1)} pág(s)</div>", unsafe_allow_html=True)

        action_col = cols[3]
        action = action_col.selectbox("", options=["...", "Remover (mover p/ lixeira)", "Baixar este arquivo"], key=f"action_{fname}", index=0)

        if action_col.button("⚙️ Gerenciar", key=f"manage_{fname}"):
            st.session_state["_manage_target"] = fname
            st.rerun()

        if action == "Remover (mover p/ lixeira)":
            src = session_folder / fname
            try:
                if src.exists():
                    src.unlink()
            except Exception:
                pass
            st.session_state["resultados"] = [x for x in st.session_state["resultados"] if x["file"] != fname]
            if fname in st.session_state.get("novos_nomes", {}):
                st.session_state["novos_nomes"].pop(fname, None)
            if fname in st.session_state.get("files_meta", {}):
                st.session_state["files_meta"].pop(fname, None)
            st.success(f"{fname} removido.")
            st.rerun()
        elif action == "Baixar este arquivo":
            src = session_folder / fname
            if src.exists():
                with open(src, "rb") as ff:
                    data = ff.read()
                st.download_button(f"⬇️ Baixar {fname}", data=data, file_name=novos_nomes.get(fname, fname), mime="application/pdf")
            else:
                st.warning("Arquivo não encontrado.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Painel de gestão por arquivo (mantido)
    if "_manage_target" in st.session_state:
        manage_target = st.session_state["_manage_target"]
        if not any(r["file"] == manage_target for r in st.session_state.get("resultados", [])):
            st.session_state.pop("_manage_target", None)
            st.rerun()

        st.markdown('<div class="manage-panel">', unsafe_allow_html=True)
        st.markdown(f"### ⚙️ Gerenciar: `{manage_target}`")

        file_path = session_folder / manage_target

        try:
            reader = PdfReader(str(file_path))
            total_pages = len(reader.pages)
            pages_info = [{"idx": i, "label": f"Página {i+1}"} for i in range(total_pages)]
        except Exception as e:
            st.error(f"Erro ao ler o arquivo: {str(e)}")
            pages_info = []
            total_pages = 0

        if pages_info:
            st.info(f"📄 O arquivo possui **{total_pages} página(s)**")

            sel_key = f"_manage_sel_{manage_target}"
            if sel_key not in st.session_state:
                st.session_state[sel_key] = []

            col_sel, col_actions = st.columns([1, 2])

            with col_sel:
                st.markdown("**Selecionar páginas:**")
                for page in pages_info:
                    is_checked = page["idx"] in st.session_state.get(sel_key, [])
                    if st.checkbox(page["label"], value=is_checked, key=f"{sel_key}_{page['idx']}"):
                        if page["idx"] not in st.session_state[sel_key]:
                            st.session_state[sel_key].append(page["idx"])
                    else:
                        if page["idx"] in st.session_state[sel_key]:
                            st.session_state[sel_key].remove(page["idx"])

            with col_actions:
                st.markdown("**Ações:**")
                selected_count = len(st.session_state.get(sel_key, []))
                st.write(f"📑 Páginas selecionadas: **{selected_count}**")
                new_name_key = f"_manage_newname_{manage_target}"
                if new_name_key not in st.session_state:
                    base_name = manage_target.rsplit('.pdf', 1)[0]
                    st.session_state[new_name_key] = f"{base_name}_parte.pdf"

                new_name = st.text_input("Nome do novo PDF:",
                                       value=st.session_state[new_name_key],
                                       key=new_name_key)

                col_sep, col_rem, col_close = st.columns(3)

                with col_sep:
                    if st.button("➗ Separar páginas", key=f"sep_{manage_target}"):
                        selected = sorted(st.session_state.get(sel_key, []))
                        if not selected:
                            st.warning("Selecione pelo menos uma página para separar.")
                        else:
                            try:
                                new_writer = PdfWriter()
                                reader = PdfReader(str(file_path))
                                for page_idx in selected:
                                    if 0 <= page_idx < len(reader.pages):
                                        new_writer.add_page(reader.pages[page_idx])
                                new_path = session_folder / new_name
                                with open(new_path, "wb") as f:
                                    new_writer.write(f)
                                new_meta = {
                                    "file": new_name,
                                    "numero": files_meta.get(manage_target, {}).get("numero", ""),
                                    "emitente": files_meta.get(manage_target, {}).get("emitente", ""),
                                    "pages": len(selected)
                                }
                                st.session_state["resultados"].append(new_meta)
                                st.session_state["files_meta"][new_name] = {
                                    "numero": new_meta["numero"],
                                    "emitente": new_meta["emitente"],
                                    "pages": new_meta["pages"]
                                }
                                st.session_state["novos_nomes"][new_name] = new_name
                                st.success(f"✅ Arquivo separado criado: `{new_name}`")
                                st.session_state[sel_key] = []
                            except Exception as e:
                                st.error(f"❌ Erro ao separar páginas: {str(e)}")

                with col_rem:
                    if st.button("🗑️ Remover páginas", key=f"rem_{manage_target}"):
                        selected = sorted(st.session_state.get(sel_key, []))
                        if not selected:
                            st.warning("Selecione páginas para remover.")
                        else:
                            try:
                                new_writer = PdfWriter()
                                reader = PdfReader(str(file_path))
                                for page_idx in range(len(reader.pages)):
                                    if page_idx not in selected:
                                        new_writer.add_page(reader.pages[page_idx])
                                if len(new_writer.pages) > 0:
                                    with open(file_path, "wb") as f:
                                        new_writer.write(f)
                                    st.session_state["files_meta"][manage_target]["pages"] = len(new_writer.pages)
                                    for r in st.session_state["resultados"]:
                                        if r["file"] == manage_target:
                                            r["pages"] = len(new_writer.pages)
                                    st.success(f"✅ {len(selected)} página(s) removida(s)")
                                else:
                                    file_path.unlink()
                                    st.session_state["resultados"] = [r for r in st.session_state["resultados"] if r["file"] != manage_target]
                                    st.session_state["files_meta"].pop(manage_target, None)
                                    st.session_state["novos_nomes"].pop(manage_target, None)
                                    st.success(f"📭 Arquivo `{manage_target}` foi excluído (ficou vazio)")
                                    st.session_state.pop("_manage_target", None)
                                st.session_state[sel_key] = []  # Limpar seleção
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Erro ao remover páginas: {str(e)}")

                with col_close:
                    if st.button("❌ Fechar", key=f"close_{manage_target}"):
                        st.session_state.pop("_manage_target", None)
                        st.session_state.pop(sel_key, None)
                        st.rerun()
        else:
            st.warning("Não foi possível carregar as páginas do arquivo.")
            if st.button("❌ Fechar", key=f"close_err_{manage_target}"):
                st.session_state.pop("_manage_target", None)
                st.rerun()

    criar_dashboard_analitico()

    if show_logs and st.session_state.get("processed_logs"):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📝 Logs de processamento (últimas páginas)")
        for entry in st.session_state["processed_logs"][-200:]:
            label, t, status, info, provider = (entry + ("", "", ""))[:5]
            if status == "OK":
                st.markdown(f"<div class='success-log'>✅ {label} — {info} — {t:.2f}s</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='warning-log'>⚠️ {label} — {info}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.session_state["novos_nomes"] = novos_nomes

    st.markdown("---")
    col_dl_a, col_dl_b = st.columns([1,3])
    with col_dl_a:
        if st.button("📦 Baixar tudo (ZIP)"):
            mem = io.BytesIO()
            with zipfile.ZipFile(mem, "w") as zf:
                for r in st.session_state.get("resultados", []):
                    fname = r["file"]
                    src = session_folder / fname
                    if src.exists():
                        zf.write(src, arcname=st.session_state.get("novos_nomes", {}).get(fname, fname))
            mem.seek(0)
            st.download_button("⬇️ Clique para baixar (ZIP)", data=mem, file_name="notas_processadas.zip", mime="application/zip")
    with col_dl_b:
        st.markdown("<div class='small-note'>Dica: edite nomes na lista e use 'Baixar Selecionadas' para baixar apenas o que precisar.</div>", unsafe_allow_html=True)

else:
    st.info("Nenhum arquivo processado ainda. Faça upload e clique em 'Processar PDFs'.")
