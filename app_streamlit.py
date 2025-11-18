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
    page_icon="📄",
    layout="wide"
)

# ======= CSS Corporativo =======
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
.card { 
  background: #fff; 
  padding: 12px; 
  border-radius:8px; 
  box-shadow: 0 6px 18px rgba(15,76,129,0.04); 
  margin-bottom:12px; 
}
.manage-panel { 
  background: #f8f9fa; 
  padding: 15px; 
  border-radius: 8px; 
  border-left: 4px solid #0f4c81; 
  margin: 10px 0; 
}
.small-note {
  font-size:13px;
  color:#6b7280;
}
</style>
""", unsafe_allow_html=True)

st.title("Automatizador de Notas Fiscais PDF — Turbo Seguro")

# =====================================================================
# CONFIGURAÇÕES GERAIS E ESTRUTURAS
# =====================================================================

TEMP_FOLDER = Path("./temp")
TEMP_FOLDER.mkdir(exist_ok=True)

CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)

CONFIG_DIR = Path("./config")
CONFIG_DIR.mkdir(exist_ok=True)

PATTERNS_FILE = CONFIG_DIR / "patterns.json"

# LIMITES E THREADS
MAX_WORKERS_DEFAULT = max(2, min(4, (os.cpu_count() or 2)))
MAX_TOTAL_PAGES = 500
MAX_RETRIES = 2
MIN_RETRY_DELAY = 5
MAX_RETRY_DELAY = 30

# =====================================================================
# SISTEMA DE CACHE INTELIGENTE
# =====================================================================
class DocumentCache:
    def __init__(self, cache_dir=CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _cache_path(self, key: str):
        safe = re.sub(r'[^a-zA-Z0-9_\-]', '_', key)
        return self.cache_dir / f"{safe}.pkl"

    def get_cache_key_file(self, file_bytes: bytes, prompt: str):
        """Chave por arquivo inteiro + prompt"""
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
                # corrup file -> remove
                try:
                    cache_file.unlink()
                except Exception:
                    pass
                return None
            except Exception:
                return None
        return None

    def set(self, key, data):
        cache_file = self._cache_path(key)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception:
            # não interrompe a execução por falha no cache
            return

    def clear(self):
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except Exception:
                pass

document_cache = DocumentCache()

# =====================================================================
# PADRÕES DE RENOMEAÇÃO PERSISTENTES (LOAD / SAVE / CRUD)
# =====================================================================
def load_patterns():
    if not PATTERNS_FILE.exists():
        default_patterns = {
            "COMPANHIA DE AGUA E ESGOTOS DA PARAIBA": "CAGEPA",
            "COMPANHIA DE AGUA E ESGOTOS DA PARAÍBA": "CAGEPA",
            "CIA DE AGUA E ESGOTO DO CEARA": "CAGECE",
            "TRANSPORTE LIDA": "TRANSPORTE_LIDA",
            "UNIPAR CARBOCLORO": "UNIPAR_CARBOCLORO",
            "EXPRESS TCM": "EXPRESS_TCM",
            "MDM RENOVADORA DE PNEUS": "MDM_RENOVADORA",
            "COMPANHIA DE AGUAS E ESGOTOS DO RN": "CAERN",
            "EKIPE TEC DE SEG E INCENDIO": "EKIPE",
            "PETROLEO BRASILEIRO": "PETROBRAS",
            "INNOVATIVE WATER CARE": "SIGURA",
            "COMERCIAL E IMPORTADORA DE PNEUS": "CAMPNEUS",
            "URP CARGAS E LOGISTICA": "URP",
            "M.F DE MELO FILHO": "MF_DE_MELO",
        }
        save_patterns(default_patterns)
        return default_patterns
    try:
        with open(PATTERNS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        return {}

def save_patterns(pats: dict):
    try:
        with open(PATTERNS_FILE, "w", encoding="utf-8") as f:
            json.dump(pats, f, ensure_ascii=False, indent=2)
    except Exception:
        return

# carregar na inicialização
PATTERNS = load_patterns()

def normalize_pattern_key(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII")
    s = re.sub(r"[^A-Z0-9 ]+", " ", s.upper())
    return re.sub(r"\s+", " ", s).strip()

def add_pattern(raw_pattern: str, substitute: str) -> tuple:
    """Adiciona novo padrão. Retorna (ok, mensagem)"""
    raw_pattern = raw_pattern.strip()
    substitute = substitute.strip()
    key_norm = normalize_pattern_key(raw_pattern)
    if not key_norm:
        return False, "Padrão vazio."
    # prevenir conflito com mesma normalização
    for existing in PATTERNS.keys():
        if normalize_pattern_key(existing) == key_norm:
            return False, "Conflito: padrão já existe (mesma normalização)."
    PATTERNS[raw_pattern] = substitute
    save_patterns(PATTERNS)
    return True, "Padrão adicionado."

def edit_pattern(old_raw: str, new_raw: str, new_sub: str) -> tuple:
    old_raw = old_raw.strip()
    new_raw = new_raw.strip()
    new_sub = new_sub.strip()
    if old_raw not in PATTERNS:
        return False, "Padrão não encontrado."
    key_norm = normalize_pattern_key(new_raw)
    for k in PATTERNS.keys():
        if k != old_raw and normalize_pattern_key(k) == key_norm:
            return False, "Conflito: outro padrão com mesma normalização."
    # substituir preservando ordem aproximada
    PATTERNS.pop(old_raw)
    PATTERNS[new_raw] = new_sub
    save_patterns(PATTERNS)
    return True, "Padrão editado."

def remove_pattern(raw_pattern: str) -> tuple:
    raw_pattern = raw_pattern.strip()
    if raw_pattern in PATTERNS:
        PATTERNS.pop(raw_pattern)
        save_patterns(PATTERNS)
        return True, "Removido"
    return False, "Não achou padrão"

# =====================================================================
# NORMALIZAÇÃO E SUBSTITUIÇÕES (USADAS NA RENOMEAÇÃO)
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
    # procurar padrões com maior especificidade primeiro (ordenar por tamanho da chave DESC)
    for padrao in sorted(PATTERNS.keys(), key=lambda x: len(x), reverse=True):
        if _normalizar_texto(padrao) in nome_norm:
            return PATTERNS[padrao]
    # fallback: transformar nome normalizado para snake
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
    for field in required_fields:
        if field not in dados or not dados[field]:
            dados[field] = "NÃO_IDENTIFICADO"
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
    if 'numero_nota' in dados:
        numero_limpo = re.sub(r'[^\d]', '', str(dados['numero_nota']))
        dados['numero_nota'] = numero_limpo if numero_limpo else "000000"
    return dados

# =====================================================================
# CONFIGURAÇÃO GEMINI
# =====================================================================
if hasattr(st, "secrets") and st.secrets.get("GOOGLE_API_KEY"):
    GEMINI_API_KEY = st.secrets["GOOGLE_API_KEY"]
else:
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    st.error("❌ Chave GOOGLE_API_KEY não encontrada.")
    st.stop()

try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(os.getenv("MODEL_NAME", "models/gemini-2.0-flash-exp"))
    st.sidebar.success("✅ Gemini configurado")
except Exception as e:
    st.error(f"❌ Erro ao configurar Gemini: {str(e)}")
    st.stop()

# =====================================================================
# FUNÇÕES DE RETRY E PROCESSAMENTO INDIVIDUAL
# =====================================================================
def calcular_delay(tentativa, error_msg):
    """Cálculo de delay seguro para Streamlit Cloud."""
    base = MIN_RETRY_DELAY * (tentativa + 1)
    if not error_msg:
        return min(base, MAX_RETRY_DELAY)
    error_msg = error_msg.lower()
    if "retry in" in error_msg:
        try:
            seg = float(re.search(r"retry in (\d+\.?\d*)s", error_msg).group(1))
            return min(seg + 2, MAX_RETRY_DELAY)
        except:
            pass
    return min(base, MAX_RETRY_DELAY)

def processar_pagina_gemini_single(prompt_instrucao: str, page_bytes: bytes, timeout: int = 60):
    """Processa uma página com retry. Retorna (dados, ok, tempo, provider)."""
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
            if texto.startswith("```"):
                texto = texto.replace("```json", "").replace("```", "").strip()

            try:
                dados = json.loads(texto)
            except:
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

# =====================================================================
# SIDEBAR: Configurações + UI de Padrões
# =====================================================================
with st.sidebar:
    st.markdown("### 🔧 Configurações")
    use_cache = st.checkbox("Usar Cache", value=True)

    st.markdown("#### Threads (Turbo Seguro)")
    worker_count = st.number_input(
        "Workers",
        min_value=1,
        max_value=8,
        value=MAX_WORKERS_DEFAULT,
        step=1
    )

    st.markdown("---")
    st.markdown("### 🧩 Padrões de Renomeação")

    with st.expander("📋 Ver padrões existentes"):
        for k, v in PATTERNS.items():
            st.markdown(f"- `{k}` → `{v}`")

    st.markdown("**Adicionar novo padrão**")
    new_pat_raw = st.text_input("Texto a reconhecer", key="new_pat_raw")
    new_pat_sub = st.text_input("Substituto", key="new_pat_sub")

    if st.button("➕ Adicionar padrão"):
        if new_pat_raw and new_pat_sub:
            ok, msg = add_pattern(new_pat_raw, new_pat_sub)
            if ok:
                st.success(msg)
                time.sleep(0.15)
                st.rerun()
            else:
                st.warning(msg)
        else:
            st.warning("Preencha ambos os campos")

    st.markdown("**Editar / Excluir**")
    edit_sel = st.selectbox("Selecione um padrão", [""] + list(PATTERNS.keys()))

    if edit_sel:
        col_e1, col_e2 = st.columns([2, 1])
        with col_e1:
            edit_raw = st.text_input("Padrão", value=edit_sel)
            edit_sub = st.text_input("Substituto", value=PATTERNS.get(edit_sel, ""))

        with col_e2:
            if st.button("✏️ Salvar"):
                ok, msg = edit_pattern(edit_sel, edit_raw, edit_sub)
                if ok:
                    st.success(msg)
                    time.sleep(0.15)
                    st.rerun()
                else:
                    st.warning(msg)

            if st.button("🗑️ Excluir"):
                ok, msg = remove_pattern(edit_sel)
                if ok:
                    st.success(msg)
                    time.sleep(0.15)
                    st.rerun()
                else:
                    st.warning(msg)

    st.markdown("---")
    if st.button("🧹 Limpar cache"):
        document_cache.clear()
        st.success("Cache limpo!")
        time.sleep(0.15)
        st.rerun()

# =====================================================================
# DASHBOARD ANALÍTICO
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
        st.metric("📁 Arquivos", len(resultados))

    with col2:
        total_paginas = sum(r.get('pages', 1) for r in resultados)
        st.metric("📄 Páginas", total_paginas)

    with col3:
        sucessos = len([log for log in logs if log[2] == "OK"])
        st.metric("✅ Sucessos", sucessos)

    with col4:
        erros = len([log for log in logs if log[2] != "OK"])
        st.metric("❌ Erros", erros)

    if resultados:
        st.markdown("#### 📈 Emitentes mais frequentes")
        emitentes = {}
        for r in resultados:
            em = r.get("emitente", "Desconhecido")
            emitentes[em] = emitentes.get(em, 0) + 1

        for em, qtd in sorted(emitentes.items(), key=lambda x: x[1], reverse=True)[:5]:
            st.write(f"`{em}`: {qtd} doc(s)")

# =====================================================================
# UPLOAD + PROCESSAMENTO MULTITHREAD POR PÁGINA (OPÇÃO A — TURBO)
# =====================================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 📎 Enviar PDFs e processar")

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
    clear_session = st.button("♻️ Limpar sessão")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# LIMPAR SESSÃO
# ---------------------------------------------------------------------
if clear_session:
    if "session_folder" in st.session_state:
        try:
            shutil.rmtree(st.session_state["session_folder"])
        except:
            pass

    for k in [
        "resultados", "session_folder", "novos_nomes",
        "processed_logs", "files_meta", "selected_files",
        "_manage_target"
    ]:
        st.session_state.pop(k, None)

    st.success("Sessão limpa.")
    time.sleep(0.15)
    st.rerun()

# ---------------------------------------------------------------------
# PROCESSAMENTO
# ---------------------------------------------------------------------
if uploaded_files and process_btn:

    session_id = str(uuid.uuid4())
    session_folder = TEMP_FOLDER / session_id
    session_folder.mkdir(exist_ok=True)

    arquivos = []
    total_paginas = 0

    # -----------------------------
    # Ler PDFs e contar páginas
    # -----------------------------
    for f in uploaded_files:
        try:
            b = f.read()
            reader = PdfReader(io.BytesIO(b))
            n_pages = len(reader.pages)
            total_paginas += n_pages

            arquivos.append({
                "name": f.name,
                "bytes": b,
                "pages": n_pages
            })

        except Exception:
            st.warning(f"❌ Erro ao abrir {f.name}, ignorado.")

    st.info(f"📄 Total de páginas a processar: **{total_paginas}**")

    # -----------------------------
    # Preparação
    # -----------------------------
    agrupados_bytes = {}
    resultados_meta = []
    processed_logs = []

    progresso = 0
    progress_bar = st.progress(0.0)
    progresso_text = st.empty()

    start_all = time.time()

    prompt = (
        "Analise a nota fiscal (DANFE). "
        "Extraia emitente, número da nota e cidade. "
        "Retorne SOMENTE JSON no formato: "
        "{\"emitente\":\"NOME\",\"numero_nota\":\"NUM\",\"cidade\":\"CIDADE\"}"
    )

    worker_count = int(st.session_state.get("worker_count", MAX_WORKERS_DEFAULT))

    # -----------------------------------------------------------
    # LOOP DOS ARQUIVOS
    # -----------------------------------------------------------
    for arquivo in arquivos:

        fname = arquivo["name"]
        file_bytes = arquivo["bytes"]
        n_pages = arquivo["pages"]

        # ---------- CACHE POR ARQUIVO INTEIRO ----------
        cache_key = document_cache.get_cache_key_file(file_bytes, prompt)
        cached = document_cache.get(cache_key) if use_cache else None

        if cached:
            page_results = cached.get("page_results", [])
        else:
            # -------------------------------------------
            # Extração das páginas e criação de jobs
            # -------------------------------------------
            try:
                reader = PdfReader(io.BytesIO(file_bytes))
            except Exception as e:
                processed_logs.append((fname, 0, "ERRO_LEITURA", str(e), "Gemini"))
                continue

            page_jobs = []
            for idx, page in enumerate(reader.pages):
                buffer = io.BytesIO()
                w = PdfWriter()
                w.add_page(page)
                w.write(buffer)
                page_jobs.append((idx, buffer.getvalue()))

            # -------------------------------------------
            # Execução paralela verdadeira por página
            # -------------------------------------------
            page_results = [None] * len(page_jobs)

            with ThreadPoolExecutor(max_workers=worker_count) as ex:
                future_map = {
                    ex.submit(processar_pagina_gemini_single, prompt, job_bytes): job_index
                    for job_index, job_bytes in page_jobs
                }

                for future in as_completed(future_map):
                    idx = future_map[future]
                    try:
                        page_results[idx] = future.result()
                    except Exception as e:
                        page_results[idx] = ({"error": str(e)}, False, 0, "Gemini")

            # Salvar no cache
            if use_cache:
                document_cache.set(cache_key, {
                    "page_results": page_results,
                    "generated_at": time.time()
                })

        # -----------------------------------------------------------
        # PROCESSAR RESPOSTAS INDIVIDUAIS
        # -----------------------------------------------------------
        for page_idx, result in enumerate(page_results):

            if result is None:
                processed_logs.append(
                    (f"{fname} (pág {page_idx+1})", 0, "ERRO_IA", "Sem resposta", "Gemini")
                )
                progresso += 1
                progress_bar.progress(progresso / total_paginas)
                continue

            dados, ok, tempo, provider = result
            page_label = f"{fname} (pág {page_idx+1})"

            # --------- ERRO GEMINI ---------
            if not ok or "error" in dados:
                processed_logs.append(
                    (page_label, tempo, "ERRO_IA", dados.get("error", "erro"), provider)
                )
                progresso += 1
                progress_bar.progress(progresso / total_paginas)
                progresso_text.markdown(
                    f"<div class='warning-log'>⚠️ {page_label} — ERRO</div>",
                    unsafe_allow_html=True
                )
                continue

            # --------- VALIDAR DADOS EXTRAÍDOS ---------
            dados = validar_e_corrigir_dados(dados)

            emit_raw = dados.get("emitente", "")
            num_raw = dados.get("numero_nota", "")
            cid_raw = dados.get("cidade", "")

            numero = limpar_numero(num_raw)
            nome_map = substituir_nome_emitente(emit_raw, cid_raw)
            emitente = limpar_emitente(nome_map)

            key = (numero, emitente)

            if key not in agrupados_bytes:
                agrupados_bytes[key] = []

            agrupados_bytes[key].append({
                "arquivo": fname,
                "pagina": page_idx
            })

            processed_logs.append(
                (page_label, tempo, "OK", f"{numero}/{emitente}", provider)
            )

            progresso += 1
            if progresso % 3 == 0 or progresso == total_paginas:
                progress_bar.progress(progresso / total_paginas)

            progresso_text.markdown(
                f"<div class='success-log'>✅ {page_label} — OK ({tempo:.2f}s)</div>",
                unsafe_allow_html=True
            )

    # =====================================================================
    # GERAR PDFs FINAIS AGRUPADOS
    # =====================================================================
    resultados = []
    files_meta = {}
    arquivos_map = {a["name"]: a["bytes"] for a in arquivos}

    for (numero, emitente), lista_paginas in agrupados_bytes.items():

        if not numero or numero == "0":
            continue

        writer = PdfWriter()
        count_added = 0

        for item in lista_paginas:
            orig = item["arquivo"]
            pg = item["pagina"]

            file_bytes = arquivos_map.get(orig)
            if not file_bytes:
                continue

            try:
                r = PdfReader(io.BytesIO(file_bytes))
                if 0 <= pg < len(r.pages):
                    writer.add_page(r.pages[pg])
                    count_added += 1
            except:
                continue

        if count_added == 0:
            continue

        nome_pdf = f"DOC {numero}_{emitente}.pdf"
        path_out = session_folder / nome_pdf

        with open(path_out, "wb") as f_out:
            writer.write(f_out)

        resultados.append({
            "file": nome_pdf,
            "numero": numero,
            "emitente": emitente,
            "pages": count_added
        })

        files_meta[nome_pdf] = {
            "numero": numero,
            "emitente": emitente,
            "pages": count_added
        }

    # =====================================================================
    # SALVAR ESTADO
    # =====================================================================
    st.session_state["resultados"] = resultados
    st.session_state["session_folder"] = str(session_folder)
    st.session_state["novos_nomes"] = {r["file"]: r["file"] for r in resultados}
    st.session_state["processed_logs"] = processed_logs
    st.session_state["files_meta"] = files_meta

    st.success(
        f"✅ Processamento concluído em {round(time.time() - start_all, 2)}s "
        f"— {len(resultados)} arquivos gerados."
    )

    criar_dashboard_analitico()
    time.sleep(0.15)
    st.rerun()

# =====================================================================
# PAINEL FINAL DE ARQUIVOS GERADOS
# =====================================================================
if "session_folder" in st.session_state and "resultados" in st.session_state:

    st.markdown("---")
    st.header("📁 Arquivos Gerados")

    session_folder = Path(st.session_state["session_folder"])
    resultados = st.session_state["resultados"]
    novos_nomes = st.session_state["novos_nomes"]
    files_meta = st.session_state["files_meta"]

    if not session_folder.exists():
        st.error("❌ Pasta da sessão não existe mais.")
        st.stop()

    # --------------------------
    # LISTA DOS ARQUIVOS
    # --------------------------
    for idx, r in enumerate(resultados):

        old_name = r["file"]
        new_name = novos_nomes.get(old_name, old_name)
        file_path = session_folder / old_name

        card_css = f"""
        <div class="card" style="margin-top:10px;border-left:4px solid #0f4c81;">
            <div style="font-weight:600;font-size:18px;">📄 {old_name}</div>
        """
        st.markdown(card_css, unsafe_allow_html=True)

        colA, colB, colC = st.columns([3, 3, 1])

        # --------------------------
        # RENOMEAÇÃO
        # --------------------------
        with colA:
            new_name_input = st.text_input(
                f"Novo nome para {old_name}",
                value=new_name,
                key=f"name_{idx}"
            )

        with colB:
            if st.button("💾 Salvar nome", key=f"save_{idx}"):
                if new_name_input.strip():
                    novos_nomes[old_name] = new_name_input.strip()
                    st.session_state["novos_nomes"] = novos_nomes
                    st.success("Nome atualizado!")
                    time.sleep(0.1)
                    st.rerun()

        # --------------------------
        # AÇÕES DO ARQUIVO
        # --------------------------
        with colC:
            if st.button("🗑️", key=f"del_{idx}"):
                try:
                    os.remove(file_path)
                except:
                    pass

                # Remover da lista final
                resultados = [x for x in resultados if x["file"] != old_name]
                st.session_state["resultados"] = resultados
                novos_nomes.pop(old_name, None)

                st.success("Arquivo excluído.")
                time.sleep(0.1)
                st.rerun()

        # --------------------------
        # PREVIEW + DOWNLOAD
        # --------------------------
        colD, colE = st.columns([1, 5])

        with colD:
            with open(file_path, "rb") as f_down:
                st.download_button(
                    "⬇️ Baixar PDF",
                    data=f_down,
                    file_name=new_name,
                    mime="application/pdf",
                    key=f"down_{idx}"
                )

        with colE:
            st.markdown(
                f"<div class='small-note'>📎 {files_meta[old_name]['pages']} páginas — {files_meta[old_name]['emitente']} / NF {files_meta[old_name]['numero']}</div>",
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------------------
    # EXTRA: SEPARAR PÁGINAS INDIVIDUAIS DOS PDFs GERADOS
    # ---------------------------------------------------------------------
    st.markdown("---")
    st.subheader("✂️ Separar páginas individuais")

    sel_file = st.selectbox(
        "Escolha um arquivo",
        [""] + [r["file"] for r in resultados],
        key="split_choice"
    )

    if sel_file:
        file_path = session_folder / sel_file

        try:
            reader = PdfReader(str(file_path))
            n_pages = len(reader.pages)

            st.info(f"📄 Total de páginas: **{n_pages}**")

            col_s1, col_s2 = st.columns([2, 1])

            with col_s1:
                pages_to_extract = st.text_input(
                    "Páginas (ex: 1,2,5-7)",
                    key="split_pages"
                )

            with col_s2:
                if st.button("✂️ Separar agora"):
                    try:
                        pages_list = []
                        for part in pages_to_extract.split(","):
                            part = part.strip()
                            if "-" in part:
                                a, b = part.split("-")
                                pages_list.extend(list(range(int(a), int(b) + 1)))
                            else:
                                pages_list.append(int(part))

                        writer = PdfWriter()
                        added = 0

                        for p in pages_list:
                            if 1 <= p <= n_pages:
                                writer.add_page(reader.pages[p - 1])
                                added += 1

                        if added == 0:
                            st.warning("Nenhuma página válida informada.")
                        else:
                            out_path = session_folder / f"{sel_file[:-4]}_split.pdf"
                            with open(out_path, "wb") as out_file:
                                writer.write(out_file)

                            with open(out_path, "rb") as f_out:
                                st.download_button(
                                    "⬇️ Baixar PDF separado",
                                    data=f_out,
                                    file_name=f"{sel_file[:-4]}_split.pdf",
                                    mime="application/pdf"
                                )

                            st.success(f"Arquivo gerado ({added} páginas).")

                    except Exception as e:
                        st.error(f"Erro ao separar páginas: {e}")

        except Exception as e:
            st.error(f"Erro ao separar páginas: {e}")

# =====================================================================
# BLOCO 6/6 — LOGS AVANÇADOS, EXPORT/IMPORT DE PADRÕES, LIMPEZA E FINALIZAÇÃO
# =====================================================================
# --- Funções utilitárias adicionais ---

def export_patterns_to_file(dest: Path):
    try:
        save_patterns(PATTERNS)
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(PATTERNS, f, ensure_ascii=False, indent=2)
        return True, f"Exportado para {dest}"
    except Exception as e:
        return False, str(e)

def import_patterns_from_file(src: Path):
    try:
        with open(src, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # merge with conflict prevention (normalize keys)
            for k, v in data.items():
                nk = normalize_pattern_key(k)
                conflict = False
                for ek in list(PATTERNS.keys()):
                    if normalize_pattern_key(ek) == nk and ek != k:
                        conflict = True
                        break
                if not conflict:
                    PATTERNS[k] = v
            save_patterns(PATTERNS)
            return True, "Importado e mesclado."
        return False, "Arquivo inválido."
    except Exception as e:
        return False, str(e)

# --- Painel de logs e exportações ---
st.markdown("---")
st.markdown("### 🧾 Logs & Exportações")

col_l1, col_l2, col_l3 = st.columns([2, 1, 1])
with col_l1:
    recent = st.session_state.get("processed_logs", [])[-500:]
    st.text_area("Logs recentes (últimas linhas)", value="\n".join(
        [f"{l[0]} | {l[2]} | {l[3]} | {l[4]}" for l in recent]
    ), height=180, key="logs_area")

with col_l2:
    # Export padrões
    export_path = CONFIG_DIR / f"patterns_export_{int(time.time())}.json"
    if st.button("📤 Exportar padrões"):
        ok, msg = export_patterns_to_file(export_path)
        if ok:
            with open(export_path, "rb") as f:
                st.download_button("⬇️ Baixar patterns.json", data=f, file_name=export_path.name, mime="application/json")
            st.success("Exportado com sucesso.")
        else:
            st.error(f"Erro: {msg}")

with col_l3:
    uploaded_patterns = st.file_uploader("📥 Importar padrões (.json)", type=["json"], key="import_patterns")
    if uploaded_patterns is not None:
        try:
            temp_import = CONFIG_DIR / f"import_{int(time.time())}.json"
            with open(temp_import, "wb") as f:
                f.write(uploaded_patterns.getvalue())
            ok, msg = import_patterns_from_file(temp_import)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
        except Exception as e:
            st.error(str(e))

# --- Consolidação e Download ZIP (final) ---
st.markdown("---")
st.markdown("### ✅ Finalizar / Baixar tudo")

col_f1, col_f2 = st.columns([1,2])
with col_f1:
    if st.button("📦 Baixar tudo (ZIP final)"):
        try:
            mem = io.BytesIO()
            with zipfile.ZipFile(mem, "w") as zf:
                for r in st.session_state.get("resultados", []):
                    fname = r["file"]
                    src = Path(st.session_state["session_folder"]) / fname
                    if src.exists():
                        arcname = st.session_state.get("novos_nomes", {}).get(fname, fname)
                        zf.write(src, arcname=arcname)
            mem.seek(0)
            st.download_button("⬇️ Confirmar download (ZIP)", data=mem, file_name="notas_processadas_final.zip", mime="application/zip")
            st.success("ZIP pronto para download.")
        except Exception as e:
            st.error(f"Erro ao gerar zip: {e}")

with col_f2:
    if st.button("🧹 Limpar sessão (arquivos temporários)"):
        try:
            sf = st.session_state.get("session_folder")
            if sf and Path(sf).exists():
                shutil.rmtree(sf, ignore_errors=True)
            # limpar variáveis
            for k in ["resultados", "session_folder", "novos_nomes", "processed_logs", "files_meta", "selected_files", "_manage_target"]:
                st.session_state.pop(k, None)
            st.success("Sessão limpa — arquivos temporários removidos.")
            # não forçar rerun automático se não quiser; aqui é usuário que chamou
            st.rerun()
        except Exception as e:
            st.error(f"Erro ao limpar sessão: {e}")

# --- Sugestões de tuning para produção (apenas UI) ---
st.markdown("---")
st.markdown("### ⚙️ Dicas rápidas de performance")
st.markdown("""
- Reduza `worker_count` se houver instabilidade no provedor de execução.
- Use `MAX_TOTAL_PAGES` por variável de ambiente para limitar lotes grandes.
- Em produção, mova o processamento pesado para um worker externo (Celery / Cloud Function) e apenas mostre resultados no Streamlit.
""")

# --- Segurança: checar uso de secrets ---
st.markdown("---")
if hasattr(st, "secrets") and st.secrets.get("GOOGLE_API_KEY"):
    st.success("🔒 Usando st.secrets para a chave Google (recomendado).")
else:
    st.warning("🔑 A chave Google está vindo de variáveis de ambiente. Considere usar st.secrets em produção.")

# --- Final: garantir que patterns estejam salvos ---
try:
    save_patterns(PATTERNS)
except Exception:
    pass
