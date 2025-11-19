# turbo_v5_refined.py
# Versão: Turbo v5 — Refined (pypdf + Regex JSON + Multithreading Real)
# Correções: fallback robusto, mapeamento perfeito, cache TTL, evitar sobrescrita, split e agrupamento manual

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
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- ATUALIZAÇÃO: Usando biblioteca moderna pypdf ---
from pypdf import PdfReader, PdfWriter

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
import streamlit as st
from dotenv import load_dotenv

# =====================================================================
# CONFIGURAÇÃO INICIAL
# =====================================================================
load_dotenv()
st.set_page_config(
    page_title="Automatizador de Notas Fiscais v5",
    page_icon="⚡",
    layout="wide"
)

# ======= CSS Corporativo =======
st.markdown(
    """
<style>
body { background-color: #f8f9fa; color: #212529; font-family: 'Segoe UI', Roboto, Arial, sans-serif; }
[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e9ecef; }
h1, h2, h3, h4 { color: #0f4c81; }
div.stButton > button { background-color: #0f4c81; color: white; border-radius: 8px; border: none; font-weight: 500; }
div.stButton > button:hover { background-color: #0b3a5a; }
.stProgress > div > div > div > div { background-color: #28a745 !important; }
.card { background: #fff; padding: 12px; border-radius:8px; box-shadow: 0 6px 18px rgba(15,76,129,0.04); margin-bottom:12px; }
.small-note { font-size:13px; color:#6b7280; }
.warning-box { background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 10px; border-radius: 4px; color: #856404; font-size: 0.9em; margin-bottom: 15px; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Automatizador de Notas Fiscais — Turbo v5 (Refined)")

# =====================================================================
# ESTRUTURAS E CONFIGURAÇÕES
# =====================================================================
TEMP_FOLDER = Path("./temp")
TEMP_FOLDER.mkdir(exist_ok=True)

CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)

CONFIG_DIR = Path("./config")
CONFIG_DIR.mkdir(exist_ok=True)

PATTERNS_FILE = CONFIG_DIR / "patterns.json"

# Limites (ajustáveis)
MAX_WORKERS_DEFAULT = 4  # Seguro para Streamlit Cloud
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
MIN_RETRY_DELAY = int(os.getenv("MIN_RETRY_DELAY", "2"))
MAX_RETRY_DELAY = int(os.getenv("MAX_RETRY_DELAY", "15"))
BATCH_SIZE_DEFAULT = int(os.getenv("BATCH_SIZE", "8"))
CACHE_TTL_DIAS = int(os.getenv("CACHE_TTL_DIAS", "5"))

# =====================================================================
# SISTEMA DE CACHE (com TTL)
# =====================================================================
class DocumentCache:
    TTL_DIAS = CACHE_TTL_DIAS  # validade do cache em dias

    def __init__(self, cache_dir=CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _cache_path(self, key: str):
        safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", key)[:240]
        return self.cache_dir / f"{safe}.pkl"

    def _is_expired(self, path: Path):
        try:
            idade = time.time() - path.stat().st_mtime
            ttl_seg = self.TTL_DIAS * 86400
            return idade > ttl_seg
        except Exception:
            return False

    def get_cache_key_file(self, file_bytes: bytes, prompt: str):
        content_hash = hashlib.md5(file_bytes).hexdigest()
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        return f"{content_hash}_{prompt_hash}"

    def get(self, key):
        f = self._cache_path(key)
        if f.exists():
            if self._is_expired(f):
                try:
                    f.unlink()
                except Exception:
                    pass
                return None
            try:
                with open(f, "rb") as h:
                    return pickle.load(h)
            except Exception:
                return None
        return None

    def set(self, key, data):
        f = self._cache_path(key)
        try:
            with open(f, "wb") as h:
                pickle.dump(data, h)
        except Exception:
            pass

    def clear(self):
        for f in self.cache_dir.glob("*.pkl"):
            try:
                f.unlink()
            except Exception:
                pass


document_cache = DocumentCache()

# =====================================================================
# GERENCIAMENTO DE PADRÕES (Persistência Local)
# =====================================================================
def load_patterns():
    if not PATTERNS_FILE.exists():
        default = {
            "COMPANHIA DE AGUA E ESGOTOS DA PARAIBA": "CAGEPA",
            "CIA DE AGUA E ESGOTO DO CEARA": "CAGECE",
            "COMPANHIA DE AGUAS E ESGOTOS DO RN": "CAERN",
            "PETROLEO BRASILEIRO": "PETROBRAS",
            "NEOENERGIA": "NEOENERGIA",
            "EQUATORIAL": "EQUATORIAL",
        }
        save_patterns(default)
        return default
    try:
        with open(PATTERNS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_patterns(pats: dict):
    try:
        with open(PATTERNS_FILE, "w", encoding="utf-8") as f:
            json.dump(pats, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


PATTERNS = load_patterns()


def normalize_pattern_key(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII")
    s = re.sub(r"[^A-Z0-9 ]+", " ", s.upper())
    return re.sub(r"\s+", " ", s).strip()


def add_pattern(raw, sub):
    raw, sub = raw.strip(), sub.strip()
    if not raw or not sub:
        return False, "Campos vazios."
    PATTERNS[raw] = sub
    save_patterns(PATTERNS)
    return True, "Adicionado."


def remove_pattern(raw):
    if raw in PATTERNS:
        PATTERNS.pop(raw)
        save_patterns(PATTERNS)
        return True, "Removido."
    return False, "Não encontrado."


# =====================================================================
# NORMALIZAÇÃO E LIMPEZA DE TEXTO
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

    # Regra especial Sabará (exemplo)
    if "SABARA" in nome_norm:
        return f"SB_{cidade_norm.split()[0]}" if cidade_norm else "SB"

    # Busca no dicionário de padrões (do maior para o menor para evitar conflitos parciais)
    for padrao in sorted(PATTERNS.keys(), key=len, reverse=True):
        if _normalizar_texto(padrao) in nome_norm:
            return PATTERNS[padrao]

    return re.sub(r"\s+", "_", nome_norm)


def limpar_emitente(nome: str) -> str:
    if not nome:
        return "SEM_NOME"
    nome = re.sub(r"[^A-Z0-9_]+", "_", nome.upper())  # Remove chars especiais
    return re.sub(r"_+", "_", nome).strip("_")


def limpar_numero(numero: str) -> str:
    if not numero:
        return "0"
    n = re.sub(r"[^\d]", "", str(numero))
    return n.lstrip("0") or "0"


# =====================================================================
# CORREÇÃO CRÍTICA: EXTRAÇÃO DE JSON ROBUSTA
# =====================================================================
def extrair_json_seguro(texto: str):
    """Tenta extrair JSON válido de respostas 'chatas' do LLM usando Regex."""
    if texto is None:
        return []
    texto = texto.strip()

    # 1. Tenta encontrar um bloco de Array [...], preferencialmente mais externo
    match_array = re.search(r'\[.*\]', texto, re.DOTALL)
    if match_array:
        try:
            return json.loads(match_array.group())
        except Exception:
            pass  # Falha no parse, tenta fallback

    # 2. Tenta encontrar um bloco de Objeto {...} (caso retorne só um)
    match_obj = re.search(r'\{.*\}', texto, re.DOTALL)
    if match_obj:
        try:
            return [json.loads(match_obj.group())]  # Encapsula em lista
        except Exception:
            pass

    # 3. Fallback: limpeza de markdown simples e tentativa direta
    clean = texto.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(clean)
        # Se for dict -> encapsula
        if isinstance(parsed, dict):
            return [parsed]
        return parsed
    except Exception:
        # último recurso: tentar extrair objetos sequenciais com regex simples
        objs = re.findall(r'\{[^{}]+\}', clean)
        results = []
        for o in objs:
            try:
                results.append(json.loads(o))
            except Exception:
                continue
        return results


# =====================================================================
# CONFIGURAÇÃO GEMINI
# =====================================================================
# Prioridade: st.secrets > Variável de Ambiente
if hasattr(st, "secrets") and st.secrets.get("GOOGLE_API_KEY"):
    GEMINI_API_KEY = st.secrets["GOOGLE_API_KEY"]
else:
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    st.error("❌ Chave GOOGLE_API_KEY não encontrada. Configure no .env ou st.secrets.")
    st.stop()

try:
    genai.configure(api_key=GEMINI_API_KEY)
    # Modelo flash é mais rápido e barato para OCR simples
    model = genai.GenerativeModel(
        "models/gemini-2.0-flash-exp" if os.getenv("USE_BETA") else "models/gemini-1.5-flash"
    )
    st.sidebar.success("✅ Gemini Ativo")
except Exception as e:
    st.error(f"❌ Erro Gemini: {str(e)}")
    st.stop()


# =====================================================================
# LÓGICA DE PROCESSAMENTO (WORKER)
# =====================================================================
def calcular_backoff(tentativa, erro_str):
    """Calcula tempo de espera baseado no erro."""
    try:
        base = MIN_RETRY_DELAY * (tentativa + 1)
        es = (erro_str or "").lower()
        if "429" in es or "resourceexhausted" in es or "rate" in es:
            return min(base + 5, MAX_RETRY_DELAY)
        return min(base, MAX_RETRY_DELAY)
    except Exception:
        return MIN_RETRY_DELAY


def processar_arquivo_worker(arquivo_dados, use_cache, batch_size=BATCH_SIZE_DEFAULT):
    """
    Função isolada para processar UM arquivo PDF inteiro.
    Executada dentro de uma Thread.
    Retorna dicionário com 'name', 'results' (lista por página), 'bytes_originais', 'cached' (bool) e 'logs' (lista).
    """
    fname = arquivo_dados["name"]
    file_bytes = arquivo_dados["bytes"]
    logs_local = []

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        num_pages = len(reader.pages)
    except Exception as e:
        return {"error": f"PDF Corrompido: {str(e)}", "name": fname, "results": [], "bytes_originais": file_bytes, "logs": [f"PDF corrompido: {e}"]}

    # Prompt otimizado (batch)
    prompt_batch = (
        "Analise as páginas da nota fiscal. Para cada página extraia: pagina (numero da página na ordem do input, começando em 1), "
        "emitente (nome da empresa/fornecedor ou null), numero_nota (apenas dígitos ou null), cidade (município ou null). "
        "Se não achar um campo, retorne null nesse campo. Retorne APENAS um JSON ARRAY: "
        "[{\"pagina\": 1, \"emitente\": \"...\", \"numero_nota\": \"...\", \"cidade\": \"...\"}, ...]"
    )

    # Cache global por arquivo + prompt
    cache_key = document_cache.get_cache_key_file(file_bytes, prompt_batch)
    if use_cache:
        cached_data = document_cache.get(cache_key)
        if cached_data:
            return {"name": fname, "results": cached_data, "bytes_originais": file_bytes, "cached": True, "logs": ["cache_hit"]}

    page_results = [None] * num_pages

    current_page = 0
    while current_page < num_pages:
        end_page = min(current_page + batch_size, num_pages)
        batch_images = []
        indices_lote = []

        # montar lote
        for i in range(current_page, end_page):
            buf = io.BytesIO()
            w = PdfWriter()
            w.add_page(reader.pages[i])
            w.write(buf)
            batch_images.append({"mime_type": "application/pdf", "data": buf.getvalue()})
            indices_lote.append(i)

        # tentativa do lote com retry
        sucesso_lote = False
        extracted_data = []
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = model.generate_content(
                    [prompt_batch] + batch_images,
                    generation_config={"response_mime_type": "application/json"},
                    request_options={"timeout": 90},
                )
                extracted_data = extrair_json_seguro(response.text)
                sucesso_lote = True
                logs_local.append(f"{fname} lote {current_page+1}-{end_page} OK (attempt {attempt})")
                break
            except Exception as e:
                back = calcular_backoff(attempt, str(e))
                logs_local.append(f"{fname} lote {current_page+1}-{end_page} erro: {e} -> backoff {back}s (attempt {attempt})")
                time.sleep(back)

        # mapeamento perfeito das respostas do lote
        if sucesso_lote and isinstance(extracted_data, list) and len(extracted_data) > 0:
            mapped = False
            # 1) tenta mapear pelo campo 'pagina'
            for dados in extracted_data:
                if not isinstance(dados, dict):
                    continue
                if "pagina" in dados:
                    try:
                        pag_json = int(dados["pagina"]) - 1
                        if pag_json in indices_lote:
                            # se já houver dados naquele index, devemos agregar (não sobrescrever)
                            existing = page_results[pag_json]
                            if existing and isinstance(existing, dict):
                                # merge/aggregate: prefer números e emitente não-nulo
                                merged = existing.copy()
                                for k, v in dados.items():
                                    if v and (not merged.get(k)):
                                        merged[k] = v
                                page_results[pag_json] = merged
                            else:
                                page_results[pag_json] = dados
                            mapped = True
                    except Exception:
                        continue

            # 2) se nada mapeou, mapear pela ordem de aparição
            if not mapped:
                for idx_relativo, dados_pagina in enumerate(extracted_data):
                    if idx_relativo < len(indices_lote):
                        real_idx = indices_lote[idx_relativo]
                        existing = page_results[real_idx]
                        if existing and isinstance(existing, dict):
                            merged = existing.copy()
                            if isinstance(dados_pagina, dict):
                                for k, v in dados_pagina.items():
                                    if v and (not merged.get(k)):
                                        merged[k] = v
                                page_results[real_idx] = merged
                        else:
                            page_results[real_idx] = dados_pagina

        else:
            if not sucesso_lote:
                logs_local.append(f"{fname} lote {current_page+1}-{end_page} falhou todas tentativas.")
            else:
                logs_local.append(f"{fname} lote {current_page+1}-{end_page} retornou vazio/malformed JSON.")

        # FALLBACK INDIVIDUAL para páginas que ficaram sem resultado
        for idx in indices_lote:
            if page_results[idx] is None:
                # tentar processamento por página com retry
                pagina_buf = io.BytesIO()
                w_single = PdfWriter()
                w_single.add_page(reader.pages[idx])
                w_single.write(pagina_buf)

                prompt_single = (
                    "Extraia emitente, numero_nota e cidade desta única página. "
                    "Retorne APENAS um JSON: {\"emitente\": \"...\", \"numero_nota\": \"...\", \"cidade\": \"...\"} "
                    "Se não achar, use null."
                )

                single_ok = False
                for attempt in range(MAX_RETRIES + 1):
                    try:
                        resp_single = model.generate_content(
                            [prompt_single, {"mime_type": "application/pdf", "data": pagina_buf.getvalue()}],
                            generation_config={"response_mime_type": "application/json"},
                            request_options={"timeout": 60},
                        )
                        dados_single_arr = extrair_json_seguro(resp_single.text)
                        # extrair_json_seguro pode retornar lista; pega primeiro dict
                        if isinstance(dados_single_arr, list) and len(dados_single_arr) >= 1:
                            dados_single = dados_single_arr[0]
                        elif isinstance(dados_single_arr, dict):
                            dados_single = dados_single_arr
                        else:
                            dados_single = None

                        if isinstance(dados_single, dict):
                            # adiciona campo 'pagina' para consistência
                            try:
                                dados_single["pagina"] = idx + 1
                            except Exception:
                                pass
                            page_results[idx] = dados_single
                            logs_local.append(f"{fname} página {idx+1} fallback OK (attempt {attempt})")
                            single_ok = True
                            break
                        else:
                            logs_local.append(f"{fname} página {idx+1} fallback retornou vazio (attempt {attempt})")
                    except Exception as e:
                        back = calcular_backoff(attempt, str(e))
                        logs_local.append(f"{fname} página {idx+1} fallback erro: {e} -> backoff {back}s (attempt {attempt})")
                        time.sleep(back)

                if not single_ok:
                    page_results[idx] = None
                    logs_local.append(f"{fname} página {idx+1} fallback FAILED após retries.")

        current_page += batch_size

    # salvar no cache
    if use_cache:
        try:
            document_cache.set(cache_key, page_results)
        except Exception:
            logs_local.append("Erro ao gravar cache (ignorado).")

    return {"name": fname, "results": page_results, "bytes_originais": file_bytes, "cached": False, "logs": logs_local}


# =====================================================================
# INTERFACE SIDEBAR
# =====================================================================
with st.sidebar:
    st.header("⚙️ Configuração")
    use_cache = st.checkbox("Usar Cache", value=True)

    st.subheader("Performance")
    workers = st.slider("Processos Simultâneos", 1, 8, MAX_WORKERS_DEFAULT, help="Aumente se tiver muitos arquivos.")

    st.markdown("---")
    st.subheader("📝 Padrões de Renomeação")

    with st.expander("Ver/Adicionar Padrões"):
        new_k = st.text_input("Texto na Nota (ex: COMPANHIA XYZ)")
        new_v = st.text_input("Substituto (ex: XYZ)")
        if st.button("➕ Adicionar"):
            ok, msg = add_pattern(new_k, new_v)
            if ok:
                st.success(msg)
            else:
                st.warning(msg)
            st.rerun()

        st.write("---")
        for k, v in PATTERNS.items():
            col_del, col_txt = st.columns([1, 4])
            with col_del:
                if st.button("🗑️", key=f"del_{k}"):
                    remove_pattern(k)
                    st.rerun()
            with col_txt:
                st.code(f"{k} -> {v}", language="text")

    st.markdown("---")
    if st.button("🧹 Limpar Cache"):
        document_cache.clear()
        st.success("Cache limpo.")


# =====================================================================
# MAIN APP
# =====================================================================

# Aviso de Volatilidade (Crítico para Cloud)
st.markdown(
    """
    <div class="warning-box">
        <b>⚠️ Atenção (Modo Nuvem):</b> Arquivos e Padrões novos são temporários. 
        Se reiniciar a página, <b>exporte seus padrões</b> no final da página para não perdê-los.
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Split de páginas (UI) ---
st.markdown("### ✂️ Separação de páginas (Split)")
split_pdf = st.file_uploader("PDF para separar páginas", type=["pdf"], key="split_pdf")
if split_pdf:
    pages_to_extract = st.text_input("Páginas (ex: 1,2,5-8):", key="split_pages_input")
    if st.button("Separar", key="split_btn"):
        try:
            reader = PdfReader(split_pdf)
            n_pages = len(reader.pages)
            pages_list = []
            for part in pages_to_extract.split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    parts = part.split("-")
                    if len(parts) == 2:
                        a = int(parts[0])
                        b = int(parts[1])
                        pages_list.extend(list(range(a, b + 1)))
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
                out = io.BytesIO()
                writer.write(out)
                out.seek(0)
                st.download_button("⬇️ Baixar PDF Separado", out, "separado.pdf", mime="application/pdf")
                st.success(f"Arquivo gerado ({added} páginas).")
        except Exception as e:
            st.error(f"Erro: {e}")

# --- Agrupamento manual (UI) ---
st.markdown("### 🔗 Agrupar PDFs manualmente")
group_files = st.file_uploader("Selecione PDFs a combinar", type=["pdf"], accept_multiple_files=True, key="group_files")
if group_files:
    if st.button("Agrupar PDFs", key="btn_group"):
        try:
            writer = PdfWriter()
            for f in group_files:
                r_g = PdfReader(f)
                for p in r_g.pages:
                    writer.add_page(p)
            out = io.BytesIO()
            writer.write(out)
            out.seek(0)
            st.download_button("⬇️ Baixar PDF Agrupado", out, "agrupado.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"Erro ao agrupar: {e}")

# --- Uploads e processamento ---
uploaded_files = st.file_uploader("Selecione seus PDFs", type=["pdf"], accept_multiple_files=True)

col_act1, col_act2 = st.columns([1, 4])
with col_act1:
    btn_processar = st.button("🚀 Processar Tudo", type="primary")
with col_act2:
    if st.button("♻️ Resetar"):
        st.session_state.clear()
        st.rerun()

if uploaded_files and btn_processar:
    session_id = str(uuid.uuid4())
    session_path = TEMP_FOLDER / session_id
    session_path.mkdir(exist_ok=True)

    st.session_state["session_path"] = str(session_path)

    # Preparar dados para threads
    files_data = []
    for f in uploaded_files:
        files_data.append({"name": f.name, "bytes": f.read()})

    total_files = len(files_data)
    completed = 0
    prog_bar = st.progress(0)
    status_txt = st.empty()

    all_results_grouped = {}  # Chave: (numero, emitente), Valor: [paginas...]
    logs_processamento = []

    start_time = time.time()

    # --- PARALELISMO REAL ---
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(processar_arquivo_worker, f, use_cache, BATCH_SIZE_DEFAULT): f["name"] for f in files_data}

        for future in as_completed(futures):
            f_name = futures[future]
            completed += 1
            prog_bar.progress(completed / total_files)
            status_txt.text(f"Processando {completed}/{total_files}: {f_name}")

            try:
                res = future.result()

                if "error" in res:
                    logs_processamento.append(f"❌ {f_name}: {res['error']}")
                    # também inclui logs locais se houver
                    if res.get("logs"):
                        logs_processamento.extend([f"{f_name} LOG: {l}" for l in res["logs"]])
                    continue

                # Processar resultados da thread
                original_bytes = res["bytes_originais"]
                paginas_res = res["results"] or []
                logs_local = res.get("logs", [])
                if logs_local:
                    logs_processamento.extend([f"{f_name} LOG: {l}" for l in logs_local])

                # Abrir reader novamente apenas para extração
                reader_local = PdfReader(io.BytesIO(original_bytes))

                for idx_pg, dados_pg in enumerate(paginas_res):
                    if not dados_pg:
                        logs_processamento.append(f"⚠️ {f_name} Pág {idx_pg+1}: Sem dados IA")
                        continue

                    # Normalização
                    raw_emit = (dados_pg.get("emitente") or "") if isinstance(dados_pg, dict) else ""
                    raw_num = (dados_pg.get("numero_nota") or "") if isinstance(dados_pg, dict) else ""
                    raw_cid = (dados_pg.get("cidade") or "") if isinstance(dados_pg, dict) else ""

                    if not raw_emit and not raw_num:
                        # ignora se nada útil
                        continue

                    emit_final = limpar_emitente(substituir_nome_emitente(raw_emit, raw_cid))
                    num_final = limpar_numero(raw_num)

                    key = (num_final, emit_final)
                    if key not in all_results_grouped:
                        all_results_grouped[key] = []

                    # Guarda buffer da página específica
                    try:
                        buf_pg = io.BytesIO()
                        w_pg = PdfWriter()
                        # garante que página existe
                        if idx_pg < len(reader_local.pages):
                            w_pg.add_page(reader_local.pages[idx_pg])
                            w_pg.write(buf_pg)
                            all_results_grouped[key].append({
                                "data": buf_pg.getvalue(),
                                "origem": f_name,
                                "pagina_origem": idx_pg + 1
                            })
                        else:
                            logs_processamento.append(f"⚠️ {f_name} página {idx_pg+1} index out of range (ignorado).")
                    except Exception as e:
                        logs_processamento.append(f"❌ Erro ao extrair página {idx_pg+1} de {f_name}: {e}")

            except Exception as e:
                logs_processamento.append(f"❌ Erro fatal em {f_name}: {str(e)}")

    # --- GERAÇÃO DOS ARQUIVOS FINAIS (agrupando corretamente) ---
    status_txt.text("Gerando arquivos finais...")

    final_files_meta = []

    for (num, emit), paginas in all_results_grouped.items():
        if num == "0":
            continue

        # ordenar páginas por origem->pagina_origem para manter sequência natural
        try:
            paginas_sorted = sorted(paginas, key=lambda x: (x.get("origem", ""), x.get("pagina_origem", 0)))
        except Exception:
            paginas_sorted = paginas

        writer_final = PdfWriter()
        for p in paginas_sorted:
            try:
                r_temp = PdfReader(io.BytesIO(p["data"]))
                writer_final.add_page(r_temp.pages[0])
            except Exception:
                continue

        base_name = f"DOC {num}_{emit}"
        nome_arq = base_name + ".pdf"
        caminho_final = session_path / nome_arq

        # Se já existir → acrescenta sufixo incremental (não sobrescreve)
        contador = 2
        while caminho_final.exists():
            nome_arq = f"{base_name}_{contador}.pdf"
            caminho_final = session_path / nome_arq
            contador += 1

        try:
            with open(caminho_final, "wb") as f_out:
                writer_final.write(f_out)

            final_files_meta.append({
                "file_name": nome_arq,
                "path": str(caminho_final),
                "pages": len(paginas_sorted),
                "origem": ", ".join(sorted(set([p["origem"] for p in paginas_sorted])))
            })
        except Exception as e:
            logs_processamento.append(f"❌ Erro ao escrever {nome_arq}: {e}")

    st.session_state["final_results"] = final_files_meta
    st.session_state["logs"] = logs_processamento

    tempo_total = round(time.time() - start_time, 2)
    st.success(f"Concluído em {tempo_total}s! {len(final_files_meta)} documentos gerados.")
    st.rerun()

# =====================================================================
# ÁREA DE RESULTADOS (PÓS-PROCESSAMENTO)
# =====================================================================
if "final_results" in st.session_state:
    st.markdown("---")
    st.header("📂 Arquivos Gerados")

    results = st.session_state["final_results"]
    logs = st.session_state.get("logs", [])

    # Dashboard Mini
    c1, c2 = st.columns(2)
    c1.metric("Documentos", len(results))
    c2.metric("Erros/Alertas", len(logs))

    if logs:
        with st.expander("Ver Logs de Erro"):
            for l in logs:
                st.write(l)

    # Lista de Arquivos
    for idx, item in enumerate(results):
        fpath = Path(item["path"])
        if not fpath.exists():
            continue

        with st.container():
            st.markdown(
                f"""<div class="card"><b>📄 {item['file_name']}</b><br>
            <span class="small-note">Origem: {item['origem']} | Páginas: {item['pages']}</span></div>""",
                unsafe_allow_html=True,
            )

            col_d1, col_d2, col_d3 = st.columns([2, 2, 1])

            # Renomear
            with col_d1:
                new_name = st.text_input("Renomear", value=item["file_name"], key=f"ren_{idx}", label_visibility="collapsed")

            with col_d2:
                if st.button("💾 Salvar Nome", key=f"btn_ren_{idx}"):
                    try:
                        new_path = fpath.parent / new_name
                        if not new_name.lower().endswith(".pdf"):
                            new_path = fpath.parent / (new_name + ".pdf")
                        fpath.rename(new_path)
                        item["path"] = str(new_path)
                        item["file_name"] = new_path.name
                        st.success("Renomeado!")
                        time.sleep(0.5)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erro: {e}")

            # Download
            with col_d3:
                with open(item["path"], "rb") as f:
                    st.download_button("⬇️ Baixar", f, file_name=item["file_name"], key=f"dl_{idx}")

    # =====================================================================
    # EXPORTAÇÃO E DOWNLOAD EM MASSA
    # =====================================================================
    st.markdown("---")
    st.subheader("📦 Download em Massa & Backup")

    c_zip, c_patterns = st.columns(2)

    with c_zip:
        # ZIP de todos os PDFs
        if st.button("Compactar Tudo (ZIP)"):
            mem_zip = io.BytesIO()
            with zipfile.ZipFile(mem_zip, "w") as zf:
                for item in results:
                    p = Path(item["path"])
                    if p.exists():
                        zf.write(p, arcname=p.name)
            mem_zip.seek(0)
            st.download_button("⬇️ Baixar ZIP Completo", mem_zip, "notas_processadas.zip", "application/zip")

    with c_patterns:
        # Backup dos Padrões (Crucial para Cloud)
        patterns_str = json.dumps(PATTERNS, indent=2, ensure_ascii=False)
        st.download_button("📤 Exportar Padrões (.json)", patterns_str, "meus_padroes.json", "application/json")

        # Importar
        uploaded_pat = st.file_uploader("Importar Padrões Salvos", type=["json"])
        if uploaded_pat:
            try:
                loaded = json.load(uploaded_pat)
                # merge, sem sobrescrever conflitos se preferir pode validar
                PATTERNS.update(loaded)
                save_patterns(PATTERNS)
                st.success("Padrões importados! A página irá recarregar.")
                time.sleep(1)
                st.rerun()
            except Exception:
                st.error("JSON inválido.")

# --- garantir que patterns estejam salvos ---
try:
    save_patterns(PATTERNS)
except Exception:
    pass

# =====================================================================
# FIM DO ARQUIVO — TURBO v5 (Refined, com correções)
# =====================================================================
