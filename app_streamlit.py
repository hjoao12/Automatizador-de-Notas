# =========================
# Parte 1/3 - setup & utils
# =========================

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
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader, PdfWriter
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

# ---------- Basic logging ----------
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("nf_automator")

# =====================================================================
# CONFIGURAÇÃO INICIAL
# =====================================================================
load_dotenv()
st.set_page_config(page_title="Automatizador de Notas Fiscais", page_icon="icone.ico")

# Light CSS (mantive seu estilo mas isolado)
st.markdown(
    """
    <style>
    /* (estilos originais aqui — omitidos por brevidade no snippet) */
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Automatizador de Notas Fiscais PDF")

# =====================================================================
# CONSTANTES E PASTAS
# =====================================================================
BASE_DIR = Path(".").resolve()
TEMP_FOLDER = BASE_DIR / "temp"
CACHE_FOLDER = BASE_DIR / "cache"
PATTERNS_FILE = BASE_DIR / "patterns.json"

TEMP_FOLDER.mkdir(exist_ok=True)
CACHE_FOLDER.mkdir(exist_ok=True)

# Carregar variáveis de ambiente com valores padrões razoáveis
MAX_TOTAL_PAGES = int(os.getenv("MAX_TOTAL_PAGES", "500"))  # limite maior por segurança
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
MIN_RETRY_DELAY = int(os.getenv("MIN_RETRY_DELAY", "3"))
MAX_RETRY_DELAY = int(os.getenv("MAX_RETRY_DELAY", "30"))
DEFAULT_MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))

# =====================================================================
# SISTEMA DE CACHE (SIMPLES E SEGURO)
# =====================================================================
class DocumentCache:
    def __init__(self, cache_dir: Path = CACHE_FOLDER):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, key: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_\-\.]", "_", key)[:240]
        return self.cache_dir / f"{safe}.pkl"

    def get_cache_key(self, pdf_bytes: bytes, prompt: str) -> str:
        """Gera chave única baseada no conteúdo do PDF e prompt"""
        content_hash = hashlib.md5(pdf_bytes).hexdigest()
        prompt_hash = hashlib.md5(prompt.encode("utf-8")).hexdigest()
        return f"{content_hash}_{prompt_hash}"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            p = self._cache_path(key)
            if not p.exists():
                return None
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Cache read error for key {key}: {e}")
            return None

    def set(self, key: str, data: Dict[str, Any]) -> None:
        try:
            p = self._cache_path(key)
            with open(p, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.warning(f"Cache write error for key {key}: {e}")

    def clear(self) -> None:
        for f in self.cache_dir.glob("*.pkl"):
            try:
                f.unlink()
            except Exception:
                pass

document_cache = DocumentCache()

# =====================================================================
# PERSISTÊNCIA / PADRÕES (patterns.json)
# =====================================================================
def load_patterns() -> Dict[str, str]:
    default = {
        "COMPANHIA DE AGUA E ESGOTOS DA PARAIBA": "CAGEPA",
        "COMPANHIA DE AGUA E ESGOTOS DA PARAÍBA": "CAGEPA",
        "CIA DE AGUA E ESGOTO DO CEARA": "CAGECE",
        "COMPANHIA DE AGUAS E ESGOTOS DO RN": "CAERN",
        "PETRÓLEO BRASILEIRO S.A": "PETROBRAS",
        "PETROLEO BRASILEIRO S.A": "PETROBRAS",
        "NEOENERGIA": "NEOENERGIA",
        "EQUATORIAL": "EQUATORIAL",
    }
    try:
        if not PATTERNS_FILE.exists():
            save_patterns(default)
            return default
        with open(PATTERNS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            return default
    except Exception as e:
        logger.warning(f"Erro ao carregar patterns.json: {e}")
        return default

def save_patterns(patterns: Dict[str, str]) -> bool:
    try:
        with open(PATTERNS_FILE, "w", encoding="utf-8") as f:
            json.dump(patterns, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"Erro ao salvar padrões: {e}")
        return False

SUBSTITUICOES_FIXAS = load_patterns()

# =====================================================================
# NORMALIZAÇÃO / LIMPEZA
# =====================================================================
def _normalizar_texto(s: Optional[str]) -> str:
    if not s:
        return ""
    s2 = unicodedata.normalize("NFKD", s)
    s2 = s2.encode("ASCII", "ignore").decode("ASCII")
    s2 = s2.upper()
    s2 = re.sub(r"[^A-Z0-9 ]+", " ", s2)
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2

def substituir_nome_emitente(nome_raw: Optional[str], cidade_raw: Optional[str] = None) -> str:
    nome_norm = _normalizar_texto(nome_raw)
    cidade_norm = _normalizar_texto(cidade_raw) if cidade_raw else None
    if "SABARA" in nome_norm:
        return f"SB_{(cidade_norm.split()[0] if cidade_norm else '')}".strip("_")
    for padrao, substituto in SUBSTITUICOES_FIXAS.items():
        if _normalizar_texto(padrao) in nome_norm:
            return substituto
    # fallback: substituir espaços por _
    return re.sub(r"\s+", "_", nome_norm) or "SEM_NOME"

def limpar_emitente(nome: Optional[str]) -> str:
    if not nome:
        return "SEM_NOME"
    s = unicodedata.normalize("NFKD", nome).encode("ASCII", "ignore").decode("ASCII")
    s = s.upper()
    s = re.sub(r"[^A-Z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "SEM_NOME"

def limpar_numero(numero: Optional[str]) -> str:
    if not numero:
        return "0"
    s = re.sub(r"[^\d]", "", str(numero))
    return s.lstrip("0") or "0"

# =====================================================================
# VALIDAÇÃO E CORREÇÃO DOS DADOS EXTRAÍDOS PELA IA
# =====================================================================
def validar_e_corrigir_dados(dados: Any) -> Dict[str, str]:
    """
    Recebe a saída (geralmente dict) da IA e normaliza/valida.
    Garante chaves: emitente, numero_nota, cidade
    """
    if not isinstance(dados, dict):
        dados = {}

    required = ["emitente", "numero_nota", "cidade"]
    for k in required:
        val = dados.get(k)
        if not val:
            dados[k] = "NÃO_IDENTIFICADO"

    # Correções conhecidas (exemplo)
    correcoes = {
        "CPFL ENERGIA": "CPFL",
        "COMPANHIA PAULISTA DE FORCA E LUZ": "CPFL",
    }
    if isinstance(dados.get("emitente"), str):
        em = dados["emitente"].upper()
        for inc, corr in correcoes.items():
            if inc in em:
                dados["emitente"] = corr
                break

    # limpar numero
    if "numero_nota" in dados:
        numero_limpo = re.sub(r"[^\d]", "", str(dados["numero_nota"]))
        dados["numero_nota"] = numero_limpo or "000000"

    return dados

# =====================================================================
# CONFIGURAÇÃO DO GEMINI (Google)
# =====================================================================
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ Chave GOOGLE_API_KEY não encontrada. Defina em .env")
    st.stop()

try:
    genai.configure(api_key=GEMINI_API_KEY)
    model_name = os.getenv("MODEL_NAME", "models/gemini-2.5-flash")
    model = genai.GenerativeModel(model_name)
    st.sidebar.success("✅ Gemini configurado")
except Exception as e:
    st.error(f"❌ Erro ao configurar Gemini: {e}")
    st.stop()

# =====================================================================
# FUNÇÕES DE PROCESSAMENTO COM RETRY (GEMINI)
# =====================================================================
def calcular_delay(tentativa: int, error_msg: str) -> float:
    text = (error_msg or "").lower()
    if "retry in" in text:
        m = re.search(r"retry in (\d+\.?\d*)s", text)
        if m:
            try:
                return min(float(m.group(1)) + 2.0, MAX_RETRY_DELAY)
            except Exception:
                pass
    return min(MIN_RETRY_DELAY * (tentativa + 1), MAX_RETRY_DELAY)

def _safe_parse_json_from_response_text(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Tenta extrair JSON de uma resposta que pode estar envolta em ```json``` ou ter ruído.
    Retorna (dict_or_none, raw_text_preview_on_error)
    """
    if not text:
        return None, None
    txt = text.strip()
    # remover fences
    txt = re.sub(r"^```json\s*", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"```\s*$", "", txt)
    # procurar primeiro bloco JSON
    try:
        parsed = json.loads(txt)
        return parsed, None
    except Exception:
        # tentar encontrar o primeiro {...} ou [ ... ]
        m = re.search(r"(\{.*\}|\[.*\])", txt, flags=re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(1))
                return parsed, None
            except Exception as e:
                return None, str(txt)[:400]
        return None, str(txt)[:400]

def processar_pagina_gemini(prompt_instrucao: str, page_bytes: bytes, timeout: int = 60) -> Tuple[Dict[str, Any], bool, float, str]:
    """
    Envia bytes de uma página ao Gemini, com retry e tratamento de erros.
    Retorna: (dados, ok_bool, tempo_segundos, provider_str)
    """
    last_error = ""
    for tentativa in range(MAX_RETRIES + 1):
        try:
            start = time.time()
            # O model.generate_content espera os inputs conforme SDK
            resp = model.generate_content(
                [prompt_instrucao, {"mime_type": "application/pdf", "data": page_bytes}],
                generation_config={"response_mime_type": "application/json"},
                request_options={"timeout": timeout}
            )
            elapsed = time.time() - start

            texto = getattr(resp, "text", "") or ""
            dados_parsed, raw_preview = _safe_parse_json_from_response_text(texto)
            if dados_parsed is None:
                return ({"error": "Resposta não era JSON válido", "_raw": raw_preview or texto[:400]}, False, round(elapsed, 2), "Gemini")
            return (dados_parsed, True, round(elapsed, 2), "Gemini")
        except ResourceExhausted as e:
            last_error = str(e)
            delay = calcular_delay(tentativa, last_error)
            logger.warning(f"Quota/ResourceExhausted (tentativa {tentativa+1}): {e} — aguardando {delay}s")
            time.sleep(delay)
        except Exception as e:
            last_error = str(e)
            logger.warning(f"Erro Gemini (tentativa {tentativa+1}): {e}")
            if tentativa < MAX_RETRIES:
                time.sleep(MIN_RETRY_DELAY)
            else:
                return ({"error": last_error}, False, 0.0, "Gemini")
    return ({"error": "Falha máxima de tentativas"}, False, 0.0, "Gemini")

def processar_pagina_worker(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Função executada em ThreadPoolExecutor. job_data deve conter:
     - bytes: bytes da página
     - prompt: prompt string
     - name: arquivo origem
     - page_idx: índice
     - use_cache: bool
    Retorna um dicionário padronizado com status/tempo/dados/erro.
    """
    try:
        pdf_bytes = job_data["bytes"]
        prompt = job_data["prompt"]
        name = job_data.get("name", "sem_nome")
        page_idx = job_data.get("page_idx", 0)
        use_cache = bool(job_data.get("use_cache", True))
    except Exception as e:
        return {"status": "ERRO", "error_msg": f"Job inválido: {e}"}

    cache_key = document_cache.get_cache_key(pdf_bytes, prompt)
    if use_cache:
        cached = document_cache.get(cache_key)
        if cached:
            return {
                "status": "CACHE",
                "dados": cached.get("dados"),
                "tempo": cached.get("tempo", 0.0),
                "provider": cached.get("provider", "cache"),
                "name": name,
                "page_idx": page_idx,
                "pdf_bytes": pdf_bytes
            }

    dados, ok, tempo, provider = processar_pagina_gemini(prompt, pdf_bytes)
    if ok and isinstance(dados, dict) and "error" not in dados:
        # salvar cache
        document_cache.set(cache_key, {"dados": dados, "tempo": tempo, "provider": provider})
        return {
            "status": "OK",
            "dados": dados,
            "tempo": tempo,
            "provider": provider,
            "name": name,
            "page_idx": page_idx,
            "pdf_bytes": pdf_bytes
        }
    else:
        return {
            "status": "ERRO",
            "dados": dados,
            "tempo": tempo,
            "provider": provider,
            "name": name,
            "page_idx": page_idx,
            "error_msg": (dados.get("error") if isinstance(dados, dict) else str(dados))
        }

# End of Part 1/3
# =========================
# Parte 2/3 - upload & processing
# =========================

from typing import List, Dict, Tuple
import math

# Ajustes de UI: parâmetros e prompt
DEFAULT_PROMPT = (
    "Analise a nota fiscal (DANFE). Extraia emitente, número da nota e cidade. "
    "Responda SOMENTE em JSON: {\"emitente\":\"NOME\",\"numero_nota\":\"NUMERO\",\"cidade\":\"CIDADE\"}"
)

def preparar_arquivos(uploaded_files) -> List[Dict[str, Any]]:
    """
    Recebe uploaded_files de Streamlit e retorna lista de dicts com name/bytes.
    Filtra arquivos inválidos.
    """
    arquivos = []
    for f in uploaded_files:
        try:
            b = f.read()
            if not b:
                logger.warning(f"Arquivo vazio: {f.name}")
                continue
            arquivos.append({"name": f.name, "bytes": b})
        except Exception as e:
            logger.warning(f"Erro ao ler {getattr(f, 'name', 'unknown')}: {e}")
    return arquivos

def criar_jobs_por_pagina(arquivos: List[Dict[str, Any]], prompt: str, use_cache: bool) -> List[Dict[str, Any]]:
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
                    "prompt": prompt,
                    "name": name,
                    "page_idx": idx,
                    "use_cache": use_cache
                })
        except Exception as e:
            logger.warning(f"Erro lendo PDF {name}: {e}")
    return jobs

def executar_jobs_parallel(jobs: List[Dict[str, Any]], max_workers: int = DEFAULT_MAX_WORKERS) -> Tuple[List[Dict[str, Any]], List[Tuple]]:
    """
    Executa jobs via ThreadPoolExecutor e retorna (results, processed_logs).
    Cada result é o retorno da processar_pagina_worker.
    processed_logs é uma lista de tuplas para exibição.
    """
    results = []
    processed_logs = []
    total_jobs = len(jobs)
    if total_jobs == 0:
        return results, processed_logs

    progress_bar = st.progress(0.0)
    progresso_text = st.empty()
    processed_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {executor.submit(processar_pagina_worker, job): job for job in jobs}
        for future in as_completed(future_to_job):
            processed_count += 1
            try:
                result = future.result()
            except Exception as e:
                logger.error(f"Erro executando job: {e}")
                result = {"status": "ERRO", "error_msg": str(e), "name": future_to_job[future].get("name", ""), "page_idx": future_to_job[future].get("page_idx", 0)}
            # atualizar barra
            progress_bar.progress(min(processed_count / total_jobs, 1.0))
            # tratar resultado para logs/UI
            name = result.get("name", "desconhecido")
            idx = result.get("page_idx", 0)
            page_label = f"{name} (pág {idx+1})"
            status = result.get("status", "ERRO")
            if status == "ERRO":
                msg = result.get("error_msg", "Erro desconhecido")
                processed_logs.append((page_label, result.get("tempo", 0.0), "ERRO_IA", msg, result.get("provider", "")))
                progresso_text.markdown(f"<span class='warning-log'>⚠️ {page_label} — ERRO: {msg}</span>", unsafe_allow_html=True)
            else:
                tempo = result.get("tempo", 0.0)
                dados = result.get("dados", {})
                provider = result.get("provider", "")
                processed_logs.append((page_label, tempo, status, json.dumps(dados, ensure_ascii=False), provider))
                css_class = "success-log" if status == "OK" else "warning-log"
                progresso_text.markdown(f"<span class='{css_class}'>✅ {page_label} — {status} ({tempo:.2f}s)</span>", unsafe_allow_html=True)
            results.append(result)

    return results, processed_logs

def agrupar_e_salvar(results: List[Dict[str, Any]], session_folder: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Agrupa páginas por (numero, emitente) e escreve PDFs finais no session_folder.
    Retorna (resultados_list, files_meta_dict)
    """
    agrupados_bytes: Dict[Tuple[str, str], List[bytes]] = {}
    resultados_meta: List[Dict[str, Any]] = []

    for res in results:
        if res.get("status") == "ERRO":
            continue
        dados = res.get("dados", {})
        dados = validar_e_corrigir_dados(dados)
        emitente_raw = dados.get("emitente", "") or ""
        numero_raw = dados.get("numero_nota", "") or ""
        cidade_raw = dados.get("cidade", "") or ""

        numero = limpar_numero(numero_raw)
        nome_map = substituir_nome_emitente(emitente_raw, cidade_raw)
        emitente = limpar_emitente(nome_map)

        if not numero or numero == "0":
            # ignorar páginas sem número detectado (evita gerar DOC 0_)
            resultados_meta.append({
                "arquivo_origem": res.get("name"),
                "pagina": res.get("page_idx") + 1,
                "status": "IGNORADO_SEM_NUM",
                "emitente_raw": emitente_raw,
                "numero_raw": numero_raw
            })
            continue

        key = (numero, emitente)
        agrupados_bytes.setdefault(key, []).append(res.get("pdf_bytes"))

        resultados_meta.append({
            "arquivo_origem": res.get("name"),
            "pagina": res.get("page_idx") + 1,
            "emitente_detectado": emitente_raw,
            "numero_detectado": numero_raw,
            "status": res.get("status"),
            "tempo_s": res.get("tempo", 0.0),
            "provider": res.get("provider", "")
        })

    resultados = []
    files_meta = {}

    for (numero, emitente), pages_bytes in agrupados_bytes.items():
        writer = PdfWriter()
        page_count = 0
        for pb in pages_bytes:
            try:
                r = PdfReader(io.BytesIO(pb))
                for p in r.pages:
                    writer.add_page(p)
                    page_count += 1
            except Exception:
                logger.warning("Erro ao juntar página interna; pulando.")
                continue
        if page_count == 0:
            continue
        nome_pdf = f"DOC {numero}_{emitente}.pdf"
        caminho = session_folder / nome_pdf
        try:
            with open(caminho, "wb") as f_out:
                writer.write(f_out)
            resultados.append({"file": nome_pdf, "numero": numero, "emitente": emitente, "pages": page_count})
            files_meta[nome_pdf] = {"numero": numero, "emitente": emitente, "pages": page_count}
        except Exception as e:
            logger.error(f"Erro ao salvar {nome_pdf}: {e}")

    return resultados, files_meta, resultados_meta

# ---------------------------
# Função principal de fluxo
# ---------------------------
def fluxo_processamento(uploaded_files, use_cache: bool = True, prompt: str = DEFAULT_PROMPT, max_workers: int = DEFAULT_MAX_WORKERS):
    if not uploaded_files:
        st.warning("Nenhum arquivo selecionado.")
        return

    session_id = str(uuid.uuid4())
    session_folder = TEMP_FOLDER / session_id
    session_folder.mkdir(parents=True, exist_ok=True)
    st.session_state["session_folder"] = str(session_folder)

    arquivos = preparar_arquivos(uploaded_files)
    total_paginas = 0
    for a in arquivos:
        try:
            r = PdfReader(io.BytesIO(a["bytes"]))
            total_paginas += len(r.pages)
        except Exception:
            st.warning(f"Arquivo inválido: {a['name']}")

    st.info(f"📄 Total de páginas a processar: {total_paginas}")
    jobs = criar_jobs_por_pagina(arquivos, prompt, use_cache)

    st.info(f"🚀 Iniciando processamento TURBO: {len(jobs)} páginas com {max_workers} workers...")
    results, processed_logs = executar_jobs_parallel(jobs, max_workers=max_workers)

    # Agrupar e salvar PDFs gerados
    resultados, files_meta, resultados_meta = agrupar_e_salvar(results, session_folder)

    # Persistir estado na sessão
    st.session_state["resultados"] = resultados
    st.session_state["novos_nomes"] = {r["file"]: r["file"] for r in resultados}
    st.session_state["processed_logs"] = processed_logs
    st.session_state["files_meta"] = files_meta

    st.success(f"✅ Processamento concluído — {len(resultados)} arquivos gerados.")
    return {
        "session_folder": str(session_folder),
        "resultados": resultados,
        "files_meta": files_meta,
        "processed_logs": processed_logs,
        "resultados_meta": resultados_meta
    }

# ---------------------------
# Integração básica com Streamlit (UI minimal)
# ---------------------------

# Esses controles podem ser integrados no seu arquivo principal Streamlit:
with st.sidebar:
    st.markdown("### ⚙️ Performance")
    max_workers = st.slider("Workers simultâneos", min_value=1, max_value=8, value=DEFAULT_MAX_WORKERS, step=1)
    use_cache_side = st.checkbox("Usar cache", value=True)
    if st.button("🔄 Limpar cache (sidebar)"):
        document_cache.clear()
        st.success("Cache limpo!")

# O botão principal no UI deve chamar fluxo_processamento quando apropriado.
# Exemplo de uso (coloque no seu fluxo principal onde tiver uploaded_files e botão):
#
# if uploaded_files and process_btn:
#     fluxo_processamento(uploaded_files, use_cache=use_cache_side, prompt=DEFAULT_PROMPT, max_workers=max_workers)
#
# End of Part 2/3
# =========================
# Parte 3/3 - painel de visualização, agrupamento e gestão avançada
# =========================

from typing import Optional, List
from PIL import Image
import io
import base64

# --- Dependências opcionais para renderizar PDFs como imagens ---
# Recomendações:
# - pdf2image (requer poppler instalado no sistema) OR
# - fitz (PyMuPDF) que faz rendering sem poppler
# O código tenta usar pdf2image, cai para fitz, e se nada disponível usa um fallback (ícone).
try:
    from pdf2image import convert_from_bytes
    _RENDERER = "pdf2image"
except Exception:
    try:
        import fitz  # PyMuPDF
        _RENDERER = "pymupdf"
    except Exception:
        _RENDERER = None

# ---------------------------
# UTIL: gerar thumbnail a partir de bytes de um PDF (primeira página)
# ---------------------------
def gerar_thumbnail_from_pdf_bytes(pdf_bytes: bytes, max_width: int = 300) -> Optional[bytes]:
    """
    Retorna bytes PNG da primeira página redimensionada (thumbnail).
    Tenta pdf2image -> fitz -> retorna None se não for possível.
    """
    try:
        if _RENDERER == "pdf2image":
            pages = convert_from_bytes(pdf_bytes, first_page=1, last_page=1, fmt='png')
            if pages:
                img = pages[0]
                img.thumbnail((max_width, max_width * 4))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return buf.getvalue()
        elif _RENDERER == "pymupdf":
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            page = doc.load_page(0)
            zoom = 2  # aumenta resolução
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            img.thumbnail((max_width, max_width * 4))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        else:
            return None
    except Exception:
        return None

# ---------------------------
# UI: Painel de visualização rápida de imagens das notas
# ---------------------------
def painel_visualizacao_rapida(session_folder: Path):
    st.markdown("---")
    st.markdown("### 🔍 Visualização rápida — Thumbnails")
    resultados = st.session_state.get("resultados", [])
    if not resultados:
        st.info("Nenhum PDF gerado para visualizar.")
        return

    cols_per_row = 4
    cards = []
    for r in resultados:
        fname = r["file"]
        path = session_folder / fname
        if not path.exists():
            continue
        # carregar primeiro page bytes para gerar thumbnail
        try:
            with open(path, "rb") as f:
                pdf_bytes = f.read()
            thumb = gerar_thumbnail_from_pdf_bytes(pdf_bytes, max_width=300)
        except Exception:
            thumb = None

        cards.append((fname, thumb))

    if not cards:
        st.info("Nenhuma visualização disponível (renderer ausente ou erro).")
        if _RENDERER is None:
            st.warning("Instale `pdf2image` + poppler **ou** `PyMuPDF` (fitz) para gerar thumbnails.")
        return

    # Mostrar em grid
    rows = math.ceil(len(cards) / cols_per_row)
    idx = 0
    for _ in range(rows):
        cols = st.columns(cols_per_row)
        for c in cols:
            if idx >= len(cards):
                break
            fname, thumb = cards[idx]
            if thumb:
                c.image(thumb, caption=fname, use_column_width=True)
            else:
                c.markdown(f"**{fname}**")
                c.write("🔳 Preview não disponível")
            # botão visualizar PDF (abre em nova aba)
            with c:
                file_path = session_folder / fname
                if file_path.exists():
                    with open(file_path, "rb") as ff:
                        b = ff.read()
                    # gerar data URI para abrir em nova aba
                    b64 = base64.b64encode(b).decode()
                    href = f'<a target="_blank" href="data:application/pdf;base64,{b64}">Abrir PDF</a>'
                    c.markdown(href, unsafe_allow_html=True)
            idx += 1

# ---------------------------
# UI: Agrupar antes de baixar (seleção + ZIP)
# ---------------------------
def painel_agrupar_e_baixar(session_folder: Path):
    st.markdown("---")
    st.markdown("### 🗂️ Agrupar / Baixar")
    resultados = st.session_state.get("resultados", [])
    if not resultados:
        st.info("Nenhum arquivo para agrupar.")
        return

    options = [r["file"] for r in resultados]
    selecionadas = st.multiselect("Selecione arquivos para agrupar/baixar", options, default=options)
    novo_nome = st.text_input("Nome do ZIP (opcional)", value="notas_selecionadas.zip")
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("📦 Criar ZIP e baixar"):
            if not selecionadas:
                st.warning("Selecione pelo menos um arquivo.")
            else:
                mem = io.BytesIO()
                with zipfile.ZipFile(mem, "w") as zf:
                    for f in selecionadas:
                        src = session_folder / f
                        if src.exists():
                            arcname = st.session_state.get("novos_nomes", {}).get(f, f)
                            zf.write(src, arcname=arcname)
                mem.seek(0)
                st.download_button("⬇️ Baixar ZIP", data=mem, file_name=novo_nome, mime="application/zip")
    with col2:
        if st.button("🗑️ Mover selecionadas para lixeira"):
            if not selecionadas:
                st.warning("Selecione ao menos um arquivo.")
            else:
                count = 0
                for f in selecionadas:
                    src = session_folder / f
                    try:
                        if src.exists():
                            src.unlink()
                    except Exception:
                        pass
                    # atualizar session_state
                    st.session_state["resultados"] = [r for r in st.session_state["resultados"] if r["file"] != f]
                    st.session_state.get("files_meta", {}).pop(f, None)
                    st.session_state.get("novos_nomes", {}).pop(f, None)
                    count += 1
                st.success(f"{count} arquivo(s) removido(s).")
                st.experimental_rerun()

# ---------------------------
# Integração com painel de gestão (melhorias plug-and-play)
# ---------------------------
def painel_gerenciamento_extra():
    """
    Complementa a UI já existente com:
    - renomear em lote (aplicando padrão DOC ...)
    - exportar metadata (JSON/CSV)
    - sugestões de organização
    """
    st.markdown("---")
    st.markdown("### ⚡ Ações rápidas / Export")
    resultados = st.session_state.get("resultados", [])
    if not resultados:
        return

    col_a, col_b, col_c = st.columns([1,1,1])
    with col_a:
        if st.button("🔤 Renomear para padrão 'DOC NNNNN_EMITENTE'"):
            changed = 0
            for r in st.session_state.get("resultados", []):
                padrão = f'DOC {r.get("numero","000000")}_{r.get("emitente","SEM_NOME")}.pdf'
                st.session_state["novos_nomes"][r["file"]] = padrão
                changed += 1
            st.success(f"Renomeados {changed} entradas (visualmente).")

    with col_b:
        if st.button("📥 Exportar metadados (JSON)"):
            meta = st.session_state.get("files_meta", {})
            data = json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8")
            st.download_button("⬇️ Baixar metadata.json", data=data, file_name="metadata.json", mime="application/json")

    with col_c:
        if st.button("📤 Exportar lista p/ Excel (CSV)"):
            import csv, pandas as pd
            rows = []
            for r in st.session_state.get("resultados", []):
                fname = r["file"]
                meta = st.session_state.get("files_meta", {}).get(fname, {})
                rows.append({
                    "file": fname,
                    "numero": meta.get("numero", r.get("numero","")),
                    "emitente": meta.get("emitente", r.get("emitente","")),
                    "pages": meta.get("pages", r.get("pages", 1))
                })
            df = pd.DataFrame(rows)
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Baixar CSV", data=csv_bytes, file_name="notas_list.csv", mime="text/csv")

# ---------------------------
# Render: integrar painéis no fluxo principal
# ---------------------------
def integrar_painel_final():
    session_folder = Path(st.session_state.get("session_folder", "")) if "session_folder" in st.session_state else None
    if not session_folder or not session_folder.exists():
        return

    painel_visualizacao_rapida(session_folder)
    painel_agrupar_e_baixar(session_folder)
    painel_gerenciamento_extra()

# Execute integração se já houver resultados
if "resultados" in st.session_state and st.session_state.get("resultados"):
    integrar_painel_final()

# =========================
# SUGESTÕES, NOTAS E REQUIREMENTS
# =========================

st.markdown("---")
st.markdown("## 💡 Sugestões e próximas melhorias (rápidas)")
st.markdown("""
- Extrair e guardar o CNPJ/CPF do emitente para agrupamentos mais robustos.
- Validar duplicatas (mesmo número + CNPJ) e oferecer _merge_ inteligente.
- Adicionar OCR fallback local (Tesseract) para PDFs que não contenham texto.
- Workflow assíncrono com filas (Redis + Celery/RQ) se processar muitos arquivos.
- Autenticação e armazenamento em S3 (ou GCS) para produção.
- Testes unitários e CI (GitHub Actions).
""")

st.markdown("## 📦 requirements.txt sugerido")
st.code("""
streamlit
PyPDF2
google-generative-ai
python-dotenv
pdf2image
pillow
pymupdf
pandas
""", language="text")

st.markdown("**Observação:** `pdf2image` precisa do `poppler` instalado no sistema (apt/brew). `pymupdf` (fitz) é uma alternativa que não precisa do poppler. Instale só um dos dois para thumbnails.")
