# turbo_v6_workflow.py
# Versão: Turbo v6 — Workflow Edition
# Melhoria: Split e Merge agora usam os arquivos JÁ processados (sem re-upload).

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

# Biblioteca moderna
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
    page_title="Automatizador de Notas v6",
    page_icon="⚡",
    layout="wide"
)

st.markdown(
    """
<style>
body { background-color: #f8f9fa; color: #212529; font-family: 'Segoe UI', Roboto, Arial, sans-serif; }
[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e9ecef; }
h1, h2, h3, h4 { color: #0f4c81; }
div.stButton > button { background-color: #0f4c81; color: white; border-radius: 8px; border: none; font-weight: 500; }
div.stButton > button:hover { background-color: #0b3a5a; }
.card { background: #fff; padding: 12px; border-radius:8px; box-shadow: 0 6px 18px rgba(15,76,129,0.04); margin-bottom:12px; }
.small-note { font-size:13px; color:#6b7280; }
.warning-box { background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 10px; border-radius: 4px; color: #856404; font-size: 0.9em; margin-bottom: 15px; }
.tool-box { background-color: #e9ecef; padding: 15px; border-radius: 8px; margin-top: 20px; border: 1px solid #dee2e6; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Automatizador de Notas Fiscais — Turbo v6 (Workflow)")

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

MAX_WORKERS_DEFAULT = 4
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
BATCH_SIZE_DEFAULT = int(os.getenv("BATCH_SIZE", "8"))

# =====================================================================
# SISTEMA DE CACHE
# =====================================================================
class DocumentCache:
    def __init__(self, cache_dir=CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def get_cache_key_file(self, file_bytes: bytes, prompt: str):
        content_hash = hashlib.md5(file_bytes).hexdigest()
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        return f"{content_hash}_{prompt_hash}"

    def get(self, key):
        f = self.cache_dir / f"{key}.pkl"
        if f.exists():
            try:
                with open(f, "rb") as h: return pickle.load(h)
            except: return None
        return None

    def set(self, key, data):
        try:
            with open(self.cache_dir / f"{key}.pkl", "wb") as h: pickle.dump(data, h)
        except: pass

    def clear(self):
        for f in self.cache_dir.glob("*.pkl"):
            try: f.unlink()
            except: pass

document_cache = DocumentCache()

# =====================================================================
# GERENCIAMENTO DE PADRÕES
# =====================================================================
def load_patterns():
    if not PATTERNS_FILE.exists():
        default = {"COMPANHIA DE AGUA": "CAGEPA", "PETROLEO BRASILEIRO": "PETROBRAS"}
        save_patterns(default)
        return default
    try:
        with open(PATTERNS_FILE, "r", encoding="utf-8") as f: return json.load(f)
    except: return {}

def save_patterns(pats: dict):
    try:
        with open(PATTERNS_FILE, "w", encoding="utf-8") as f: json.dump(pats, f, indent=2, ensure_ascii=False)
    except: pass

PATTERNS = load_patterns()

def add_pattern(raw, sub):
    raw, sub = raw.strip(), sub.strip()
    if not raw or not sub: return False
    PATTERNS[raw] = sub
    save_patterns(PATTERNS)
    return True

def remove_pattern(raw):
    if raw in PATTERNS:
        PATTERNS.pop(raw)
        save_patterns(PATTERNS)
        return True
    return False

# =====================================================================
# FUNÇÕES AUXILIARES
# =====================================================================
def _normalizar_texto(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII")
    s = re.sub(r"[^A-Z0-9 ]+", " ", s.upper())
    return re.sub(r"\s+", " ", s).strip()

def substituir_nome_emitente(nome_raw: str, cidade_raw: str = None) -> str:
    nome_norm = _normalizar_texto(nome_raw)
    cidade_norm = _normalizar_texto(cidade_raw) if cidade_raw else None
    if "SABARA" in nome_norm: return f"SB_{cidade_norm.split()[0]}" if cidade_norm else "SB"
    for padrao in sorted(PATTERNS.keys(), key=len, reverse=True):
        if _normalizar_texto(padrao) in nome_norm: return PATTERNS[padrao]
    return re.sub(r"\s+", "_", nome_norm)

def limpar_emitente(nome: str) -> str:
    if not nome: return "SEM_NOME"
    nome = re.sub(r"[^A-Z0-9_]+", "_", nome.upper())
    return re.sub(r"_+", "_", nome).strip("_")

def limpar_numero(numero: str) -> str:
    if not numero: return "0"
    n = re.sub(r"[^\d]", "", str(numero))
    return n.lstrip("0") or "0"

def extrair_json_seguro(texto: str):
    if not texto: return []
    texto = texto.strip()
    # 1. Array
    m_arr = re.search(r'\[.*\]', texto, re.DOTALL)
    if m_arr:
        try: return json.loads(m_arr.group())
        except: pass
    # 2. Object
    m_obj = re.search(r'\{.*\}', texto, re.DOTALL)
    if m_obj:
        try: return [json.loads(m_obj.group())]
        except: pass
    # 3. Fallback
    clean = texto.replace("```json", "").replace("```", "").strip()
    try:
        res = json.loads(clean)
        return [res] if isinstance(res, dict) else res
    except: return []

# =====================================================================
# CONFIG GEMINI
# =====================================================================
if hasattr(st, "secrets") and st.secrets.get("GOOGLE_API_KEY"):
    GEMINI_API_KEY = st.secrets["GOOGLE_API_KEY"]
else:
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    st.error("❌ Configure a GOOGLE_API_KEY.")
    st.stop()

try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("models/gemini-1.5-flash")
except Exception as e:
    st.error(f"Erro API: {e}")
    st.stop()

# =====================================================================
# WORKER (PROCESSAMENTO)
# =====================================================================
def processar_arquivo_worker(arquivo_dados, use_cache, batch_size=BATCH_SIZE_DEFAULT):
    fname = arquivo_dados["name"]
    file_bytes = arquivo_dados["bytes"]
    logs = []
    
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        num_pages = len(reader.pages)
    except Exception as e:
        return {"error": f"PDF Corrompido: {e}", "name": fname, "results": [], "bytes_originais": file_bytes, "logs": []}

    prompt = (
        "Extraia para cada página: pagina (numero), emitente, numero_nota, cidade. "
        "Retorne JSON ARRAY: [{\"pagina\": 1, \"emitente\": \"...\", \"numero_nota\": \"...\", \"cidade\": \"...\"}, ...]"
    )

    cache_key = document_cache.get_cache_key_file(file_bytes, prompt)
    if use_cache:
        cached = document_cache.get(cache_key)
        if cached: return {"name": fname, "results": cached, "bytes_originais": file_bytes, "cached": True, "logs": ["Cache Hit"]}

    page_results = [None] * num_pages
    curr = 0
    
    while curr < num_pages:
        end = min(curr + batch_size, num_pages)
        batch_imgs = []
        idxs = []
        for i in range(curr, end):
            buf = io.BytesIO()
            PdfWriter().add_page(reader.pages[i]).write(buf) if len(reader.pages) > i else None
            batch_imgs.append({"mime_type": "application/pdf", "data": buf.getvalue()})
            idxs.append(i)

        # Call API
        extracted = []
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = model.generate_content([prompt] + batch_imgs, generation_config={"response_mime_type": "application/json"})
                extracted = extrair_json_seguro(resp.text)
                break
            except Exception as e:
                time.sleep(2 * (attempt + 1))
        
        # Map results
        if extracted:
            # Tenta mapear por numero de pagina
            for item in extracted:
                if isinstance(item, dict) and "pagina" in item:
                    try:
                        pg_real = int(item["pagina"]) - 1
                        if pg_real in idxs: page_results[pg_real] = item
                    except: pass
            
            # Fallback: mapear por ordem se falhou
            for i, val in enumerate(extracted):
                if i < len(idxs) and page_results[idxs[i]] is None:
                    page_results[idxs[i]] = val
        
        curr += batch_size

    if use_cache: document_cache.set(cache_key, page_results)
    return {"name": fname, "results": page_results, "bytes_originais": file_bytes, "cached": False, "logs": logs}

# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.header("⚙️ Configuração")
    use_cache = st.checkbox("Usar Cache", value=True)
    workers = st.slider("Workers", 1, 8, MAX_WORKERS_DEFAULT)
    
    st.markdown("---")
    st.subheader("📝 Padrões")
    new_k = st.text_input("Texto Original")
    new_v = st.text_input("Novo Nome")
    if st.button("➕ Add Padrão"):
        add_pattern(new_k, new_v)
        st.rerun()
    
    with st.expander("Ver Padrões"):
        for k,v in PATTERNS.items():
            if st.button(f"🗑️ {k} -> {v}", key=k):
                remove_pattern(k)
                st.rerun()
    
    if st.button("Limpar Cache"):
        document_cache.clear()

# =====================================================================
# APP PRINCIPAL
# =====================================================================
st.markdown('<div class="warning-box">⚠️ Modo Nuvem: Exporte seus padrões antes de sair.</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader("Selecione PDFs para processar", type=["pdf"], accept_multiple_files=True)

col1, col2 = st.columns([1, 4])
if col1.button("🚀 Processar", type="primary") and uploaded_files:
    session_path = TEMP_FOLDER / str(uuid.uuid4())
    session_path.mkdir(exist_ok=True)
    st.session_state["session_path"] = str(session_path)
    
    files_data = [{"name": f.name, "bytes": f.read()} for f in uploaded_files]
    prog = st.progress(0)
    status = st.empty()
    
    all_results = {}
    logs = []
    
    with ThreadPoolExecutor(max_workers=workers) as exc:
        futures = {exc.submit(processar_arquivo_worker, f, use_cache): f["name"] for f in files_data}
        done = 0
        for fut in as_completed(futures):
            res = fut.result()
            done += 1
            prog.progress(done / len(files_data))
            
            if "error" in res:
                logs.append(f"❌ {res['name']}: {res['error']}")
                continue
                
            # Agrupamento
            reader = PdfReader(io.BytesIO(res["bytes_originais"]))
            for i, page_data in enumerate(res["results"]):
                if not page_data: continue
                
                emit = limpar_emitente(substituir_nome_emitente(page_data.get("emitente", ""), page_data.get("cidade", "")))
                num = limpar_numero(page_data.get("numero_nota", ""))
                
                if num == "0": continue
                
                key = (num, emit)
                if key not in all_results: all_results[key] = []
                
                buf = io.BytesIO()
                w = PdfWriter()
                w.add_page(reader.pages[i])
                w.write(buf)
                all_results[key].append({"data": buf.getvalue(), "origem": res["name"]})

    # Gerar Finais
    final_meta = []
    for (num, emit), pages in all_results.items():
        w_final = PdfWriter()
        for p in pages:
            w_final.add_page(PdfReader(io.BytesIO(p["data"])).pages[0])
            
        fname = f"DOC {num}_{emit}.pdf"
        path = session_path / fname
        c = 2
        while path.exists():
            path = session_path / f"DOC {num}_{emit}_{c}.pdf"
            c += 1
            
        with open(path, "wb") as f: w_final.write(f)
        
        final_meta.append({
            "file_name": path.name,
            "path": str(path),
            "pages": len(pages),
            "origem": list(set(p["origem"] for p in pages))
        })
        
    st.session_state["final_results"] = final_meta
    st.session_state["logs"] = logs
    st.success("Processamento concluído!")
    st.rerun()

if col2.button("♻️ Limpar Tudo"):
    st.session_state.clear()
    st.rerun()

# =====================================================================
# ÁREA DE RESULTADOS & EDIÇÃO
# =====================================================================
if "final_results" in st.session_state:
    st.markdown("---")
    st.header("📂 Arquivos Gerados")
    
    results = st.session_state["final_results"]
    
    # --- LISTA PRINCIPAL ---
    for i, item in enumerate(results):
        p = Path(item["path"])
        if not p.exists(): continue
        
        with st.container():
            col_a, col_b, col_c = st.columns([3, 2, 1])
            with col_a:
                st.markdown(f"**📄 {item['file_name']}** <span style='color:gray;font-size:0.8em'>({item['pages']} pág)</span>", unsafe_allow_html=True)
            with col_b:
                new_n = st.text_input("Renomear", item["file_name"], key=f"rn_{i}", label_visibility="collapsed")
                if new_n != item["file_name"]:
                    if st.button("Salvar Nome", key=f"btn_rn_{i}"):
                        new_p = p.parent / new_n if new_n.endswith(".pdf") else p.parent / f"{new_n}.pdf"
                        p.rename(new_p)
                        item["path"] = str(new_p)
                        item["file_name"] = new_p.name
                        st.rerun()
            with col_c:
                with open(p, "rb") as f:
                    st.download_button("⬇️", f, file_name=item["file_name"], key=f"dl_{i}")

    # =================================================================
    # 🛠️ FERRAMENTAS DE EDIÇÃO (PÓS-PROCESSAMENTO)
    # =================================================================
    st.markdown("---")
    st.markdown("### 🛠️ Ferramentas de Edição (Nos arquivos acima)")
    
    # Opções disponíveis (apenas nomes atuais)
    opcoes_arquivos = {res["file_name"]: res for res in results if Path(res["path"]).exists()}
    lista_nomes = list(opcoes_arquivos.keys())

    c_split, c_merge = st.columns(2)

    # --- FERRAMENTA 1: SPLIT (SEPARAR) ---
    with c_split:
        st.markdown('<div class="tool-box"><h5>✂️ Separar Páginas</h5>', unsafe_allow_html=True)
        
        sel_split = st.selectbox("Escolha o arquivo:", [""] + lista_nomes, key="sel_split")
        paginas_str = st.text_input("Páginas (ex: 1, 3-5):", key="pg_split")
        
        if st.button("Separar PDF", key="btn_split") and sel_split and paginas_str:
            try:
                # Pega o caminho real do arquivo selecionado
                caminho_origem = opcoes_arquivos[sel_split]["path"]
                reader = PdfReader(caminho_origem)
                total_pg = len(reader.pages)
                
                writer = PdfWriter()
                added = 0
                
                # Parse das páginas
                for parte in paginas_str.split(","):
                    parte = parte.strip()
                    if "-" in parte:
                        inicio, fim = map(int, parte.split("-"))
                        for p in range(inicio, fim + 1):
                            if 1 <= p <= total_pg:
                                writer.add_page(reader.pages[p-1])
                                added += 1
                    else:
                        p = int(parte)
                        if 1 <= p <= total_pg:
                            writer.add_page(reader.pages[p-1])
                            added += 1
                
                if added > 0:
                    out = io.BytesIO()
                    writer.write(out)
                    out.seek(0)
                    st.download_button(f"⬇️ Baixar Separado ({added} pág)", out, f"split_{sel_split}", "application/pdf")
                    st.success("Pronto para baixar!")
                else:
                    st.warning("Nenhuma página válida selecionada.")
            except Exception as e:
                st.error(f"Erro: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # --- FERRAMENTA 2: MERGE (JUNTAR) ---
    with c_merge:
        st.markdown('<div class="tool-box"><h5>🔗 Juntar Arquivos (Merge)</h5>', unsafe_allow_html=True)
        
        sel_merge = st.multiselect("Selecione os arquivos (na ordem):", lista_nomes, key="sel_merge")
        nome_merge = st.text_input("Nome do novo arquivo:", "agrupado.pdf", key="name_merge")
        
        if st.button("Juntar PDFs", key="btn_merge") and sel_merge:
            try:
                writer = PdfWriter()
                count_files = 0
                
                for nome in sel_merge:
                    caminho = opcoes_arquivos[nome]["path"]
                    r_temp = PdfReader(caminho)
                    for p in r_temp.pages:
                        writer.add_page(p)
                    count_files += 1
                
                out = io.BytesIO()
                writer.write(out)
                out.seek(0)
                
                safe_name = nome_merge if nome_merge.endswith(".pdf") else f"{nome_merge}.pdf"
                st.download_button(f"⬇️ Baixar ({count_files} arquivos unidos)", out, safe_name, "application/pdf")
                st.success("Unido com sucesso!")
                
            except Exception as e:
                st.error(f"Erro: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

    # =================================================================
    # DOWNLOAD EM MASSA
    # =================================================================
    st.markdown("---")
    if st.button("📦 Baixar ZIP Completo"):
        mem_zip = io.BytesIO()
        with zipfile.ZipFile(mem_zip, "w") as zf:
            for item in results:
                p = Path(item["path"])
                if p.exists(): zf.write(p, arcname=p.name)
        mem_zip.seek(0)
        st.download_button("⬇️ Download ZIP", mem_zip, "processados.zip", "application/zip")
    
    # Backup Padrões
    st.markdown("---")
    st.download_button("Exportar Padrões", json.dumps(PATTERNS), "padroes.json")
