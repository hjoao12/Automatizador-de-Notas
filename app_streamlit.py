# turbo_v7_ultimate.py
# Versão: Turbo v7 — Ultimate (Multi-Key + Global Page Queue)
# Inovação: Processa páginas de todos os arquivos simultaneamente e rotaciona chaves de API.

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
import requests
import itertools
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Biblioteca moderna de PDF
from pypdf import PdfReader, PdfWriter

import streamlit as st
from dotenv import load_dotenv

# =====================================================================
# CONFIGURAÇÃO INICIAL E VISUAL
# =====================================================================
load_dotenv()
st.set_page_config(
    page_title="Automatizador v7 (Multi-Key)",
    page_icon="🚀",
    layout="wide"
)

st.markdown(
    """
<style>
body { background-color: #f8f9fa; color: #212529; font-family: 'Segoe UI', Roboto, Arial, sans-serif; }
[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e9ecef; }
div.stButton > button { background-color: #0f4c81; color: white; border-radius: 8px; border: none; font-weight: 500; }
div.stButton > button:hover { background-color: #0b3a5a; }
.card { background: #fff; padding: 15px; border-radius:8px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom:15px; }
.success-log { color: #155724; background-color: #d4edda; padding: 4px 8px; border-radius: 4px; font-size: 0.9em; }
.warning-log { color: #856404; background-color: #fff3cd; padding: 4px 8px; border-radius: 4px; font-size: 0.9em; }
.api-badge { background-color: #e2e8f0; color: #475569; padding: 2px 6px; border-radius: 4px; font-size: 0.8em; font-family: monospace; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Automatizador de Notas — Turbo v7 (Multi-Key 🚀)")

# =====================================================================
# ESTRUTURAS
# =====================================================================
TEMP_FOLDER = Path("./temp")
TEMP_FOLDER.mkdir(exist_ok=True)
CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)
CONFIG_DIR = Path("./config")
CONFIG_DIR.mkdir(exist_ok=True)
PATTERNS_FILE = CONFIG_DIR / "patterns.json"

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))

# =====================================================================
# GERENCIAMENTO DE CHAVES (MULTI-KEY)
# =====================================================================
def load_api_keys():
    """Carrega chaves do secrets ou env e prepara o ciclo."""
    # Tenta pegar do secrets ou do env
    keys_str = ""
    if hasattr(st, "secrets") and st.secrets.get("GOOGLE_API_KEY"):
        keys_str = st.secrets["GOOGLE_API_KEY"]
    else:
        keys_str = os.getenv("GOOGLE_API_KEY", "")
    
    if not keys_str:
        return []
    
    # Divide por vírgula e limpa espaços
    lista_keys = [k.strip() for k in keys_str.split(",") if k.strip()]
    return lista_keys

API_KEYS = load_api_keys()
if not API_KEYS:
    st.error("❌ Nenhuma API Key encontrada. Configure GOOGLE_API_KEY no .env (pode usar várias separadas por vírgula).")
    st.stop()

# Criar um ciclo infinito de chaves (Round Robin)
KEY_CYCLE = itertools.cycle(API_KEYS)

def get_next_key():
    """Pega a próxima chave da lista."""
    return next(KEY_CYCLE)

# =====================================================================
# CLIENTE REST API (Thread-Safe)
# =====================================================================
def call_gemini_rest(prompt: str, pdf_bytes: bytes, api_key: str):
    """
    Chama a API REST diretamente para evitar conflito de estado global da lib oficial
    ao usar múltiplas chaves em threads.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
    # Codificar PDF para Base64 é feito automaticamente pelo json dump se preparar certo,
    # mas aqui vamos usar a estrutura inline data do protocolo.
    import base64
    b64_data = base64.b64encode(pdf_bytes).decode("utf-8")
    
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {
                    "mime_type": "application/pdf",
                    "data": b64_data
                }}
            ]
        }],
        "generationConfig": {
            "response_mime_type": "application/json"
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if response.status_code == 429:
            raise ResourceExhausted("Quota excedida REST")
        raise e

# =====================================================================
# CACHE & PADRÕES
# =====================================================================
class DocumentCache:
    def __init__(self):
        self.path = CACHE_DIR
    def get_key(self, b, p):
        return f"{hashlib.md5(b).hexdigest()}_{hashlib.md5(p.encode()).hexdigest()}"
    def get(self, k):
        f = self.path / f"{k}.pkl"
        if f.exists():
            try:
                with open(f, "rb") as h: return pickle.load(h)
            except: pass
        return None
    def set(self, k, v):
        try:
            with open(self.path / f"{k}.pkl", "wb") as h: pickle.dump(v, h)
        except: pass
    def clear(self):
        for f in self.path.glob("*.pkl"): f.unlink(missing_ok=True)

doc_cache = DocumentCache()

def load_patterns():
    if not PATTERNS_FILE.exists():
        p = {"COMPANHIA DE AGUA": "CAGEPA", "PETROLEO BRASILEIRO": "PETROBRAS"}
        with open(PATTERNS_FILE, "w") as f: json.dump(p, f)
        return p
    try:
        with open(PATTERNS_FILE, "r") as f: return json.load(f)
    except: return {}

def save_patterns(p):
    with open(PATTERNS_FILE, "w") as f: json.dump(p, f)

PATTERNS = load_patterns()

# =====================================================================
# FUNÇÕES DE TEXTO
# =====================================================================
def normalize(s):
    if not s: return ""
    s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII")
    return re.sub(r"[^A-Z0-9 ]+", " ", s.upper()).strip()

def clean_emitente(raw, city=None):
    norm = normalize(raw)
    c_norm = normalize(city) if city else None
    if "SABARA" in norm: return f"SB_{c_norm.split()[0]}" if c_norm else "SB"
    for k in sorted(PATTERNS.keys(), key=len, reverse=True):
        if normalize(k) in norm: return PATTERNS[k]
    return re.sub(r"\s+", "_", norm)

def clean_num(n): return re.sub(r"[^\d]", "", str(n or "0")).lstrip("0") or "0"

def extract_json(txt):
    if not txt: return []
    try:
        # Tenta pegar direto
        return json.loads(txt)
    except:
        # Tenta regex array
        m = re.search(r'\[.*\]', txt, re.DOTALL)
        if m: 
            try: return json.loads(m.group())
            except: pass
        # Tenta regex objeto
        m = re.search(r'\{.*\}', txt, re.DOTALL)
        if m:
            try: return [json.loads(m.group())]
            except: pass
        return []

# =====================================================================
# WORKER DE PÁGINA ÚNICA (GRANULARIDADE FINA)
# =====================================================================
def processar_pagina_atomica(task):
    """
    Processa UMA página isolada.
    task: { 'bytes': b, 'prompt': s, 'origem': s, 'page_idx': i, 'use_cache': b }
    """
    pdf_bytes = task["bytes"]
    prompt = task["prompt"]
    
    # Cache check
    ckey = doc_cache.get_key(pdf_bytes, prompt)
    if task["use_cache"]:
        cached = doc_cache.get(ckey)
        if cached:
            return {**task, "status": "CACHE", "result": cached, "key_used": "cache"}

    # API Call (Round Robin de chaves)
    result_data = None
    used_key_preview = ""
    
    for attempt in range(MAX_RETRIES + 1):
        current_key = get_next_key()
        used_key_preview = current_key[:5] + "..." # Para log
        
        try:
            # Usa REST API para thread-safety total com múltiplas chaves
            resp_json = call_gemini_rest(prompt, pdf_bytes, current_key)
            
            # Parse da resposta do Gemini REST
            try:
                text_content = resp_json["candidates"][0]["content"]["parts"][0]["text"]
                result_data = extract_json(text_content)
                break # Sucesso
            except Exception:
                # Resposta veio mas formato inesperado
                result_data = []
                break

        except Exception as e:
            # Backoff
            time.sleep(2 * (attempt + 1))
    
    if result_data is not None:
        doc_cache.set(ckey, result_data)
        return {**task, "status": "OK", "result": result_data, "key_used": used_key_preview}
    else:
        return {**task, "status": "ERRO", "error": "Falha API", "key_used": used_key_preview}

# =====================================================================
# INTERFACE E FLUXO
# =====================================================================
with st.sidebar:
    st.header("🚀 Configuração Turbo")
    st.info(f"🔑 Chaves de API carregadas: **{len(API_KEYS)}**")
    workers = st.slider("Processos Simultâneos (Threads)", 2, 16, 8)
    use_cache = st.checkbox("Usar Cache", True)
    
    st.markdown("---")
    st.subheader("Padrões")
    k = st.text_input("De (Original)")
    v = st.text_input("Para (Novo)")
    if st.button("➕ Adicionar") and k and v:
        PATTERNS[k] = v
        save_patterns(PATTERNS)
        st.rerun()
    
    with st.expander("Ver Lista"):
        for pk, pv in PATTERNS.items():
            if st.button(f"🗑️ {pk}", key=pk):
                del PATTERNS[pk]
                save_patterns(PATTERNS)
                st.rerun()

st.markdown('<div class="warning-box">⚠️ Ambiente Nuvem: Exporte seus padrões se precisar salvá-los.</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader("Selecione PDFs", type=["pdf"], accept_multiple_files=True)

if st.button("⚡ INICIAR TURBO V7", type="primary") and uploaded_files:
    session_path = TEMP_FOLDER / str(uuid.uuid4())
    session_path.mkdir(exist_ok=True)
    st.session_state["session_path"] = str(session_path)
    
    # 1. DESMONTAR TODOS OS ARQUIVOS EM PÁGINAS (GLOBAL QUEUE)
    st.info("🔨 Desmontando arquivos para processamento paralelo...")
    all_tasks = []
    
    prompt = (
        "Extraia: emitente, numero_nota, cidade. "
        "Retorne JSON ARRAY: [{\"emitente\": \"...\", \"numero_nota\": \"...\", \"cidade\": \"...\"}]"
    )
    
    for f in uploaded_files:
        try:
            f_bytes = f.read()
            reader = PdfReader(io.BytesIO(f_bytes))
            for i, page in enumerate(reader.pages):
                # Extrair página individual
                buf = io.BytesIO()
                w = PdfWriter()
                w.add_page(page)
                w.write(buf)
                
                all_tasks.append({
                    "bytes": buf.getvalue(),
                    "prompt": prompt,
                    "origem": f.name,
                    "page_idx": i,
                    "use_cache": use_cache
                })
        except Exception as e:
            st.error(f"Erro ao ler {f.name}: {e}")

    total_pages = len(all_tasks)
    st.write(f"🔥 Total de páginas na fila: **{total_pages}**")
    
    # 2. PROCESSAMENTO PARALELO MASSIVO
    prog_bar = st.progress(0)
    status_text = st.empty()
    logs = []
    results_map = {} # Map para reagrupamento
    
    done_count = 0
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(processar_pagina_atomica, t) for t in all_tasks]
        
        for future in as_completed(futures):
            done_count += 1
            res = future.result()
            
            # Log visual
            if res["status"] == "OK":
                k_label = f"Key: {res['key_used']}"
                status_text.markdown(f"<span class='success-log'>✅ {res['origem']} p.{res['page_idx']+1} ({k_label})</span>", unsafe_allow_html=True)
            elif res["status"] == "CACHE":
                status_text.markdown(f"<span class='api-badge'>💾 Cache Hit: {res['origem']} p.{res['page_idx']+1}</span>", unsafe_allow_html=True)
            else:
                status_text.markdown(f"<span class='warning-log'>❌ Erro: {res['origem']} p.{res['page_idx']+1}</span>", unsafe_allow_html=True)
                logs.append(f"Erro em {res['origem']}: {res.get('error')}")
            
            prog_bar.progress(done_count / total_pages)
            
            # Processar dados para reagrupamento
            data_list = res.get("result", [])
            # Normalizar retorno (pode vir lista ou dict)
            if isinstance(data_list, dict): data_list = [data_list]
            if not isinstance(data_list, list): data_list = []
            
            # Pega o primeiro resultado válido da página
            item = data_list[0] if data_list else {}
            
            emit = clean_emitente(item.get("emitente"), item.get("cidade"))
            num = clean_num(item.get("numero_nota"))
            
            if num != "0":
                key = (num, emit)
                if key not in results_map: results_map[key] = []
                results_map[key].append({
                    "bytes": res["bytes"],
                    "origem": res["origem"],
                    "pg_idx": res["page_idx"]
                })

    # 3. MONTAGEM DOS ARQUIVOS FINAIS
    status_text.text("🏗️ Montando PDFs finais...")
    final_results = []
    
    for (num, emit), pages_data in results_map.items():
        # Ordenar páginas para manter coerência (Origem A p1 vem antes de Origem A p2)
        pages_data.sort(key=lambda x: (x["origem"], x["pg_idx"]))
        
        writer = PdfWriter()
        origens_set = set()
        
        for p in pages_data:
            r_temp = PdfReader(io.BytesIO(p["bytes"]))
            writer.add_page(r_temp.pages[0])
            origens_set.add(p["origem"])
            
        fname = f"DOC {num}_{emit}.pdf"
        fpath = session_path / fname
        
        # Evitar sobrescrita
        c = 2
        stem = fpath.stem
        while fpath.exists():
            fpath = session_path / f"{stem}_{c}.pdf"
            c += 1
            
        with open(fpath, "wb") as f:
            writer.write(f)
            
        final_results.append({
            "file_name": fpath.name,
            "path": str(fpath),
            "pages": len(pages_data),
            "origem": ", ".join(origens_set)
        })
        
    st.session_state["final_results"] = final_results
    st.success(f"🏁 Pronto! {len(final_results)} documentos gerados.")
    st.rerun()

# =====================================================================
# ÁREA DE RESULTADOS E DOWNLOAD
# =====================================================================
if "final_results" in st.session_state:
    st.markdown("---")
    st.subheader("📂 Arquivos Prontos")
    
    results = st.session_state["final_results"]
    
    # Botão de Download em Massa (Topo)
    if st.button("📦 Baixar ZIP de Todos"):
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, "w") as zf:
            for r in results:
                p = Path(r["path"])
                if p.exists(): zf.write(p, arcname=r["file_name"])
        mem.seek(0)
        st.download_button("⬇️ Download ZIP", mem, "arquivos_processados.zip", "application/zip")

    # Lista Individual
    for i, res in enumerate(results):
        p = Path(res["path"])
        if not p.exists(): continue
        
        with st.container():
            c1, c2, c3 = st.columns([3, 2, 1])
            c1.markdown(f"**📄 {res['file_name']}** <span style='color:gray;font-size:0.8em'>({res['pages']} pág)</span>", unsafe_allow_html=True)
            
            new_name = c2.text_input("Renomear", res["file_name"], key=f"rn_{i}", label_visibility="collapsed")
            if new_name != res["file_name"]:
                if c2.button("Salvar", key=f"save_{i}"):
                    new_p = p.parent / new_name
                    if not new_name.endswith(".pdf"): new_p = p.parent / f"{new_name}.pdf"
                    p.rename(new_p)
                    res["path"] = str(new_p)
                    res["file_name"] = new_p.name
                    st.rerun()
            
            with open(p, "rb") as f:
                c3.download_button("⬇️", f, file_name=res["file_name"], key=f"dl_{i}")

    # Botão Exportar Padrões
    st.markdown("---")
    st.download_button("Exportar Padrões (.json)", json.dumps(PATTERNS), "padroes.json")