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
import base64
import gc
import pandas as pd
from supabase import create_client, Client
from streamlit_pdf_viewer import pdf_viewer
from pathlib import Path
from pypdf import PdfReader, PdfWriter
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image
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

# ======= CSS Corporativo Claro =======
st.markdown("""
<style>
body { background-color: #f8f9fa; color: #212529; font-family: 'Segoe UI', Roboto, Arial, sans-serif; }
[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e9ecef; }
h1, h2, h3, h4 { color: #0f4c81; }
div.stButton > button { background-color: #0f4c81; color: white; border-radius: 8px; border: none; font-weight: 500; }
div.stButton > button:hover { background-color: #0b3a5a; }
.stProgress > div > div > div > div { background-color: #28a745 !important; }
.success-log { color: #155724; background-color: #d4edda; padding: 6px 10px; border-radius: 6px; font-size: 0.9rem; }
.warning-log { color: #856404; background-color: #fff3cd; padding: 6px 10px; border-radius: 6px; font-size: 0.9rem; }
.error-log { color: #721c24; background-color: #f8d7da; padding: 6px 10px; border-radius: 6px; font-size: 0.9rem; }
.card { background: #fff; padding: 15px; border-radius:8px; box-shadow: 0 4px 12px rgba(15,76,129,0.08); margin-bottom:15px; }
.manage-panel { background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #0f4c81; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

st.title("Automatizador de Notas Fiscais PDF")

# =====================================================================
# SISTEMA DE CACHE INTELIGENTE
# =====================================================================
class DocumentCache:
    def __init__(self, cache_dir="./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_key(self, pdf_bytes, prompt):
        content_hash = hashlib.md5(pdf_bytes).hexdigest()
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        return f"{content_hash}_{prompt_hash}"
    
    def get(self, key):
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except: return None
        return None
    
    def set(self, key, data):
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except: pass
    
    def clear(self):
        for cache_file in self.cache_dir.glob("*.pkl"):
            try: cache_file.unlink()
            except: pass

document_cache = DocumentCache()

# =====================================================================
# CONFIGURAÇÃO GEMINI
# =====================================================================
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    try: GEMINI_API_KEY = st.secrets["GOOGLE_API_KEY"]
    except: pass

if not GEMINI_API_KEY:
    st.error("❌ Chave GOOGLE_API_KEY não encontrada (.env ou secrets).")
    st.stop()

try:
    genai.configure(api_key=GEMINI_API_KEY)
    # Dica: gemini-1.5-flash costuma ser mais estável para OCR massivo que o 2.0-flash (preview)
    # Se der erro, tente trocar "models/gemini-2.0-flash" por "gemini-1.5-flash"
    model = genai.GenerativeModel(os.getenv("MODEL_NAME", "models/gemini-2.5-flash")) 
except Exception as e:
    st.error(f"❌ Erro ao configurar Gemini: {str(e)}")
    st.stop()

# =====================================================================
# CONFIGURAÇÕES GERAIS
# =====================================================================
TEMP_FOLDER = Path("./temp")
os.makedirs(TEMP_FOLDER, exist_ok=True)

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
MIN_RETRY_DELAY = int(os.getenv("MIN_RETRY_DELAY", "2"))
MAX_RETRY_DELAY = int(os.getenv("MAX_RETRY_DELAY", "15"))

# =====================================================================
# GESTÃO DE PADRÕES (VIA SUPABASE)
# =====================================================================
@st.cache_resource
def init_supabase():
    url = None
    key = None
    
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
    except: pass

    if not url:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")

    if url and key:
        return create_client(url, key)
    return None

supabase = init_supabase()

def get_patterns_db():
    if not supabase: return {}
    try:
        response = supabase.table("invoice_patterns").select("*").execute()
        return {item["origin"]: item["target"] for item in response.data}
    except Exception as e:
        print(f"Erro ao ler banco: {e}")
        return {}

def sync_patterns_db(new_dict):
    if not supabase: return False
    try:
        current_data = supabase.table("invoice_patterns").select("origin").execute()
        db_keys = {row['origin'] for row in current_data.data}
        new_keys = set(new_dict.keys())

        to_delete = list(db_keys - new_keys)
        if to_delete:
            supabase.table("invoice_patterns").delete().in_("origin", to_delete).execute()

        upsert_data = [{"origin": k, "target": v} for k, v in new_dict.items()]
        if upsert_data:
            supabase.table("invoice_patterns").upsert(upsert_data, on_conflict="origin").execute()
        return True
    except Exception as e:
        st.error(f"Erro ao salvar: {e}")
        return False

if "db_patterns" not in st.session_state:
    st.session_state["db_patterns"] = get_patterns_db()

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
        
    patterns = st.session_state.get("db_patterns", {})
    for padrao, substituto in patterns.items():
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
    
def limpar_para_nome_arquivo(texto):
    if not texto: return "DESCONHECIDO"
    texto = re.sub(r'[\\/*?:"<>|]', "", texto)
    texto = unicodedata.normalize("NFKD", texto).encode("ASCII", "ignore").decode("ASCII")
    return texto.strip()[:60]

# --- LÓGICA HÍBRIDA: AI + REGEX NO TEXTO REAL ---
def validar_e_corrigir_dados(dados, texto_pdf_real=""):
    if not isinstance(dados, dict):
        if isinstance(dados, list) and len(dados) > 0 and isinstance(dados[0], dict):
            dados = dados[0]
        else:
            return {"emitente": "ERRO_FORMATO", "numero_nota": "000", "cidade": ""}
    
    dados_norm = {}
    for k, v in dados.items():
        k_lower = k.lower().strip()
        if any(x in k_lower for x in ["numero", "nota", "nfs", "danfe"]): key = "numero_nota"
        elif any(x in k_lower for x in ["emitente", "prestador", "social"]): key = "emitente"
        elif "cidade" in k_lower: key = "cidade"
        else: key = k
        dados_norm[key] = str(v) if v else ""
    dados = dados_norm

    raw_num = dados.get('numero_nota', '')
    numeros_limpos = re.sub(r'[^\d]', '', raw_num)

    # Se a IA falhou no número E temos texto real (não escaneado), tentamos regex
    if (not numeros_limpos or int(numeros_limpos) == 0) and texto_pdf_real:
        padroes_resgate = [
            r"N[°ºo]\s*([0-9\.]+)",
            r"NF[ \-]*e?\s*[:.]?\s*([0-9\.]+)",
            r"Número\s*[:.]?\s*([0-9\.]+)"
        ]
        for p in padroes_resgate:
            match = re.search(p, texto_pdf_real, re.IGNORECASE)
            if match:
                candidato = match.group(1).replace('.', '')
                if candidato.isdigit() and int(candidato) > 0:
                    dados['numero_nota'] = candidato
                    break
    
    final_num = re.sub(r'[^\d]', '', str(dados.get('numero_nota', '')))
    dados['numero_nota'] = final_num.lstrip('0') if final_num else "000"

    emitente = dados.get('emitente', '').strip()

    if not emitente:
        dados['emitente'] = "EMITENTE_DESCONHECIDO"
    else:
        dados['emitente'] = emitente


    if 'cidade' not in dados:
        dados['cidade'] = ""
    
    return dados

def extrair_cabecalho_pagina(pdf_bytes, page_idx, dpi=220):
    images = convert_from_bytes(
        pdf_bytes,
        dpi=dpi,
        first_page=page_idx + 1,
        last_page=page_idx + 1
    )

    img = images[0]
    w, h = img.size

    # Crop adaptativo (topo da DANFE)
    crop_ratio = 0.35 if h < 3500 else 0.40
    header = img.crop((0, 0, w, int(h * crop_ratio)))

    buf = io.BytesIO()
    header.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf


# =====================================================================
# PROCESSAMENTO GEMINI
# =====================================================================
def calcular_delay(tentativa, error_msg):
    if "retry in" in error_msg.lower():
        try:
            val = float(re.search(r"retry in (\d+\.?\d*)s", error_msg.lower()).group(1))
            return min(val + 1, MAX_RETRY_DELAY)
        except: pass
    return min(MIN_RETRY_DELAY * (tentativa + 1), MAX_RETRY_DELAY)


def processar_pagina_worker(job_data):
    """
    Processa uma única página de PDF usando OCR/AI (Gemini),
    com validação de existência de páginas e fallback seguro.
    """
    pdf_bytes = job_data["bytes"]
    prompt = job_data["prompt"]
    name = job_data["name"]
    page_idx = job_data["page_idx"]

    texto_pdf_real = ""

    try:
        # --- Validação do PDF ---
        reader = PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(reader.pages)
        if page_idx >= total_pages:
            return {
                "status": "ERRO",
                "dados": {"emitente": "", "numero_nota": "000", "cidade": ""},
                "tempo": 0,
                "provider": "",
                "name": name,
                "page_idx": page_idx,
                "error_msg": f"Página {page_idx+1} inexistente (total {total_pages})",
                "pdf_bytes": pdf_bytes,
                "texto_real": texto_pdf_real
            }

        # --- Extrair cabeçalho ---
        try:
            img_header = extrair_cabecalho_pagina(pdf_bytes, page_idx)
        except Exception as e:
            st.warning(f"⚠️ Erro ao extrair imagem do PDF {name}, página {page_idx+1}: {e}")
            # fallback: imagem em branco 1x1
            img_header = Image.new("RGB", (1, 1), (255, 255, 255))
            buf = io.BytesIO()
            img_header.save(buf, format="PNG")
            buf.seek(0)
            img_header = buf

        img_bytes = img_header.getvalue()

        # --- Cache ---
        cache_key = document_cache.get_cache_key(img_bytes, prompt)
        cached_result = document_cache.get(cache_key)
        if cached_result and job_data.get("use_cache", True):
            return {
                "status": "CACHE",
                "dados": cached_result['dados'],
                "tempo": cached_result['tempo'],
                "provider": cached_result['provider'],
                "name": name,
                "page_idx": page_idx,
                "pdf_bytes": pdf_bytes,
                "texto_real": texto_pdf_real
            }

        # --- Chamada ao Gemini ---
        # Nota: page_stream ou img_bytes pode ser usado dependendo da implementação
        try:
            dados, ok, tempo, provider = processar_pagina_gemini(prompt, img_bytes)
        except Exception as e:
            return {
                "status": "ERRO",
                "dados": {"emitente": "", "numero_nota": "000", "cidade": ""},
                "tempo": 0,
                "provider": "",
                "name": name,
                "page_idx": page_idx,
                "error_msg": f"Erro Gemini: {e}",
                "pdf_bytes": pdf_bytes,
                "texto_real": texto_pdf_real
            }

        # --- Armazenar no cache se sucesso ---
        if ok and "error" not in dados:
            document_cache.set(cache_key, {'dados': dados, 'tempo': tempo, 'provider': provider})
            return {
                "status": "OK",
                "dados": dados,
                "tempo": tempo,
                "provider": provider,
                "name": name,
                "page_idx": page_idx,
                "pdf_bytes": pdf_bytes,
                "texto_real": texto_pdf_real
            }
        else:
            return {
                "status": "ERRO",
                "dados": dados,
                "tempo": tempo,
                "provider": provider,
                "name": name,
                "page_idx": page_idx,
                "error_msg": dados.get("error", "Erro desconhecido"),
                "pdf_bytes": pdf_bytes,
                "texto_real": texto_pdf_real
            }

    except Exception as e_outer:
        # Proteção final: nunca quebra o loop
        return {
            "status": "ERRO",
            "dados": {"emitente": "", "numero_nota": "000", "cidade": ""},
            "tempo": 0,
            "provider": "",
            "name": name,
            "page_idx": page_idx,
            "error_msg": f"Erro crítico inesperado: {e_outer}",
            "pdf_bytes": pdf_bytes,
            "texto_real": texto_pdf_real
        }


# =====================================================================
# UI & MAIN FLOW
# =====================================================================
with st.sidebar:
    st.markdown("### ⚙️ Painel de Controle")
    if supabase:
        st.markdown("Status DB: <span style='color:green'><b>● Conectado</b></span>", unsafe_allow_html=True)
    else:
        st.markdown("Status DB: <span style='color:orange'><b>● Local (Offline)</b></span>", unsafe_allow_html=True)
    st.markdown("---")
    
    with st.expander("🛠️ Preferências", expanded=False):
        use_cache = st.toggle("Ativar Memória Rápida (Cache)", value=True)
        if st.button("🧹 Limpar Memória", use_container_width=True):
            document_cache.clear()
            st.toast("Memória limpa!", icon="🧹")
            time.sleep(0.5)
            st.rerun()

    st.markdown("---")
    st.markdown("### 🏷️ Regras de Renomeação")
    if "db_patterns" not in st.session_state:
        st.session_state["db_patterns"] = {}
        
    current_dict = st.session_state["db_patterns"]
    df_padroes = pd.DataFrame(list(current_dict.items()), columns=["origem", "destino"])
    
    df_editado = st.data_editor(
        df_padroes, num_rows="dynamic", use_container_width=True, hide_index=True, key="editor_patterns",
        column_config={
            "origem": st.column_config.TextColumn("📄 Texto no PDF", required=True, width="medium"),
            "destino": st.column_config.TextColumn("🏷️ Novo Nome", required=True, width="small")
        }
    )
    if st.button("💾 Salvar Regras", type="primary", use_container_width=True):
        novo_dict = {str(r["origem"]).strip().upper(): str(r["destino"]).strip().upper() for i, r in df_editado.iterrows() if r["origem"]}
        st.session_state["db_patterns"] = novo_dict # Atualiza local
        if sync_patterns_db(novo_dict): # Tenta atualizar nuvem
            st.toast("Salvo na Nuvem!", icon="☁️")
        else:
            st.toast("Salvo Localmente!", icon="💻")
        time.sleep(1)
        st.rerun()

def criar_dashboard_analitico():
    if "resultados" not in st.session_state: return
    st.markdown("---")
    st.markdown("### 📊 Dashboard Analítico")
    resultados = st.session_state["resultados"]
    logs = st.session_state.get("processed_logs", [])
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("📁 Arquivos Gerados", len(resultados))
    with col2: st.metric("📄 Páginas Processadas", sum(r.get('pages', 1) for r in resultados))
    with col3: st.metric("✅ Sucessos", len([log for log in logs if log[2] == "OK"]))
    with col4: st.metric("⚠️ Cache/Erros", len([log for log in logs if log[2] != "OK"]))

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 📎 Enviar PDFs")
uploaded_files = st.file_uploader("Arraste e solte seus PDFs aqui", type=["pdf"], accept_multiple_files=True, key="uploader")
col_up_a, col_up_b = st.columns([1,1])
with col_up_a: process_btn = st.button("🚀 Processar PDFs", type="primary", use_container_width=True)
with col_up_b: clear_session = st.button("♻️ Limpar sessão", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

if clear_session:
    if "session_folder" in st.session_state:
        try: shutil.rmtree(st.session_state["session_folder"])
        except: pass
    keys_to_clear = ["resultados", "session_folder", "novos_nomes", "processed_logs", "files_meta", "selected_files", "_manage_target"]
    for k in keys_to_clear:
        if k in st.session_state: del st.session_state[k]
    st.rerun()

if uploaded_files and process_btn:
    session_id = str(uuid.uuid4())
    session_folder = TEMP_FOLDER / session_id
    os.makedirs(session_folder, exist_ok=True)

    arquivos = []
    for f in uploaded_files:
        try: arquivos.append({"name": f.name, "bytes": f.read()})
        except: st.warning(f"Erro ao ler {f.name}, ignorado.")

    jobs = []
    
    # --- MODIFICAÇÃO PRINCIPAL: Prompt Otimizado para OCR ---
    # Instrução explícita para tratar o arquivo como imagem (scan)
    prompt = """
Documento: NOTA FISCAL BRASILEIRA (NF-e / DANFE), PDF ESCANEADO (imagem).

Use SOMENTE visão computacional (OCR). Não dependa de texto selecionável.

Extraia visualmente:
- "emitente": nome ou razão social do emissor (topo do documento)
- "numero_nota": número da nota fiscal (apenas dígitos)
- "cidade": cidade do emissor

Se não encontrar:
- emitente: "EMITENTE_DESCONHECIDO"
- numero_nota: "000"
- cidade: ""

Retorne APENAS um JSON válido com estas chaves exatas:
{ "emitente": "", "numero_nota": "", "cidade": "" }
"""

    # ---------------------------------------------------------

    for a in arquivos:
        try:
            reader = PdfReader(io.BytesIO(a["bytes"]))
            for idx, page in enumerate(reader.pages):
                b = io.BytesIO()
                w = PdfWriter()
                w.add_page(page)
                w.write(b)
                jobs.append({
                    "bytes": b.getvalue(), "prompt": prompt, "name": a["name"], "page_idx": idx,
                    "use_cache": st.session_state.get("use_cache", True)
                })
        except Exception as e:
            st.error(f"Erro ao ler {a['name']}: {e}")

    agrupados_dados = {}
    processed_logs = []
    processed_count = 0
    progress_bar = st.progress(0.0)
    progresso_text = st.empty()
    start_all = time.time()
    
    MAX_WORKERS = 4
    total_jobs = len(jobs) if jobs else 1
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_job = {executor.submit(processar_pagina_worker, job): job for job in jobs}
        
        for future in as_completed(future_to_job):
            processed_count += 1
            try:
                result = future.result()
                name = result["name"]
                idx = result["page_idx"]
                page_label = f"{name} (pág {idx+1})"
                
                if result["status"] == "ERRO":
                    dados_iniciais = {"emitente": "", "numero_nota": "000", "cidade": ""}
                else:
                    dados_iniciais = result["dados"]
                
                dados = validar_e_corrigir_dados(dados_iniciais, result.get("texto_real", ""))
                
                status_lbl = "CACHE" if result["status"] == "CACHE" else "OK"
                css_class = "success-log" if result["status"] == "OK" else "warning-log"
                
                log_info = f"{dados.get('numero_nota')} | {dados.get('emitente')[:20]}"
                if dados.get('numero_nota') == "000":
                    log_info = "REVISAR (Não encontrado)"
                    css_class = "error-log"
                    
                processed_logs.append((page_label, result["tempo"], status_lbl, log_info, result["provider"]))
                progresso_text.markdown(f"<span class='{css_class}'>✅ Processado: {page_label}</span>", unsafe_allow_html=True)

                emitente_raw = dados.get("emitente", "") or f"REVISAR_{idx}"
                numero_raw = dados.get("numero_nota", "") or "000"
                cidade_raw = dados.get("cidade", "") or ""
                
                numero = limpar_numero(numero_raw)
                
                if numero == "0" or numero == "000":
                    emitente = f"REVISAR_{limpar_emitente(emitente_raw)}"
                    key = (f"000_REV_{idx}_{uuid.uuid4().hex[:4]}", emitente)     
                else:
                    nome_map = substituir_nome_emitente(emitente_raw, cidade_raw)
                    emitente = limpar_emitente(nome_map)
                    key = (numero, emitente)

                agrupados_dados.setdefault(key, []).append({
                    "page_idx": idx, "pdf_bytes": result["pdf_bytes"], "file_origin": name
                })

            except Exception as e:
                st.error(f"Erro crítico no loop: {e}")
            
            progress_bar.progress(min(processed_count/total_jobs, 1.0))

    resultados = []
    files_meta = {}
    
    for (numero, emitente), pages_list in agrupados_dados.items():
        pages_list.sort(key=lambda x: (x['file_origin'], x['page_idx']))
        writer = PdfWriter()
        for p_data in pages_list:
            try:
                r = PdfReader(io.BytesIO(p_data["pdf_bytes"]))
                for p in r.pages: writer.add_page(p)
            except: continue
        
        emitente_safe = limpar_para_nome_arquivo(emitente)
        nome_pdf = f"DOC {numero}_{emitente_safe}.pdf"
        caminho = session_folder / nome_pdf
        with open(caminho, "wb") as f_out: writer.write(f_out)
            
        resultados.append({"file": nome_pdf, "numero": numero, "emitente": emitente, "pages": len(pages_list)})
        files_meta[nome_pdf] = {"numero": numero, "emitente": emitente, "pages": len(pages_list)}

    gc.collect() 
    st.session_state["resultados"] = resultados
    st.session_state["session_folder"] = str(session_folder)
    st.session_state["novos_nomes"] = {r["file"]: r["file"] for r in resultados}
    st.session_state["processed_logs"] = processed_logs
    st.session_state["files_meta"] = files_meta

    st.success(f"✅ Processamento concluído em {round(time.time() - start_all, 2)}s.")
    criar_dashboard_analitico()
    time.sleep(1)
    st.rerun()

if "resultados" in st.session_state:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🗂️ Gerenciamento")
    resultados = st.session_state["resultados"]
    session_folder = Path(st.session_state["session_folder"])
    novos_nomes = st.session_state.get("novos_nomes", {r["file"]: r["file"] for r in resultados})
    files_meta = st.session_state.get("files_meta", {})

    col1, col2, col3, col4 = st.columns([3, 2, 2, 3])
    with col1: q = st.text_input("🔎 Buscar", placeholder="Emitente ou número...", label_visibility="collapsed")
    with col2: sort_by = st.selectbox("Ordenar", ["Nome (A-Z)", "Nome (Z-A)", "Número (asc)", "Número (desc)"], label_visibility="collapsed")
    with col3: show_logs = st.toggle("Ver Logs")
    with col4:
        c_act = st.columns(3)
        with c_act[0]:
            if st.button("⬇️", help="Baixar Selecionados"):
                sel = st.session_state.get("selected_files", [])
                if sel:
                    mem = io.BytesIO()
                    with zipfile.ZipFile(mem, "w") as zf:
                        for f in sel:
                            src = session_folder / f
                            if src.exists(): zf.write(src, arcname=novos_nomes.get(f, f))
                    mem.seek(0)
                    st.download_button("💾", data=mem, file_name="selecionados.zip", mime="application/zip")
        with c_act[1]:
            if st.button("🗑️", help="Excluir Selecionados"):
                sel = st.session_state.get("selected_files", [])
                if sel:
                    for f in sel:
                        try: (session_folder/f).unlink()
                        except: pass
                        st.session_state["resultados"] = [r for r in st.session_state["resultados"] if r["file"] != f]
                    st.session_state["selected_files"] = []
                    st.rerun()
        with c_act[2]:
            if st.button("🔗", help="Unir Selecionados"):
                sel = st.session_state.get("selected_files", [])
                if len(sel) > 1:
                    m = PdfWriter()
                    for f in sorted(sel):
                        try:
                            r = PdfReader(str(session_folder/f))
                            for p in r.pages: m.add_page(p)
                        except: pass
                    nm = f"AGRUPADO_{int(time.time())}.pdf"
                    with open(session_folder/nm, "wb") as f: m.write(f)
                    meta = {"file": nm, "numero": "AGRUP", "emitente": "VÁRIOS", "pages": len(m.pages)}
                    st.session_state["resultados"].insert(0, meta)
                    st.session_state["files_meta"][nm] = meta
                    st.session_state["novos_nomes"][nm] = nm
                    st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    
    visible = resultados.copy()
    if q:
        q_up = q.strip().upper()
        visible = [r for r in visible if q_up in r["file"].upper() or q_up in r["emitente"].upper() or q_up in r["numero"]]
    
    if sort_by == "Nome (A-Z)": visible.sort(key=lambda x: x["file"])
    elif sort_by == "Nome (Z-A)": visible.sort(key=lambda x: x["file"], reverse=True)
    elif sort_by == "Número (asc)": visible.sort(key=lambda x: int(x["numero"]) if x["numero"].isdigit() else 0)
    else: visible.sort(key=lambda x: int(x["numero"]) if x["numero"].isdigit() else 0, reverse=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    if "selected_files" not in st.session_state: st.session_state["selected_files"] = []
    
    for r in visible:
        fn = r["file"]
        ck = fn in st.session_state["selected_files"]
        c = st.columns([0.05, 0.5, 0.25, 0.2])
        if c[0].checkbox("", value=ck, key=f"cb_{fn}"):
            if fn not in st.session_state["selected_files"]: st.session_state["selected_files"].append(fn)
        else:
            if fn in st.session_state["selected_files"]: st.session_state["selected_files"].remove(fn)
            
        novos_nomes[fn] = c[1].text_input(fn, value=novos_nomes.get(fn, fn), key=f"ren_{fn}", label_visibility="collapsed")
        c[2].caption(f"🏢 {r['emitente']}<br>🔢 Nº {r['numero']}", unsafe_allow_html=True)
        if c[3].button("⚙️ Editar", key=f"m_{fn}"):
            st.session_state["_manage_target"] = fn
            st.rerun()
            
    st.markdown("</div>", unsafe_allow_html=True)
    
    # --- GERENCIADOR DETALHADO ---
    if "_manage_target" in st.session_state:
        tgt = st.session_state["_manage_target"]
        if not (session_folder/tgt).exists():
            st.session_state.pop("_manage_target")
            st.rerun()
            
        st.markdown('<div class="manage-panel">', unsafe_allow_html=True)
        c1, c2 = st.columns([0.9, 0.1])
        c1.markdown(f"🛠️ Editando: **{tgt}**")
        if c2.button("❌", help="Fechar"):
            st.session_state.pop("_manage_target")
            st.rerun()
            
        with st.expander("👁️ Visualizar PDF", expanded=True):
            pdf_viewer(str(session_folder/tgt), height=600)
            
        # Ações de Separar/Remover Páginas
        try:
            reader_obj = PdfReader(str(session_folder/tgt))
            total_pgs = len(reader_obj.pages)
            pgs = [f"Pág {i+1}" for i in range(total_pgs)]
            sel_pgs = st.multiselect("Selecionar páginas para ação:", range(total_pgs), format_func=lambda x: pgs[x])
            
            # Recupera metadados do dicionário auxiliar (Correção CRÍTICA 2)
            meta_original = st.session_state["files_meta"].get(tgt, {"numero": "000", "emitente": "DESC"})
            
            c_a, c_b = st.columns(2)
            if c_a.button("✂️ Separar Selecionadas (Criar novo PDF)"):
                if sel_pgs:
                    nw = PdfWriter()
                    for i in sorted(sel_pgs): nw.add_page(reader_obj.pages[i])
                    
                    nn = f"{tgt[:-4]}_parte.pdf"
                    with open(session_folder/nn, "wb") as f: nw.write(f)
                    
                    nm = {
                        "file": nn, 
                        "numero": meta_original["numero"], 
                        "emitente": meta_original["emitente"], 
                        "pages": len(sel_pgs)
                    }
                    st.session_state["resultados"].append(nm)
                    st.session_state["files_meta"][nn] = nm
                    st.session_state["novos_nomes"][nn] = nn
                    st.success(f"Criado: {nn}")
            
            if c_b.button("🗑️ Remover Selecionadas (Do PDF atual)"):
                if sel_pgs:
                    nw = PdfWriter()
                    keep_count = 0
                    for i in range(total_pgs):
                        if i not in sel_pgs: 
                            nw.add_page(reader_obj.pages[i])
                            keep_count += 1
                    
                    if keep_count > 0:
                        with open(session_folder/tgt, "wb") as f: nw.write(f)
                        
                        # Atualiza estado e metadados
                        st.session_state["files_meta"][tgt]["pages"] = keep_count
                        for x in st.session_state["resultados"]:
                            if x["file"] == tgt: x["pages"] = keep_count
                        st.rerun()
                    else:
                        # Se removeu tudo, deleta o arquivo
                        (session_folder/tgt).unlink()
                        st.session_state["resultados"] = [x for x in st.session_state["resultados"] if x["file"] != tgt]
                        st.session_state.pop("_manage_target")
                        st.rerun()
        except Exception as e:
            st.error(f"Erro ao manipular PDF: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    if show_logs:
        st.caption("Logs recentes:")
        for log in st.session_state.get("processed_logs", [])[-10:]:
            st.text(f"{log[2]} | {log[3]} ({log[1]}s)")
    
    st.markdown("---")
    if st.button("⬇️ BAIXAR TUDO (.ZIP)", type="primary", use_container_width=True):
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, "w") as zf:
            for r in resultados:
                src = session_folder / r["file"]
                if src.exists(): zf.write(src, arcname=novos_nomes.get(r["file"], r["file"]))
        mem.seek(0)
        st.download_button("Clique para salvar o ZIP final", data=mem, file_name="notas_fiscais_processadas.zip", mime="application/zip")
