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
import gc  # <--- MELHORIA: Garbage Collector
import pandas as pd
from supabase import create_client, Client
from streamlit_pdf_viewer import pdf_viewer
from pathlib import Path
from pypdf import PdfReader, PdfWriter
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

st.title("Automatizador de Notas Fiscais PDF")

# =====================================================================
# SISTEMA DE CACHE INTELIGENTE
# =====================================================================
class DocumentCache:
    def __init__(self, cache_dir="./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_key(self, pdf_bytes, prompt):
        """Gera chave única baseada no conteúdo do PDF e prompt"""
        content_hash = hashlib.md5(pdf_bytes).hexdigest()
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        return f"{content_hash}_{prompt_hash}"
    
    def get(self, key):
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return None
        return None
    
    def set(self, key, data):
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except:
            pass
    
    def clear(self):
        """Limpa todo o cache"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except:
                pass

document_cache = DocumentCache()

# =====================================================================
# CONFIGURAÇÃO GEMINI
# =====================================================================
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

# =====================================================================
# CONFIGURAÇÕES GERAIS
# =====================================================================
PRIMARY = "#0f4c81"
ACCENT = "#6fb3b8"
BG = "#F7FAFC"
CARD_BG = "#FFFFFF"
TEXT_MUTED = "#6b7280"

TEMP_FOLDER = Path("./temp")
os.makedirs(TEMP_FOLDER, exist_ok=True)

MAX_TOTAL_PAGES = int(os.getenv("MAX_TOTAL_PAGES", "50"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
MIN_RETRY_DELAY = int(os.getenv("MIN_RETRY_DELAY", "5"))
MAX_RETRY_DELAY = int(os.getenv("MAX_RETRY_DELAY", "30"))

# =====================================================================
# GESTÃO DE PADRÕES (VIA SUPABASE)
# =====================================================================
@st.cache_resource
def init_supabase():
    """Conecta ao Supabase usando secrets do Streamlit"""
    try:
        # Tenta pegar dos secrets
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        return create_client(url, key)
    except Exception:
        return None

supabase = init_supabase()

def get_patterns_db():
    """Baixa os padrões do banco de dados"""
    if not supabase: return {}
    try:
        response = supabase.table("invoice_patterns").select("*").execute()
        return {item["origin"]: item["target"] for item in response.data}
    except Exception as e:
        st.error(f"Erro ao ler banco: {e}")
        return {}

def sync_patterns_db(new_dict):
    """Sincroniza a planilha da tela com o banco de dados"""
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

def validar_e_corrigir_dados(dados):
    """Valida e corrige dados extraídos da IA com heurísticas de recuperação"""
    if not isinstance(dados, dict):
        if isinstance(dados, list) and len(dados) > 0 and isinstance(dados[0], dict):
            dados = dados[0]
        else:
            return {"emitente": "ERRO_FORMATO", "numero_nota": "000", "cidade": ""}
    
    dados_norm = {}
    for k, v in dados.items():
        k_lower = k.lower().strip()
        if "numero" in k_lower or "nota" in k_lower: key = "numero_nota"
        elif "emitente" in k_lower or "prestador" in k_lower: key = "emitente"
        elif "cidade" in k_lower: key = "cidade"
        else: key = k
        dados_norm[key] = v
    dados = dados_norm

    emitente = str(dados.get('emitente', ''))
    if 'CPFL' in emitente.upper() or 'PAULISTA DE FORCA' in emitente.upper():
        emitente = 'CPFL'
    elif not emitente or emitente.upper() in ["NULL", "NONE", "NÃO IDENTIFICADO", "DESCONHECIDO"]:
        emitente = "EMITENTE_DESCONHECIDO"
    dados['emitente'] = emitente

    raw_num = str(dados.get('numero_nota', ''))
    so_digitos = re.sub(r'[^\d]', '', raw_num)
    
    if so_digitos:
        dados['numero_nota'] = so_digitos
    else:
        dados['numero_nota'] = "000000" 
        
    if 'cidade' not in dados:
        dados['cidade'] = ""
        
    return dados

# =====================================================================
# PROCESSAMENTO GEMINI
# =====================================================================
def calcular_delay(tentativa, error_msg):
    if "retry in" in error_msg.lower():
        try:
            return min(float(re.search(r"retry in (\d+\.?\d*)s", error_msg.lower()).group(1)) + 2, MAX_RETRY_DELAY)
        except:
            pass
    return min(MIN_RETRY_DELAY * (tentativa + 1), MAX_RETRY_DELAY)

def processar_pagina_gemini(prompt_instrucao, page_stream):
    """Processa uma página PDF com Gemini com retry e limpeza robusta de JSON"""
    for tentativa in range(MAX_RETRIES + 1):
        try:
            start = time.time()
            resp = model.generate_content(
                [prompt_instrucao, {"mime_type": "application/pdf", "data": page_stream.getvalue()}],
                generation_config={"response_mime_type": "application/json"},
                request_options={'timeout': 60}
            )
            tempo = round(time.time() - start, 2)
            
            # --- MELHORIA: BUSCA JSON MAIS ROBUSTA ---
            texto_raw = resp.text
            # Tenta encontrar o primeiro { e o último } para isolar o JSON
            try:
                idx_inicio = texto_raw.find('{')
                idx_fim = texto_raw.rfind('}')
                
                if idx_inicio != -1 and idx_fim != -1:
                    texto_limpo = texto_raw[idx_inicio : idx_fim + 1]
                    dados = json.loads(texto_limpo)
                else:
                    # Fallback
                    dados = json.loads(texto_raw)
            except json.JSONDecodeError:
                 # Último recurso: remover markdown
                texto_limpo = texto_raw.replace("```json", "").replace("```", "").strip()
                dados = json.loads(texto_limpo)
            # ----------------------------------------

            return dados, True, tempo, "Gemini"

        except ResourceExhausted as e:
            delay = calcular_delay(tentativa, str(e))
            st.sidebar.warning(f"⚠️ Quota excedida (tentativa {tentativa + 1}/{MAX_RETRIES}). Aguardando {delay}s...")
            time.sleep(delay)
        except Exception as e:
            if tentativa < MAX_RETRIES:
                st.sidebar.warning(f"⚠️ Erro Gemini (tentativa {tentativa + 1}/{MAX_RETRIES}): {str(e)}")
                time.sleep(MIN_RETRY_DELAY)
            else:
                return {"error": str(e)}, False, 0, "Gemini"
    
    return {"error": "Falha máxima de tentativas"}, False, 0, "Gemini"

def processar_pagina_worker(job_data):
    pdf_bytes = job_data["bytes"]
    prompt = job_data["prompt"]
    name = job_data["name"]
    page_idx = job_data["page_idx"]
    
    cache_key = document_cache.get_cache_key(pdf_bytes, prompt)
    cached_result = document_cache.get(cache_key)
    
    if cached_result and job_data["use_cache"]:
        return {
            "status": "CACHE",
            "dados": cached_result['dados'],
            "tempo": cached_result['tempo'],
            "provider": cached_result['provider'],
            "name": name,
            "page_idx": page_idx,
            "pdf_bytes": pdf_bytes
        }

    page_stream = io.BytesIO(pdf_bytes)
    dados, ok, tempo, provider = processar_pagina_gemini(prompt, page_stream)
    
    if ok and "error" not in dados:
        document_cache.set(cache_key, {
            'dados': dados,
            'tempo': tempo,
            'provider': provider
        })
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
            "error_msg": dados.get("error", "Erro desconhecido"),
            "pdf_bytes": pdf_bytes
        }

# =====================================================================
# UI & MAIN FLOW
# =====================================================================
with st.sidebar:
    st.markdown("### ⚙️ Painel de Controle")
    if supabase:
        st.markdown("Status: <span style='color:green'><b>● Conectado à Nuvem</b></span>", unsafe_allow_html=True)
    else:
        st.error("🔴 Sem conexão com Supabase")
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
    st.caption("Defina como o robô deve renomear os arquivos encontrados.")

    if supabase:
        current_dict = st.session_state.get("db_patterns", {})
        df_padroes = pd.DataFrame(list(current_dict.items()), columns=["origem", "destino"])
        
        df_editado = st.data_editor(
            df_padroes,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="editor_patterns",
            column_config={
                "origem": st.column_config.TextColumn("📄 Texto no PDF", required=True, width="medium"),
                "destino": st.column_config.TextColumn("🏷️ Novo Nome", required=True, width="small")
            }
        )

        col_save, col_info = st.columns([0.7, 0.3])
        with col_save:
            if st.button("💾 Salvar Regras", type="primary", use_container_width=True):
                novo_dict = {}
                for index, row in df_editado.iterrows():
                    try:
                        chave = str(row["origem"]).strip().upper()
                        valor = str(row["destino"]).strip().upper()
                        if chave and valor and chave != "NONE" and chave != "NAN":
                            novo_dict[chave] = valor
                    except: continue
                
                with st.spinner("Sincronizando com a nuvem..."):
                    if sync_patterns_db(novo_dict):
                        st.session_state["db_patterns"] = novo_dict
                        st.toast("Regras salvas com sucesso!", icon="✅")
                        time.sleep(1)
                        st.rerun()
        with col_info:
            st.markdown(f"<div style='text-align:center; font-size:12px; color:gray; padding-top:10px'>{len(current_dict)} regras</div>", unsafe_allow_html=True)

def criar_dashboard_analitico():
    if "resultados" not in st.session_state: return
    st.markdown("---")
    st.markdown("### 📊 Dashboard Analítico")
    resultados = st.session_state["resultados"]
    logs = st.session_state.get("processed_logs", [])
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("📁 Arquivos Processados", len(resultados))
    with col2: st.metric("📄 Total de Páginas", sum(r.get('pages', 1) for r in resultados))
    with col3: st.metric("✅ Sucessos", len([log for log in logs if log[2] == "OK"]))
    with col4: st.metric("❌ Erros", len([log for log in logs if log[2] != "OK"]))

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 📎 Enviar PDFs e processar ")
uploaded_files = st.file_uploader("Selecione arquivos PDF", type=["pdf"], accept_multiple_files=True, key="uploader")
col_up_a, col_up_b = st.columns([1,1])
with col_up_a: process_btn = st.button("🚀 Processar PDFs")
with col_up_b: clear_session = st.button("♻️ Limpar sessão")
st.markdown("</div>", unsafe_allow_html=True)

if clear_session:
    if "session_folder" in st.session_state:
        try: shutil.rmtree(st.session_state["session_folder"])
        except: pass
    for k in ["resultados", "session_folder", "novos_nomes", "processed_logs", "files_meta", "selected_files", "_manage_target"]:
        if k in st.session_state: del st.session_state[k]
    st.success("Sessão limpa.")
    st.rerun()

if uploaded_files and process_btn:
    session_id = str(uuid.uuid4())
    session_folder = TEMP_FOLDER / session_id
    os.makedirs(session_folder, exist_ok=True)

    arquivos = []
    for f in uploaded_files:
        try: arquivos.append({"name": f.name, "bytes": f.read()})
        except: st.warning(f"Erro ao ler {f.name}, ignorado.")

    total_paginas = 0
    for a in arquivos:
        try: total_paginas += len(PdfReader(io.BytesIO(a["bytes"])).pages)
        except: pass
    st.info(f"📄 Total de páginas a processar: {total_paginas}")

    agrupados_dados = {}
    resultados_meta = []
    processed_logs = []
    processed_count = 0
    progress_bar = st.progress(0.0)
    progresso_text = st.empty()
    start_all = time.time()

    prompt = (
        "Atue como um Auditor de Notas Fiscais Brasileiro. Analise a imagem do documento. "
        "Sua missão é extrair dados mesmo em documentos com layout complexo ou baixa qualidade. "
        "1. EMITENTE (Prestador): Busque o nome da empresa que PRESTOU o serviço. "
        "   - Dica: Procure blocos como 'Prestador de Serviços', 'Razão Social' ou o logo no cabeçalho. "
        "   - Ignore: 'Prefeitura', 'Tomador', 'Receita Federal'. "
        "2. NÚMERO DA NOTA: Identifique o identificador único do documento. "
        "   - Dica: Procure por rótulos 'Nº', 'Número', 'NFS-e', 'DANFE', 'Fatura'. "
        "   - Geralmente está no topo direito ou em destaque negrito. "
        "3. CIDADE: A cidade do prestador do serviço. "
        "FORMATO DE RESPOSTA (JSON Puro): "
        "Responda APENAS um objeto JSON com as chaves: 'emitente', 'numero_nota', 'cidade'. "
        "Se um campo estiver muito ilegível, tente inferir pelo contexto. Use 'DESCONHECIDO' apenas em último caso."
    )

    jobs = []
    for a in arquivos:
        try:
            reader = PdfReader(io.BytesIO(a["bytes"]))
            for idx, page in enumerate(reader.pages):
                b = io.BytesIO()
                w = PdfWriter()
                w.add_page(page)
                w.write(b)
                jobs.append({
                    "bytes": b.getvalue(),
                    "prompt": prompt,
                    "name": a["name"],
                    "page_idx": idx,
                    "use_cache": st.session_state.get("use_cache", True)
                })
        except Exception as e:
            processed_logs.append((a["name"], 0, "ERRO_LEITURA", str(e), "System"))

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
                    processed_logs.append((page_label, result["tempo"], "ERRO_IA", result["error_msg"], result["provider"]))
                    progresso_text.markdown(f"<span class='error-log'>⚠️ {page_label} — FALHA (Salvo para revisão)</span>", unsafe_allow_html=True)
                    dados = {"emitente": f"REVISAR_{name}", "numero_nota": "000", "cidade": ""}
                else:
                    dados = validar_e_corrigir_dados(result["dados"])
                    status_lbl = "CACHE" if result["status"] == "CACHE" else "OK"
                    css_class = "success-log" if result["status"] == "OK" else "warning-log"
                    processed_logs.append((page_label, result["tempo"], status_lbl, f"{dados.get('numero_nota')} / {dados.get('emitente')}", result["provider"]))
                    progresso_text.markdown(f"<span class='{css_class}'>✅ {page_label} — {status_lbl}</span>", unsafe_allow_html=True)

                emitente_raw = dados.get("emitente", "") or f"REVISAR_{idx}"
                numero_raw = dados.get("numero_nota", "") or "000"
                cidade_raw = dados.get("cidade", "") or ""
                numero = limpar_numero(numero_raw)
                
                if result["status"] == "ERRO" or numero == "0":
                     emitente = emitente_raw 
                else:
                     nome_map = substituir_nome_emitente(emitente_raw, cidade_raw)
                     emitente = limpar_emitente(nome_map)

                if numero == "0" or numero == "000":
                    key = (f"000_REV_{idx}", emitente)
                else:
                    key = (numero, emitente)

                agrupados_dados.setdefault(key, []).append({
                    "page_idx": idx,
                    "pdf_bytes": result["pdf_bytes"],
                    "file_origin": name
                })
                
                status_final = "FALHA_SALVA" if result["status"] == "ERRO" else ("CACHE" if result["status"] == "CACHE" else "OK")
                resultados_meta.append({
                    "arquivo_origem": name, "pagina": idx+1, "status": status_final, 
                    "tempo_s": round(result["tempo"], 2), "provider": result["provider"]
                })

            except Exception as e:
                st.error(f"Erro crítico: {e}")
            progress_bar.progress(min(processed_count/total_jobs, 1.0))

    resultados = []
    files_meta = {}
    
    for (numero, emitente), pages_list in agrupados_dados.items():
        if not numero or numero == "0": continue
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

    # --- MELHORIA: LIMPEZA DE MEMÓRIA APÓS PROCESSAMENTO PESADO ---
    gc.collect() 
    # -------------------------------------------------------------

    st.session_state["resultados"] = resultados
    st.session_state["session_folder"] = str(session_folder)
    st.session_state["novos_nomes"] = {r["file"]: r["file"] for r in resultados}
    st.session_state["processed_logs"] = processed_logs
    st.session_state["files_meta"] = files_meta

    st.toast(f"Processamento Finalizado! {len(resultados)} arquivos gerados.", icon="🎉")
    st.success(f"✅ Processamento concluído em {round(time.time() - start_all, 2)}s.")
    
    criar_dashboard_analitico()
    st.rerun()

if "resultados" in st.session_state:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Gerenciamento — selecione e aplique ações")
    resultados = st.session_state["resultados"]
    session_folder = Path(st.session_state["session_folder"])
    novos_nomes = st.session_state.get("novos_nomes", {r["file"]: r["file"] for r in resultados})
    files_meta = st.session_state.get("files_meta", {})

    col1, col2, col3, col4 = st.columns([3, 2, 2, 3])
    with col1: q = st.text_input("🔎 Buscar", value="", placeholder="parte do nome, emitente ou número")
    with col2: sort_by = st.selectbox("Ordenar por", ["Nome (A-Z)", "Nome (Z-A)", "Número (asc)", "Número (desc)"], index=0)
    with col3: show_logs = st.checkbox("Mostrar logs", value=False)
    with col4:
        st.write("")
        top_actions_cols = st.columns([1, 1, 1])
        with top_actions_cols[0]:
            if st.button("⬇️ Zip"):
                sel = st.session_state.get("selected_files", [])
                if not sel: st.warning("Selecione itens.")
                else:
                    mem = io.BytesIO()
                    with zipfile.ZipFile(mem, "w") as zf:
                        for f in sel:
                            src = session_folder / f
                            if src.exists(): zf.write(src, arcname=novos_nomes.get(f, f))
                    mem.seek(0)
                    st.download_button("💾", data=mem, file_name="selecionadas.zip", mime="application/zip")

        with top_actions_cols[1]:
            if st.button("🗑️ Del"):
                sel = st.session_state.get("selected_files", [])
                if not sel: st.warning("Selecione itens.")
                else:
                    for f in sel:
                        src = session_folder / f
                        try:
                            if src.exists(): src.unlink()
                        except: pass
                        st.session_state["resultados"] = [r for r in st.session_state["resultados"] if r["file"] != f]
                        st.session_state["novos_nomes"].pop(f, None)
                        st.session_state["files_meta"].pop(f, None)
                    st.session_state["selected_files"] = []
                    st.rerun()

        with top_actions_cols[2]:
            if st.button("🔗 Unir"):
                sel = st.session_state.get("selected_files", [])
                if len(sel) < 2: st.warning("Selecione + de 1")
                else:
                    try:
                        merger = PdfWriter()
                        for fname in sorted(sel):
                            src = session_folder / fname
                            if src.exists():
                                reader = PdfReader(str(src))
                                for page in reader.pages: merger.add_page(page)
                        new_name = f"AGRUPADO_{int(time.time())}.pdf"
                        with open(session_folder / new_name, "wb") as f: merger.write(f)
                        new_meta = {"file": new_name, "numero": "AGRUP", "emitente": "VÁRIOS", "pages": len(merger.pages)}
                        st.session_state["resultados"].insert(0, new_meta)
                        st.session_state["files_meta"][new_name] = new_meta
                        st.session_state["novos_nomes"][new_name] = new_name
                        st.rerun()
                    except Exception as e: st.error(f"Erro: {e}")

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
    st.markdown("### 📁 Notas processadas")
    if "selected_files" not in st.session_state: st.session_state["selected_files"] = []

    for r in visible:
        fname = r["file"]
        meta = files_meta.get(fname, {})
        cols = st.columns([0.05, 0.50, 0.25, 0.20])
        
        checked = fname in st.session_state.get("selected_files", [])
        cb = cols[0].checkbox("", value=checked, key=f"cb_{fname}")
        if cb and fname not in st.session_state["selected_files"]: st.session_state["selected_files"].append(fname)
        if (not cb) and fname in st.session_state["selected_files"]: st.session_state["selected_files"].remove(fname)

        novos_nomes[fname] = cols[1].text_input(label=fname, value=novos_nomes.get(fname, fname), key=f"rename_{fname}", label_visibility="collapsed")
        cols[2].markdown(f"<div class='small-note'>{meta.get('emitente','-')}<br>Nº {meta.get('numero','-')} • {r.get('pages',1)} pág(s)</div>", unsafe_allow_html=True)
        if cols[3].button("⚙️ Gerenciar", key=f"manage_{fname}"):
            st.session_state["_manage_target"] = fname
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    if "_manage_target" in st.session_state:
        manage_target = st.session_state["_manage_target"]
        if not any(r["file"] == manage_target for r in st.session_state.get("resultados", [])):
            st.session_state.pop("_manage_target", None)
            st.rerun()
        
        st.markdown('<div class="manage-panel">', unsafe_allow_html=True)
        col_tit, col_x = st.columns([0.9, 0.1])
        col_tit.markdown(f"### ⚙️ Gerenciar: `{manage_target}`")
        if col_x.button("❌", key=f"close_{manage_target}"):
            st.session_state.pop("_manage_target", None)
            st.rerun()
        
        file_path = session_folder / manage_target
        with st.expander("👁️ Visualizar Arquivo Completo", expanded=True):
            if file_path.exists(): pdf_viewer(input=str(file_path), width=700, height=800)
            else: st.warning("Arquivo não encontrado no disco.")

        try:
            reader = PdfReader(str(file_path))
            pages_info = [{"idx": i, "label": f"Página {i+1}"} for i in range(len(reader.pages))]
        except: pages_info = []
        
        if pages_info:
            sel_key = f"_manage_sel_{manage_target}"
            if sel_key not in st.session_state: st.session_state[sel_key] = []
            
            col_sel, col_actions = st.columns([1, 2])
            with col_sel:
                st.markdown("**Selecionar páginas:**")
                for page in pages_info:
                    if st.checkbox(page["label"], value=page["idx"] in st.session_state[sel_key], key=f"{sel_key}_{page['idx']}"):
                        if page["idx"] not in st.session_state[sel_key]: st.session_state[sel_key].append(page["idx"])
                    else:
                        if page["idx"] in st.session_state[sel_key]: st.session_state[sel_key].remove(page["idx"])
            
            with col_actions:
                st.write(f"📑 Selecionadas: **{len(st.session_state.get(sel_key, []))}**")
                new_name_key = f"_newname_{manage_target}"
                if new_name_key not in st.session_state: st.session_state[new_name_key] = f"{manage_target.rsplit('.pdf', 1)[0]}_parte.pdf"
                new_name = st.text_input("Nome:", key=new_name_key)
                
                c1, c2 = st.columns(2)
                if c1.button("➗ Separar", key=f"sep_{manage_target}"):
                    sel = sorted(st.session_state.get(sel_key, []))
                    if sel:
                        try:
                            nw = PdfWriter()
                            r = PdfReader(str(file_path))
                            for i in sel: nw.add_page(r.pages[i])
                            with open(session_folder / new_name, "wb") as f: nw.write(f)
                            new_meta = {"file": new_name, "numero": files_meta.get(manage_target, {}).get("numero", ""), "emitente": files_meta.get(manage_target, {}).get("emitente", ""), "pages": len(sel)}
                            st.session_state["resultados"].append(new_meta)
                            st.session_state["files_meta"][new_name] = new_meta
                            st.session_state["novos_nomes"][new_name] = new_name
                            st.session_state[sel_key] = []
                            st.rerun()
                        except Exception as e: st.error(str(e))
                
                if c2.button("🗑️ Remover", key=f"rem_{manage_target}"):
                    sel = sorted(st.session_state.get(sel_key, []))
                    if sel:
                        try:
                            nw = PdfWriter()
                            r = PdfReader(str(file_path))
                            for i in range(len(r.pages)):
                                if i not in sel: nw.add_page(r.pages[i])
                            if len(nw.pages) > 0:
                                with open(file_path, "wb") as f: nw.write(f)
                                st.session_state["files_meta"][manage_target]["pages"] = len(nw.pages)
                                for res in st.session_state["resultados"]: 
                                    if res["file"] == manage_target: res["pages"] = len(nw.pages)
                                st.session_state[sel_key] = []
                                st.rerun()
                            else:
                                file_path.unlink()
                                st.session_state["resultados"] = [r for r in st.session_state["resultados"] if r["file"] != manage_target]
                                st.session_state.pop("_manage_target", None)
                                st.rerun()
                        except Exception as e: st.error(str(e))
        st.markdown("</div>", unsafe_allow_html=True)

    criar_dashboard_analitico()

    if show_logs and st.session_state.get("processed_logs"):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📝 Logs")
        for entry in st.session_state["processed_logs"][-200:]:
            label, t, status, info, provider = (entry + ("", "", ""))[:5]
            css = "success-log" if status == "OK" else "warning-log"
            st.markdown(f"<div class='{css}'>{status} | {label} — {info}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w") as zf:
        for r in st.session_state.get("resultados", []):
            src = session_folder / r["file"]
            if src.exists():
                fn = st.session_state.get("novos_nomes", {}).get(r["file"], r["file"])
                if not fn.lower().endswith(".pdf"): fn += ".pdf"
                zf.write(src, arcname=fn)
    mem.seek(0)
    st.download_button(label="⬇️ Baixar Tudo (.zip)", data=mem, file_name="notas.zip", mime="application/zip", use_container_width=True, type="primary")