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
# NORMALIZAÇÃO E SUBSTITUIÇÕES
# =====================================================================
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
        # Converte a lista do banco em um dicionário {origem: destino}
        return {item["origin"]: item["target"] for item in response.data}
    except Exception as e:
        st.error(f"Erro ao ler banco: {e}")
        return {}

def sync_patterns_db(new_dict):
    """Sincroniza a planilha da tela com o banco de dados"""
    if not supabase: return False
    try:
        # 1. Pega o que tem no banco hoje para comparar
        current_data = supabase.table("invoice_patterns").select("origin").execute()
        db_keys = {row['origin'] for row in current_data.data}
        new_keys = set(new_dict.keys())

        # 2. Deleta o que você removeu da planilha
        to_delete = list(db_keys - new_keys)
        if to_delete:
            supabase.table("invoice_patterns").delete().in_("origin", to_delete).execute()

        # 3. Atualiza/Insere o que está na planilha
        upsert_data = [{"origin": k, "target": v} for k, v in new_dict.items()]
        if upsert_data:
            supabase.table("invoice_patterns").upsert(upsert_data, on_conflict="origin").execute()
            
        return True
    except Exception as e:
        st.error(f"Erro ao salvar: {e}")
        return False

# Carrega os padrões na memória ao iniciar (Session State)
if "db_patterns" not in st.session_state:
    st.session_state["db_patterns"] = get_patterns_db()

# Carrega os padrões para a memória ao iniciar o script

def _normalizar_texto(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII")
    s = re.sub(r"[^A-Z0-9 ]+", " ", s.upper())
    return re.sub(r"\s+", " ", s).strip()

def substituir_nome_emitente(nome_raw: str, cidade_raw: str = None) -> str:
    nome_norm = _normalizar_texto(nome_raw)
    cidade_norm = _normalizar_texto(cidade_raw) if cidade_raw else None
    
    # 1. Regra Fixa (Ex: Sabará)
    if "SABARA" in nome_norm:
        return f"SB_{cidade_norm.split()[0]}" if cidade_norm else "SB"
        
    # 2. Regra Dinâmica (Vinda do Supabase/Memória)
    patterns = st.session_state.get("db_patterns", {})
    
    for padrao, substituto in patterns.items():
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
    
def limpar_para_nome_arquivo(texto):
    """Remove caracteres proibidos pelo sistema operacional"""
    if not texto: return "DESCONHECIDO"
    # Remove caracteres proibidos no Windows: \ / : * ? " < > |
    texto = re.sub(r'[\\/*?:"<>|]', "", texto)
    # Remove acentos e caracteres estranhos
    texto = unicodedata.normalize("NFKD", texto).encode("ASCII", "ignore").decode("ASCII")
    return texto.strip()[:60] # Limita a 60 caracteres para não dar erro de path longo

def validar_e_corrigir_dados(dados):
    """Valida e corrige dados extraídos da IA"""
    if not isinstance(dados, dict):
        dados = {}
    
    required_fields = ['emitente', 'numero_nota', 'cidade']
    
    # Verifica campos obrigatórios
    for field in required_fields:
        if field not in dados or not dados[field]:
            dados[field] = "NÃO_IDENTIFICADO"
    
    # Correções comuns
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
# PROCESSAMENTO GEMINI (SIMPLIFICADO)
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
            
            # --- CORREÇÃO CIRÚRGICA AQUI ---
            # Em vez de apenas strip(), buscamos o primeiro bloco que parece um JSON { ... }
            texto_raw = resp.text
            match = re.search(r"\{.*\}", texto_raw, re.DOTALL)
            
            if match:
                texto_limpo = match.group(0) # Pega só o que está entre chaves
            else:
                texto_limpo = texto_raw # Tenta o texto todo se não achar chaves

            try:
                dados = json.loads(texto_limpo)
            except json.JSONDecodeError:
                # Última tentativa de limpeza forçada
                texto_limpo = texto_raw.replace("```json", "").replace("```", "").strip()
                dados = json.loads(texto_limpo)
            # -------------------------------

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
                # Retorna erro formatado para não quebrar o worker
                return {"error": str(e)}, False, 0, "Gemini"
    
    return {"error": "Falha máxima de tentativas"}, False, 0, "Gemini"
def processar_pagina_worker(job_data):
    """Função executada em paralelo para processar uma página"""
    pdf_bytes = job_data["bytes"]
    prompt = job_data["prompt"]
    name = job_data["name"]
    page_idx = job_data["page_idx"]
    
    # 1. Verificar Cache
    cache_key = document_cache.get_cache_key(pdf_bytes, prompt)
    cached_result = document_cache.get(cache_key)
    
    # Se tiver cache e o usuário quiser usar
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

    # 2. Se não tiver cache, chama o Gemini
    page_stream = io.BytesIO(pdf_bytes)
    dados, ok, tempo, provider = processar_pagina_gemini(prompt, page_stream)
    
    # Salvar no cache se deu certo
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
        # ALTERAÇÃO AQUI: Retornamos o pdf_bytes também no erro
        return {
            "status": "ERRO",
            "dados": dados,
            "tempo": tempo,
            "provider": provider,
            "name": name,
            "page_idx": page_idx,
            "error_msg": dados.get("error", "Erro desconhecido"),
            "pdf_bytes": pdf_bytes  # Devolver o arquivo mesmo com erro
        }

# =====================================================================
# SIDEBAR CONFIGURAÇÕES (VERSÃO ESTILOSA)
# =====================================================================
with st.sidebar:
    st.markdown("### ⚙️ Painel de Controle")
    
    # --- Status do Banco de Dados ---
    if supabase:
        st.markdown("Status: <span style='color:green'><b>● Conectado à Nuvem</b></span>", unsafe_allow_html=True)
    else:
        st.error("🔴 Sem conexão com Supabase")

    st.markdown("---")
    
    # --- Configurações Gerais ---
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
        # 1. Prepara dados
        current_dict = st.session_state.get("db_patterns", {})
        df_padroes = pd.DataFrame(
            list(current_dict.items()), 
            columns=["origem", "destino"] # Nomes internos simples
        )

        # 2. Planilha Estilosa
        df_editado = st.data_editor(
            df_padroes,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="editor_patterns",
            # AQUI ESTÁ A MÁGICA DO ESTILO:
            column_config={
                "origem": st.column_config.TextColumn(
                    "📄 Texto no PDF", # Título bonito
                    help="O texto que aparece na nota fiscal (ex: RAZAO SOCIAL LTDA)",
                    placeholder="Ex: ELETROPAULO...",
                    required=True,
                    width="medium"
                ),
                "destino": st.column_config.TextColumn(
                    "🏷️ Novo Nome", # Título bonito
                    help="Como o arquivo será salvo (ex: ENEL)",
                    placeholder="Ex: ENEL",
                    required=True,
                    width="small"
                )
            }
        )

        # 3. Botão de Salvar com destaque
        col_save, col_info = st.columns([0.7, 0.3])
        
        with col_save:
            if st.button("💾 Salvar Regras", type="primary", use_container_width=True):
                # Reconstrói o dicionário
                novo_dict = {}
                for index, row in df_editado.iterrows():
                    try:
                        # Força maiúsculo e remove espaços extras
                        chave = str(row["origem"]).strip().upper()
                        valor = str(row["destino"]).strip().upper()
                        
                        if chave and valor and chave != "NONE" and chave != "NAN":
                            novo_dict[chave] = valor
                    except:
                        continue
                
                # Envia para o Supabase
                with st.spinner("Sincronizando com a nuvem..."):
                    if sync_patterns_db(novo_dict):
                        st.session_state["db_patterns"] = novo_dict
                        st.toast("Regras salvas com sucesso!", icon="✅")
                        time.sleep(1)
                        st.rerun()
        
        with col_info:
            st.markdown(f"<div style='text-align:center; font-size:12px; color:gray; padding-top:10px'>{len(current_dict)} regras</div>", unsafe_allow_html=True)
# =====================================================================
# DASHBOARD ANALÍTICO
# =====================================================================
def criar_dashboard_analitico():
    """Cria dashboard com métricas e analytics"""
    if "resultados" not in st.session_state:
        return
    
    st.markdown("---")
    st.markdown("### 📊 Dashboard Analítico")
    
    resultados = st.session_state["resultados"]
    logs = st.session_state.get("processed_logs", [])
    
    # Métricas principais
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
    
    # Estatísticas por emitente
    if resultados:
        st.markdown("#### 📈 Emitentes Mais Frequentes")
        emitentes = {}
        for r in resultados:
            emitente = r.get('emitente', 'Desconhecido')
            emitentes[emitente] = emitentes.get(emitente, 0) + 1
        
        for emitente, count in sorted(emitentes.items(), key=lambda x: x[1], reverse=True)[:5]:
            st.write(f"`{emitente}`: {count} documento(s)")

# =====================================================================
# UPLOAD E PROCESSAMENTO
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

    # Dicionário agora vai guardar metadados para ordenação
    agrupados_dados = {} 
    
    resultados_meta = []
    processed_logs = []
    processed_count = 0
    
    progress_bar = st.progress(0.0)
    progresso_text = st.empty()
    start_all = time.time()

    prompt = (
        "Você é um extrator de dados OCR. Analise esta página. "
        "Extraia: 'emitente' (Nome fantasia principal), 'numero_nota' (Apenas dígitos) e 'cidade'. "
        "REGRAS CRÍTICAS: "
        "1. Se não encontrar o número da nota explicitamente, retorne null. "
        "2. Se não encontrar o emitente, retorne null. "
        "Responda EXCLUSIVAMENTE o JSON bruto (sem markdown ```json): "
        "{\"emitente\": \"string ou null\", \"numero_nota\": \"string ou null\", \"cidade\": \"string ou null\"}"
    )

    # 1. Preparar trabalhos
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
                
                # Importante: Estamos mandando o 'idx' (número da página) junto
                jobs.append({
                    "bytes": page_bytes,
                    "prompt": prompt,
                    "name": name,
                    "page_idx": idx, 
                    "use_cache": st.session_state.get("use_cache", True)
                })
        except Exception as e:
            processed_logs.append((name, 0, "ERRO_LEITURA", str(e), "System"))

    # 2. Executar em Paralelo
    MAX_WORKERS = 4
    total_jobs = len(jobs) if jobs else 1
    
    st.info(f"🚀 Iniciando processamento de {len(jobs)} páginas...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_job = {executor.submit(processar_pagina_worker, job): job for job in jobs}
        
        for future in as_completed(future_to_job):
            processed_count += 1
            try:
                result = future.result()
                name = result["name"]
                idx = result["page_idx"]
                page_label = f"{name} (pág {idx+1})"
                
                # --- LÓGICA MODIFICADA PARA NÃO DESCARTAR ERROS ---
                if result["status"] == "ERRO":
                    # Loga o erro, mas define dados padrão para não perder o arquivo
                    processed_logs.append((page_label, result["tempo"], "ERRO_IA", result["error_msg"], result["provider"]))
                    progresso_text.markdown(f"<span class='error-log'>⚠️ {page_label} — FALHA (Salvo para revisão)</span>", unsafe_allow_html=True)
                    
                    # Define dados para criar o arquivo "REVISAR"
                    dados = {
                        "emitente": f"REVISAR_{name}", # Nome do arquivo original para você achar fácil
                        "numero_nota": "000",          # Número 0 para ficar no topo ou fim da lista
                        "cidade": ""
                    }
                else:
                    dados = result["dados"]
                    dados = validar_e_corrigir_dados(dados)
                    status_lbl = "CACHE" if result["status"] == "CACHE" else "OK"
                    css_class = "success-log" if result["status"] == "OK" else "warning-log"
                    
                    emitente_raw = dados.get("emitente", "") or "DESCONHECIDO"
                    numero_raw = dados.get("numero_nota", "") or "000"
                    processed_logs.append((page_label, result["tempo"], status_lbl, f"{numero_raw} / {emitente_raw}", result["provider"]))
                    progresso_text.markdown(f"<span class='{css_class}'>✅ {page_label} — {status_lbl}</span>", unsafe_allow_html=True)

                # --- PROCESSAMENTO COMUM PARA ERROS E SUCESSOS ---
                emitente_raw = dados.get("emitente", "") or f"REVISAR_{idx}"
                numero_raw = dados.get("numero_nota", "") or "000"
                cidade_raw = dados.get("cidade", "") or ""

                numero = limpar_numero(numero_raw)
                
                # Se for erro ou revisão, não tenta substituir nome
                if result["status"] == "ERRO" or numero == "0":
                     emitente = emitente_raw 
                else:
                     nome_map = substituir_nome_emitente(emitente_raw, cidade_raw)
                     emitente = limpar_emitente(nome_map)

                # Para evitar que erros de arquivos diferentes se misturem no mesmo PDF,
                # se for erro (numero 0), adicionamos o ID da página na chave
                if numero == "0" or numero == "000":
                    key = (f"000_REV_{idx}", emitente) # Chave única para erros
                else:
                    key = (numero, emitente)

                agrupados_dados.setdefault(key, []).append({
                    "page_idx": idx,
                    "pdf_bytes": result["pdf_bytes"],
                    "file_origin": name
                })

                # Adiciona log para métricas (mesmo se falhou, conta como processado)
                if result["status"] == "ERRO":
                     status_final = "FALHA_SALVA"
                else:
                     status_final = "CACHE" if result["status"] == "CACHE" else "OK"

                resultados_meta.append({
                    "arquivo_origem": name,
                    "pagina": idx+1,
                    "emitente_detectado": emitente_raw,
                    "numero_detectado": numero_raw,
                    "status": status_final,
                    "tempo_s": round(result["tempo"], 2),
                    "provider": result["provider"]
                })

            except Exception as e:
                st.error(f"Erro crítico: {e}")
            progress_bar.progress(min(processed_count/total_jobs, 1.0))

    # 3. Gerar arquivos finais (COM ORDENAÇÃO CORRIGIDA)
    resultados = []
    files_meta = {}
    
    for (numero, emitente), pages_list in agrupados_dados.items():
        if not numero or numero == "0":
            continue
        #se for nossa chave de erro criada acima (ex: 000_REV_1), limpamos para o nome do arquivo ficar bonito
        if "REV_" in str(numero):
             numero_display = "000_REVISAR"
        else:
             numero_display = numero   
        # --- A CORREÇÃO MÁGICA ---
        # Ordena a lista de páginas baseada no 'page_idx' (número da página original)
        # Isso garante que a Página 1 venha antes da Página 2, mesmo que a 2 tenha sido processada antes.
        pages_list.sort(key=lambda x: (x['file_origin'], x['page_idx']))
        # -------------------------

        writer = PdfWriter()
        for p_data in pages_list:
            try:
                r = PdfReader(io.BytesIO(p_data["pdf_bytes"]))
                for p in r.pages:
                    writer.add_page(p)
            except Exception:
                continue
        
        emitente_safe = limpar_para_nome_arquivo(emitente)
        nome_pdf = f"DOC {numero}_{emitente_safe}.pdf"
        caminho = session_folder / nome_pdf
        
        with open(caminho, "wb") as f_out:
            writer.write(f_out)
            
        resultados.append({
            "file": nome_pdf,
            "numero": numero,
            "emitente": emitente,
            "pages": len(pages_list)
        })
        files_meta[nome_pdf] = {"numero": numero, "emitente": emitente, "pages": len(pages_list)}

    st.session_state["resultados"] = resultados
    st.session_state["session_folder"] = str(session_folder)
    st.session_state["novos_nomes"] = {r["file"]: r["file"] for r in resultados}
    st.session_state["processed_logs"] = processed_logs
    st.session_state["files_meta"] = files_meta

    st.success(f"✅ Processamento concluído em {round(time.time() - start_all, 2)}s — {len(resultados)} arquivos gerados.")
    
    criar_dashboard_analitico()
    
    st.rerun()

# =====================================================================
# PAINEL CORPORATIVO - COM AGRUPAMENTO E VISUALIZAÇÃO
# =====================================================================
if "resultados" in st.session_state:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Gerenciamento — selecione e aplique ações")
    resultados = st.session_state["resultados"]
    session_folder = Path(st.session_state["session_folder"])
    novos_nomes = st.session_state.get("novos_nomes", {r["file"]: r["file"] for r in resultados})
    files_meta = st.session_state.get("files_meta", {})

    # Ajustei as colunas para caber o novo botão
    col1, col2, col3, col4 = st.columns([3, 2, 2, 3]) 
    with col1:
        q = st.text_input("🔎 Buscar arquivo ou emitente", value="", placeholder="parte do nome, emitente ou número")
    with col2:
        sort_by = st.selectbox("Ordenar por", ["Nome (A-Z)", "Nome (Z-A)", "Número (asc)", "Número (desc)"], index=0)
    with col3:
        show_logs = st.checkbox("Mostrar logs detalhados", value=False)
    with col4:
        st.write("") # Espaçamento
        top_actions_cols = st.columns([1, 1, 1])
        
        # Botão Baixar
        with top_actions_cols[0]:
            if st.button("⬇️ Zip"):
                sel = st.session_state.get("selected_files", [])
                if not sel:
                    st.warning("Selecione itens.")
                else:
                    mem = io.BytesIO()
                    with zipfile.ZipFile(mem, "w") as zf:
                        for f in sel:
                            src = session_folder / f
                            if src.exists():
                                arcname = novos_nomes.get(f, f)
                                zf.write(src, arcname=arcname)
                    mem.seek(0)
                    st.download_button("💾 Salvar", data=mem, file_name="selecionadas.zip", mime="application/zip")

        # Botão Excluir
        with top_actions_cols[1]:
            if st.button("🗑️ Del"):
                sel = st.session_state.get("selected_files", [])
                if not sel:
                    st.warning("Selecione itens.")
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
                    st.success("Excluídos!")
                    st.rerun()

        # ### NOVO: Botão Agrupar
        with top_actions_cols[2]:
            if st.button("🔗 Unir"):
                sel = st.session_state.get("selected_files", [])
                if len(sel) < 2:
                    st.warning("Selecione + de 1")
                else:
                    try:
                        merger = PdfWriter()
                        # Ordena a seleção para garantir ordem lógica
                        sel_sorted = sorted(sel)
                        
                        for fname in sel_sorted:
                            src = session_folder / fname
                            if src.exists():
                                reader = PdfReader(str(src))
                                for page in reader.pages:
                                    merger.add_page(page)
                        
                        new_name = f"AGRUPADO_{int(time.time())}.pdf"
                        out_path = session_folder / new_name
                        with open(out_path, "wb") as f:
                            merger.write(f)
                        
                        # Adiciona ao estado
                        new_meta = {
                            "file": new_name, "numero": "AGRUP", "emitente": "VÁRIOS", "pages": len(merger.pages)
                        }
                        st.session_state["resultados"].insert(0, new_meta) # Insere no topo
                        st.session_state["files_meta"][new_name] = new_meta
                        st.session_state["novos_nomes"][new_name] = new_name
                        st.success("Agrupado!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erro: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

    # Filtragem e Ordenação
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
        cols = st.columns([0.05, 0.50, 0.25, 0.20])
        
        checked = fname in st.session_state.get("selected_files", [])
        cb = cols[0].checkbox("", value=checked, key=f"cb_{fname}")
        
        if cb and fname not in st.session_state["selected_files"]:
            st.session_state["selected_files"].append(fname)
        if (not cb) and fname in st.session_state["selected_files"]:
            st.session_state["selected_files"].remove(fname)

        novos_nomes[fname] = cols[1].text_input(label=fname, value=novos_nomes.get(fname, fname), key=f"rename_input_{fname}", label_visibility="collapsed")

        emit = meta.get("emitente", r.get("emitente", "-"))
        num = meta.get("numero", r.get("numero", "-"))
        cols[2].markdown(f"<div class='small-note'>{emit}<br>Nº {num} • {r.get('pages',1)} pág(s)</div>", unsafe_allow_html=True)

        action_col = cols[3]
        if action_col.button("⚙️ Gerenciar", key=f"manage_{fname}"):
            st.session_state["_manage_target"] = fname
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================================
    # PAINEL DE GERENCIAMENTO (COM VISUALIZAÇÃO)
    # =====================================================================
    if "_manage_target" in st.session_state:
        manage_target = st.session_state["_manage_target"]
        
        if not any(r["file"] == manage_target for r in st.session_state.get("resultados", [])):
            st.session_state.pop("_manage_target", None)
            st.rerun()
        
        st.markdown('<div class="manage-panel">', unsafe_allow_html=True)
        col_tit, col_x = st.columns([0.9, 0.1])
        col_tit.markdown(f"### ⚙️ Gerenciar: `{manage_target}`")
        if col_x.button("❌", key=f"close_main_{manage_target}"):
            st.session_state.pop("_manage_target", None)
            st.rerun()
        
        file_path = session_folder / manage_target
        
        # ### VISUALIZADOR PROFISSIONAL (VIA BIBLIOTECA) ###
        with st.expander("👁️ Visualizar Arquivo Completo", expanded=True):
            if file_path.exists():
                try:
                    # width="100%" ajusta a largura à coluna
                    # height=800 define a altura da janela de rolagem
                    pdf_viewer(input=str(file_path), width=700, height=800)
                except Exception as e:
                    st.error(f"Erro ao renderizar PDF: {e}")
            else:
                st.warning("Arquivo não encontrado no disco.")
        # ##################################################

        # (Código original de separar páginas continua aqui...)
        try:
            reader = PdfReader(str(file_path))
            total_pages = len(reader.pages)
            pages_info = [{"idx": i, "label": f"Página {i+1}"} for i in range(total_pages)]
        except Exception as e:
            st.error(f"Erro ao ler o arquivo: {str(e)}")
            pages_info = []
            total_pages = 0
        
        if pages_info:
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
                st.markdown("**Ações Avançadas:**")
                
                selected_count = len(st.session_state.get(sel_key, []))
                st.write(f"📑 Selecionadas: **{selected_count}**")
                
                new_name_key = f"_manage_newname_{manage_target}"
                if new_name_key not in st.session_state:
                    base_name = manage_target.rsplit('.pdf', 1)[0]
                    st.session_state[new_name_key] = f"{base_name}_parte.pdf"
                
                new_name = st.text_input("Nome do novo PDF:", key=new_name_key)
                
                col_sep, col_rem = st.columns(2)
                
                with col_sep:
                    if st.button("➗ Separar páginas", key=f"sep_{manage_target}"):
                        selected = sorted(st.session_state.get(sel_key, []))
                        if not selected:
                            st.warning("Selecione páginas.")
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
                                st.session_state["files_meta"][new_name] = new_meta
                                st.session_state["novos_nomes"][new_name] = new_name
                                st.success(f"Criado: `{new_name}`")
                                st.session_state[sel_key] = [] 
                            except Exception as e:
                                st.error(f"Erro: {str(e)}")
                
                with col_rem:
                    if st.button("🗑️ Remover páginas", key=f"rem_{manage_target}"):
                        selected = sorted(st.session_state.get(sel_key, []))
                        if not selected:
                            st.warning("Selecione páginas.")
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
                                    st.success(f"Páginas removidas.")
                                else:
                                    file_path.unlink()
                                    st.session_state["resultados"] = [r for r in st.session_state["resultados"] if r["file"] != manage_target]
                                    st.session_state["files_meta"].pop(manage_target, None)
                                    st.session_state["novos_nomes"].pop(manage_target, None)
                                    st.session_state.pop("_manage_target", None)
                                    st.rerun()
                                st.session_state[sel_key] = []
                                st.rerun()
                            except Exception as e:
                                st.error(f"Erro: {str(e)}")
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Dashboard analítico
    criar_dashboard_analitico()

    # Mostrar logs se solicitado
    if show_logs and st.session_state.get("processed_logs"):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📝 Logs")
        for entry in st.session_state["processed_logs"][-200:]:
            label, t, status, info, provider = (entry + ("", "", ""))[:5]
            if status == "OK":
                st.markdown(f"<div class='success-log'>✅ {label} — {info}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='warning-log'>⚠️ {label} — {info}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.session_state["novos_nomes"] = novos_nomes

    st.markdown("---")
    st.markdown("### 📤 Baixar Arquivos")
    
    # Prepara o buffer do zip na memória
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w") as zf:
        for r in st.session_state.get("resultados", []):
            fname = r["file"]
            src = session_folder / fname
            if src.exists():
                # Pega o nome renomeado pelo usuário, se houver
                nome_final = st.session_state.get("novos_nomes", {}).get(fname, fname)
                
                # Garante que tenha extensão .pdf
                if not nome_final.lower().endswith(".pdf"):
                    nome_final += ".pdf"
                
                zf.write(src, arcname=nome_final)
    mem.seek(0)
    
    st.info("✅ Processamento finalizado! Clique abaixo para baixar tudo.")
    
    # Botão de Download Único
    st.download_button(
        label="⬇️ Baixar Todas as Notas (.zip)", 
        data=mem, 
        file_name="notas_processadas.zip", 
        mime="application/zip", 
        key="btn_zip_final",
        use_container_width=True,
        type="primary"
    )
