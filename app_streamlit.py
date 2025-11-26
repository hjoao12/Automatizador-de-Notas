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
from streamlit_pdf_viewer import pdf_viewer
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
# GESTÃO DE PADRÕES (NOVA LÓGICA DINÂMICA)
# =====================================================================
PATTERNS_FILE = "patterns.json"

def load_patterns():
    """Carrega padrões do arquivo JSON ou cria padrões padrão se não existir"""
    # Seus padrões originais ficam aqui como backup/inicialização
    default_patterns = {
        "COMPANHIA DE AGUA E ESGOTOS DA PARAIBA": "CAGEPA",
        "COMPANHIA DE AGUA E ESGOTOS DA PARAÍBA": "CAGEPA",
        "CIA DE AGUA E ESGOTO DO CEARA": "CAGECE",
        "COMPANHIA DE AGUAS E ESGOTOS DO RN": "CAERN",
        "PETRÓLEO BRASILEIRO S.A": "PETROBRAS",
        "PETROLEO BRASILEIRO S.A": "PETROBRAS",
        "NEOENERGIA": "NEOENERGIA",
        "EQUATORIAL": "EQUATORIAL"
        # ... adicione outros essenciais aqui se quiser garantir que sempre existam no reset
    }

    # Se o arquivo não existe, cria ele com os defaults
    if not os.path.exists(PATTERNS_FILE):
        save_patterns(default_patterns)
        return default_patterns
    
    try:
        with open(PATTERNS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default_patterns

def save_patterns(patterns):
    """Salva os padrões no arquivo JSON"""
    try:
        with open(PATTERNS_FILE, 'w', encoding='utf-8') as f:
            json.dump(patterns, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        st.error(f"Erro ao salvar padrões: {e}")
        return False

# Carrega os padrões para a memória ao iniciar o script
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
        return {
            "status": "ERRO",
            "dados": dados,
            "tempo": tempo,
            "provider": provider,
            "name": name,
            "page_idx": page_idx,
            "error_msg": dados.get("error", "Erro desconhecido")
        }

# =====================================================================
# SIDEBAR CONFIGURAÇÕES
# =====================================================================
with st.sidebar:
    st.markdown("### 🔧 Configurações")
    
    # Configuração de cache
    st.markdown("#### Otimizações")
    use_cache = st.checkbox("Usar Cache", value=True, key="use_cache")
    
    if st.button("🔄 Limpar Cache"):
        document_cache.clear()
        st.success("Cache limpo!")
        st.rerun()
    st.markdown("---")
    st.markdown("### 📝 Gerenciar Padrões")
    
    with st.expander("Adicionar / Remover"):
        st.markdown("O sistema aprende com esses padrões.")
        
        # --- ADICIONAR ---
        with st.form("add_pattern_form"):
            st.write("**Novo Padrão:**")
            new_key = st.text_input("Texto na Nota (Original)", placeholder="Ex: CIA DE ELETRICIDADE")
            new_value = st.text_input("Renomear para", placeholder="Ex: NEOENERGIA")
            
            if st.form_submit_button("💾 Salvar Novo"):
                if new_key and new_value:
                    SUBSTITUICOES_FIXAS[new_key.upper()] = new_value.upper()
                    if save_patterns(SUBSTITUICOES_FIXAS):
                        st.success("Salvo!")
                        time.sleep(0.5)
                        st.rerun()
                else:
                    st.warning("Preencha os dois campos.")

        st.markdown("---")
        
        # --- REMOVER ---
        st.write("**Padrões Ativos:**")
        # Lista ordenada para facilitar
        lista_padroes = sorted(SUBSTITUICOES_FIXAS.keys())
        
        sel_del = st.selectbox("Selecione para ver/excluir", [""] + lista_padroes)
        
        if sel_del:
            st.info(f"Substitui por: **{SUBSTITUICOES_FIXAS[sel_del]}**")
            if st.button("🗑️ Excluir este padrão"):
                del SUBSTITUICOES_FIXAS[sel_del]
                save_patterns(SUBSTITUICOES_FIXAS)
                st.success("Removido!")
                time.sleep(0.5)
                st.rerun()

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

# --- INÍCIO DO BLOCO NOVO (TURBO) ---
    
    # 1. Preparar trabalhos (Jobs)
    jobs = []
    for a in arquivos:
        name = a["name"]
        try:
            reader = PdfReader(io.BytesIO(a["bytes"]))
            for idx, page in enumerate(reader.pages):
                # Extrair bytes da página individualmente para enviar ao worker
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
                    "use_cache": st.session_state.get("use_cache", True)
                })
        except Exception as e:
            processed_logs.append((name, 0, "ERRO_LEITURA", str(e), "System"))

    # 2. Executar em Paralelo
    MAX_WORKERS = 2  # Número de processamentos simultâneos (seguro)
    processed_count = 0
    total_jobs = len(jobs) if jobs else 1
    
    st.info(f"🚀 Iniciando processamento TURBO de {len(jobs)} páginas com {MAX_WORKERS} threads simultâneas...")
    
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
                    progresso_text.markdown(f"<span class='warning-log'>⚠️ {page_label} — ERRO</span>", unsafe_allow_html=True)
                    resultados_meta.append({
                        "arquivo_origem": name, "pagina": idx+1, "status": "ERRO", "provider": result["provider"]
                    })
                else:
                    # Sucesso (OK ou CACHE)
                    dados = result["dados"]
                    tempo = result["tempo"]
                    provider = result["provider"]
                    
                    # Validação e Correção (usando suas funções existentes)
                    dados = validar_e_corrigir_dados(dados)
                    
                    emitente_raw = dados.get("emitente", "") or ""
                    numero_raw = dados.get("numero_nota", "") or ""
                    cidade_raw = dados.get("cidade", "") or ""

                    numero = limpar_numero(numero_raw)
                    nome_map = substituir_nome_emitente(emitente_raw, cidade_raw)
                    emitente = limpar_emitente(nome_map)

                    # Guardar para gerar o PDF final
                    key = (numero, emitente)
                    # Importante: result["pdf_bytes"] contém a página individual
                    agrupados_bytes.setdefault(key, []).append(result["pdf_bytes"])

                    status_lbl = "CACHE" if result["status"] == "CACHE" else "OK"
                    css_class = "success-log" if result["status"] == "OK" else "warning-log"
                    
                    processed_logs.append((page_label, tempo, status_lbl, f"{numero} / {emitente}", provider))
                    resultados_meta.append({
                        "arquivo_origem": name,
                        "pagina": idx+1,
                        "emitente_detectado": emitente_raw,
                        "numero_detectado": numero_raw,
                        "status": status_lbl,
                        "tempo_s": round(tempo, 2),
                        "provider": provider
                    })
                    progresso_text.markdown(f"<span class='{css_class}'>✅ {page_label} — {status_lbl} ({tempo:.2f}s)</span>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Erro crítico no worker: {e}")
            
            progress_bar.progress(min(processed_count/total_jobs, 1.0))

    # --- FIM DO BLOCO NOVO ---
    resultados = []
    files_meta = {}
    for (numero, emitente), pages_bytes in agrupados_bytes.items():
        if not numero or numero == "0":
            continue
        writer = PdfWriter()
        for pb in pages_bytes:
            try:
                r = PdfReader(io.BytesIO(pb))
                for p in r.pages:
                    writer.add_page(p)
            except Exception:
                continue
        nome_pdf = f"DOC {numero}_{emitente}.pdf"
        caminho = session_folder / nome_pdf
        with open(caminho, "wb") as f_out:
            writer.write(f_out)
        resultados.append({
            "file": nome_pdf,
            "numero": numero,
            "emitente": emitente,
            "pages": len(pages_bytes)
        })
        files_meta[nome_pdf] = {"numero": numero, "emitente": emitente, "pages": len(pages_bytes)}

    st.session_state["resultados"] = resultados
    st.session_state["session_folder"] = str(session_folder)
    st.session_state["novos_nomes"] = {r["file"]: r["file"] for r in resultados}
    st.session_state["processed_logs"] = processed_logs
    st.session_state["files_meta"] = files_meta

    st.success(f"✅ Processamento concluído em {round(time.time() - start_all, 2)}s — {len(resultados)} arquivos gerados.")
    
    # Mostrar dashboard após processamento
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
       # --- CÓDIGO NOVO COMEÇA AQUI ---
        # 2. Carregar lógica do arquivo
        try:
            reader = PdfReader(str(file_path))
            n_pages = len(reader.pages)
        except Exception as e:
            st.error(f"Erro ao ler: {e}")
            n_pages = 0
        
        if n_pages > 0:
            st.divider()
            
            # === A: REORDENAÇÃO RÁPIDA (CAMPO DE TEXTO) ===
            st.markdown("#### 🔃 Reordenar Páginas")
            st.caption("Edite os números abaixo para mudar a ordem (ex: mude '1, 2' para '2, 1').")
            
            # Cria a lista padrão: "1, 2, 3..."
            current_order = [str(i+1) for i in range(n_pages)]
            val_padrao = ", ".join(current_order)
            
            c_ord_in, c_ord_btn = st.columns([4, 1])
            # O input onde você digita a nova ordem
            new_order = c_ord_in.text_input("Ordem das páginas:", value=val_padrao, key=f"ord_{manage_target}", label_visibility="collapsed")
            
            if c_ord_btn.button("Aplicar", key=f"apply_{manage_target}"):
                try:
                    # Transforma o texto "1, 3, 2" em números que o Python entende
                    parts = [p.strip() for p in new_order.split(",") if p.strip().isdigit()]
                    indices = [int(x)-1 for x in parts]
                    
                    # Verifica se está tudo certo
                    if not indices:
                        st.warning("Lista vazia.")
                    elif any(x < 0 or x >= n_pages for x in indices):
                        st.error(f"Existem páginas inválidas (o arquivo só tem {n_pages}).")
                    else:
                        # SALVA O NOVO PDF
                        w = PdfWriter()
                        for i in indices: w.add_page(reader.pages[i])
                        with open(file_path, "wb") as f: w.write(f)
                        
                        # Atualiza o sistema
                        st.session_state["files_meta"][manage_target]["pages"] = len(indices)
                        for r in st.session_state["resultados"]:
                            if r["file"] == manage_target: r["pages"] = len(indices)
                        
                        st.toast("Ordem atualizada! 🔄")
                        time.sleep(1)
                        st.rerun()
                except Exception as e:
                    st.error(f"Erro ao salvar: {e}")

            st.divider()

            # === B: FERRAMENTAS EXTRAS (SEPARAR/REMOVER) ===
            with st.expander("✂️ Ferramentas de Seleção (Separar/Excluir)"):
                pages_info = [{"idx": i, "label": f"Página {i+1}"} for i in range(n_pages)]
                sel_key = f"_sel_{manage_target}"
                if sel_key not in st.session_state: st.session_state[sel_key] = []
                
                c_sel, c_act = st.columns([1, 1])
                
                with c_sel:
                    st.caption("Selecione páginas específicas:")
                    for page in pages_info:
                        chk = page["idx"] in st.session_state[sel_key]
                        if st.checkbox(page["label"], value=chk, key=f"pg_{manage_target}_{page['idx']}"):
                            if not chk: st.session_state[sel_key].append(page["idx"])
                        else:
                            if chk: st.session_state[sel_key].remove(page["idx"])

                with c_act:
                    st.caption("Ações com as selecionadas:")
                    new_name_part = st.text_input("Nome do novo arquivo:", value=f"{manage_target.replace('.pdf','')}_parte.pdf", key=f"name_{manage_target}")
                    
                    b1, b2 = st.columns(2)
                    
                    if b1.button("Extrair ✅", key=f"ext_{manage_target}"):
                        sel = sorted(st.session_state[sel_key])
                        if sel:
                            w = PdfWriter()
                            for i in sel: w.add_page(reader.pages[i])
                            out = session_folder / new_name_part
                            with open(out, "wb") as f: w.write(f)
                            
                            # Adiciona na lista principal
                            meta = {"file": new_name_part, "numero": "PART", "emitente": "EXT", "pages": len(sel)}
                            st.session_state["resultados"].insert(0, meta)
                            st.session_state["files_meta"][new_name_part] = meta
                            st.session_state["novos_nomes"][new_name_part] = new_name_part
                            st.success("Extraído!")
                            st.session_state[sel_key] = []
                            st.rerun()
                        else:
                            st.warning("Selecione páginas.")
                    
                    if b2.button("Remover ❌", key=f"rm_{manage_target}"):
                        sel = sorted(st.session_state[sel_key])
                        if sel:
                            keep = [i for i in range(n_pages) if i not in sel]
                            if keep:
                                w = PdfWriter()
                                for i in keep: w.add_page(reader.pages[i])
                                with open(file_path, "wb") as f: w.write(f)
                                st.session_state["files_meta"][manage_target]["pages"] = len(keep)
                                for r in st.session_state["resultados"]:
                                    if r["file"] == manage_target: r["pages"] = len(keep)
                                st.success("Removido!")
                                st.session_state[sel_key] = []
                                st.rerun()
                            else:
                                st.error("Não pode remover todas.")
                        else:
                            st.warning("Selecione páginas.")
        # --- CÓDIGO NOVO TERMINA AQUI ---
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
            st.download_button("⬇️ ZIP Completo", data=mem, file_name="todas_notas.zip", mime="application/zip")

else:
    st.info("Nenhum arquivo processado ainda. Faça upload e clique em 'Processar PDFs'.")
