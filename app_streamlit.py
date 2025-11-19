# turbo_v5_final_fixed.py
# Versão: Turbo v5 — Corrigida e Completa (com Download ZIP e UI de Resultados)

# =====================================================================
# PARTE 1/6 — IMPORTS E CONFIGURAÇÃO INICIAL
# =====================================================================
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
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF
import openai
import streamlit as st
from PyPDF2 import PdfReader, PdfWriter
from dotenv import load_dotenv

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

# =====================================================================
# CONFIGURAÇÃO INICIAL
# =====================================================================
load_dotenv()
st.set_page_config(page_title="Automatizador de Notas Fiscais", page_icon="📄", layout="centered")

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
  width: 100%;
}
div.stButton > button:hover {
  background-color: #0b3a5a;
}
.stProgress > div > div > div > div {
  background-color: #28a745 !important;
}
.success-log { color: #155724; background-color: #d4edda; padding: 4px 8px; border-radius: 4px; font-size: 0.9em; }
.warning-log { color: #856404; background-color: #fff3cd; padding: 4px 8px; border-radius: 4px; font-size: 0.9em; }
.error-log { color: #721c24; background-color: #f8d7da; padding: 4px 8px; border-radius: 4px; font-size: 0.9em; }
.card { background: #fff; padding: 20px; border-radius:10px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom:20px; }
</style>
""", unsafe_allow_html=True)

st.title("Automatizador de Notas Fiscais PDF")

# =====================================================================
# PARTE 2/6 — CACHE INTELIGENTE E VARIÁVEIS GLOBAIS
# =====================================================================
class DocumentCache:
    def __init__(self, cache_dir="./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def get_cache_key(self, pdf_bytes, prompt):
        content_hash = hashlib.md5(pdf_bytes).hexdigest()
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        return f"{content_hash}_{prompt_hash}"

    def _cache_path(self, key):
        safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", key)[:240]
        return self.cache_dir / f"{safe}.pkl"

    def get(self, key, ttl_seconds: int = 3600):
        cache_file = self._cache_path(key)
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    obj = pickle.load(f)
                ts = obj.get("ts", 0)
                if ttl_seconds <= 0 or (time.time() - ts) < ttl_seconds:
                    return obj.get("data")
                else:
                    try:
                        cache_file.unlink()
                    except:
                        pass
                    return None
            except Exception:
                return None
        return None

    def set(self, key, data, ttl_seconds: int = 86400):
        cache_file = self._cache_path(key)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({"data": data, "ts": time.time(), "ttl": ttl_seconds}, f)
        except Exception:
            pass

    def clear(self):
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except:
                pass

document_cache = DocumentCache()

# =====================================================================
# PARTE 3/6 — CONFIG, PADRÕES E NORMALIZAÇÃO
# =====================================================================
TEMP_FOLDER = Path("./temp")
TEMP_FOLDER.mkdir(exist_ok=True)

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
MIN_RETRY_DELAY = int(os.getenv("MIN_RETRY_DELAY", "2"))
MAX_RETRY_DELAY = int(os.getenv("MAX_RETRY_DELAY", "10"))
PATTERNS_FILE = "patterns.json"

def load_patterns():
    default_patterns = {
        "COMPANHIA DE AGUA E ESGOTOS DA PARAIBA": "CAGEPA",
        "COMPANHIA DE AGUA E ESGOTOS DA PARAÍBA": "CAGEPA",
        "CIA DE AGUA E ESGOTO DO CEARA": "CAGECE",
        "COMPANHIA DE AGUAS E ESGOTOS DO RN": "CAERN",
        "PETRÓLEO BRASILEIRO S.A": "PETROBRAS",
        "PETROLEO BRASILEIRO S.A": "PETROBRAS",
        "NEOENERGIA": "NEOENERGIA",
        "EQUATORIAL": "EQUATORIAL"
    }
    if not os.path.exists(PATTERNS_FILE):
        save_patterns(default_patterns)
        return default_patterns
    try:
        with open(PATTERNS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default_patterns

def save_patterns(patterns):
    try:
        with open(PATTERNS_FILE, 'w', encoding='utf-8') as f:
            json.dump(patterns, f, ensure_ascii=False, indent=4)
        return True
    except Exception:
        return False

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
    if not isinstance(dados, dict):
        return {"emitente": "NÃO_IDENTIFICADO", "numero_nota": "000000", "cidade": "NÃO_IDENTIFICADO"}
    required_fields = ['emitente', 'numero_nota', 'cidade']
    for field in required_fields:
        if field not in dados or not dados[field]:
            dados[field] = "NÃO_IDENTIFICADO"
    if 'numero_nota' in dados:
        numero_limpo = re.sub(r"[^\d]", "", str(dados['numero_nota']))
        dados['numero_nota'] = numero_limpo if numero_limpo else "000000"
    return dados

# =====================================================================
# PARTE 4/6 — GEMINI / OPENAI CLIENTS E FUNÇÕES DE PROCESSAMENTO
# =====================================================================
# Gemini
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ Chave GOOGLE_API_KEY não encontrada no .env.")
    st.stop()
try:
    genai.configure(api_key=GEMINI_API_KEY)
    # Ajuste aqui para um modelo padrão estável se o usuário não definir a variável
    model = genai.GenerativeModel(os.getenv("MODEL_NAME", "gemini-1.5-flash"))
except Exception as e:
    st.error(f"❌ Erro ao configurar Gemini: {str(e)}")
    st.stop()

# OpenAI client (opcional)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = None
if OPENAI_API_KEY:
    try:
        openai.api_key = OPENAI_API_KEY
        openai_client = openai
    except Exception:
        openai_client = None

def extrair_json_seguro(texto: str):
    texto = (texto or "").strip()
    # Tenta encontrar array JSON
    match_array = re.search(r"\[.*\]", texto, re.DOTALL)
    if match_array:
        try:
            return json.loads(match_array.group())
        except:
            pass
    # Tenta encontrar objeto JSON
    match_obj = re.search(r"\{.*\}", texto, re.DOTALL)
    if match_obj:
        try:
            return json.loads(match_obj.group())
        except:
            pass
    # Limpeza básica markdown
    clean = texto.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean)
    except:
        return {} # Retorna dict vazio em vez de lista vazia para evitar erro de tipo

def calcular_delay(tentativa, error_msg):
    try:
        em = (error_msg or "").lower()
        if "retry in" in em:
            seg = float(re.search(r"retry in (\d+\.?\d*)s", em).group(1))
            return min(seg + 2, MAX_RETRY_DELAY)
    except:
        pass
    return min(MIN_RETRY_DELAY * (tentativa + 1), MAX_RETRY_DELAY)

def processar_pagina_gemini(prompt_instrucao, page_stream):
    for tentativa in range(MAX_RETRIES + 1):
        try:
            start = time.time()
            resp = model.generate_content(
                [prompt_instrucao, {"mime_type": "application/pdf", "data": page_stream.getvalue()}],
                generation_config={"response_mime_type": "application/json"},
                request_options={'timeout': 60}
            )
            tempo = round(time.time() - start, 2)
            texto = (resp.text or "").strip()
            
            # Usa a função segura para extrair
            dados = extrair_json_seguro(texto)
            
            if not dados:
                dados = {"error": "JSON inválido ou vazio", "_raw": texto[:200]}
                
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

def processar_pagina_openai_fallback(prompt_sistema, pdf_bytes):
    if not openai_client:
        return {"error": "OpenAI não configurada"}, False, 0, "OpenAI"
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if len(doc) < 1:
            return {"error": "PDF vazio"}, False, 0, "OpenAI"
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_bytes = pix.tobytes("png")
        base64_image = base64.b64encode(img_bytes).decode('utf-8')
        
        messages = [
            {"role": "system", "content": "Extraia emitente, numero_nota e cidade. Retorne JSON."},
            {"role": "user", "content": prompt_sistema + "\nIMAGEM: data:image/png;base64," + base64_image}
        ]
        resp = openai_client.ChatCompletion.create(model="gpt-4o", messages=messages, temperature=0)
        conteudo = resp.choices[0].message['content']
        dados = extrair_json_seguro(conteudo)
        return dados, True, 0, "GPT-Vision"
    except Exception as e:
        return {"error": f"Erro Vision: {str(e)}"}, False, 0, "GPT-Vision"

# =====================================================================
# PARTE 5/6 — WORKER E LOOP PRINCIPAL
# =====================================================================

def processar_pagina_worker(job_data):
    pdf_bytes = job_data["bytes"]
    prompt = job_data["prompt"]
    name = job_data["name"]
    page_idx = job_data["page_idx"]
    mode = job_data.get("mode", "GEMINI")
    use_cache = job_data.get("use_cache", True)

    cache_key = document_cache.get_cache_key(pdf_bytes, prompt)
    
    if use_cache:
        cached_result = document_cache.get(cache_key)
        if cached_result:
            return {
                "status": "CACHE", 
                "dados": cached_result.get('dados'), 
                "tempo": cached_result.get('tempo', 0), 
                "provider": cached_result.get('provider','CACHE'), 
                "name": name, "page_idx": page_idx, "pdf_bytes": pdf_bytes
            }

    if mode == "OPENAI":
        dados, ok, tempo, provider = processar_pagina_openai_fallback(prompt, pdf_bytes)
    else:
        page_stream = io.BytesIO(pdf_bytes)
        dados, ok, tempo, provider = processar_pagina_gemini(prompt, page_stream)
        
        # Fallback para OpenAI se Gemini falhar e OpenAI estiver configurada
        if (not ok or 'error' in dados) and openai_client:
            dados2, ok2, tempo2, prov2 = processar_pagina_openai_fallback(prompt, pdf_bytes)
            if ok2 and 'error' not in dados2:
                dados, ok, tempo, provider = dados2, ok2, tempo2, prov2

    if ok and isinstance(dados, dict) and 'error' not in dados:
        document_cache.set(cache_key, {'dados': dados, 'tempo': tempo, 'provider': provider})
        return {"status": "OK", "dados": dados, "tempo": tempo, "provider": provider, "name": name, "page_idx": page_idx, "pdf_bytes": pdf_bytes}
    else:
        err = dados.get('error') if isinstance(dados, dict) else str(dados)
        return {"status": "ERRO", "dados": dados, "tempo": tempo, "provider": provider, "name": name, "page_idx": page_idx, "error_msg": err, "pdf_bytes": pdf_bytes}

# =====================================================================
# PARTE 6/6 — UI, SIDEBAR E UPLOAD
# =====================================================================
with st.sidebar:
    st.header("⚙️ Configuração")
    provider_label = st.radio("Modelo Principal", ["Gemini (Google)", "GPT-4o (OpenAI)"], index=0)
    mode_sel = "GEMINI" if "Gemini" in provider_label else "OPENAI"
    st.markdown("---")
    use_cache = st.checkbox("Usar Cache", value=True, key="use_cache")
    workers = st.slider("Processamento Paralelo (Workers)", 1, 8, 4)
    if st.button("🔄 Limpar Cache"):
        document_cache.clear()
        st.success("Cache limpo!")

# Lógica para limpar sessão
if st.session_state.get("_limpar_sessao"):
    if "session_folder" in st.session_state and st.session_state["session_folder"]:
        try:
            shutil.rmtree(st.session_state["session_folder"])
        except:
            pass
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.session_state["_limpar_sessao"] = False
    st.rerun()

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 📎 Upload de Arquivos")
uploaded_files = st.file_uploader("Selecione arquivos PDF", type=["pdf"], accept_multiple_files=True, key="uploader")
col1, col2 = st.columns([1,1])
with col1:
    process_btn = st.button("🚀 Processar Agora")
with col2:
    if st.button("♻️ Nova Sessão"):
        st.session_state["_limpar_sessao"] = True
        st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

# =====================================================================
# LÓGICA DE PROCESSAMENTO
# =====================================================================
if uploaded_files and process_btn:
    session_id = str(uuid.uuid4())
    session_folder = TEMP_FOLDER / session_id
    os.makedirs(session_folder, exist_ok=True)

    arquivos = []
    for f in uploaded_files:
        try:
            arquivos.append({"name": f.name, "bytes": f.read()})
        except:
            st.warning(f"Ignorado (erro leitura): {f.name}")

    if not arquivos:
        st.error("Nenhum arquivo válido.")
        st.stop()

    # Prepara Jobs
    jobs = []
    for a in arquivos:
        try:
            reader = PdfReader(io.BytesIO(a["bytes"]))
            for idx, page in enumerate(reader.pages):
                b_io = io.BytesIO()
                writer = PdfWriter()
                writer.add_page(page)
                writer.write(b_io)
                jobs.append({
                    "bytes": b_io.getvalue(),
                    "prompt": 'Analise a nota fiscal. Retorne SOMENTE JSON: {"emitente":"NOME","numero_nota":"123","cidade":"CIDADE"}',
                    "name": a["name"],
                    "page_idx": idx,
                    "use_cache": use_cache,
                    "mode": mode_sel
                })
        except Exception as e:
            st.warning(f"Erro no arquivo {a['name']}: {e}")

    if not jobs:
        st.error("Nenhuma página encontrada.")
        st.stop()

    # Executa Paralelo
    agrupados_bytes = {}
    processed_logs = []
    processed_count = 0
    total = len(jobs)
    
    bar = st.progress(0)
    status_txt = st.empty()
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_job = {executor.submit(processar_pagina_worker, j): j for j in jobs}
        
        for future in as_completed(future_to_job):
            processed_count += 1
            try:
                res = future.result()
                label = f"{res['name']} (Pág {res['page_idx']+1})"
                
                if res["status"] == "ERRO":
                    status_txt.markdown(f"<span class='error-log'>❌ {label}: {res.get('error_msg')}</span>", unsafe_allow_html=True)
                else:
                    dados = validar_e_corrigir_dados(res["dados"])
                    numero = limpar_numero(dados.get("numero_nota"))
                    cidade = dados.get("cidade")
                    emitente = limpar_emitente(substituir_nome_emitente(dados.get("emitente"), cidade))
                    
                    key = (numero, emitente)
                    agrupados_bytes.setdefault(key, []).append(res["pdf_bytes"])
                    
                    status_txt.markdown(f"<span class='success-log'>✅ {label} > {numero} - {emitente}</span>", unsafe_allow_html=True)
                    
                processed_logs.append(res)
            except Exception as e:
                print(f"Erro thread: {e}")
            
            bar.progress(min(processed_count / total, 1.0))

    # Gera PDFs Finais
    resultados = []
    for (num, emit), pages_list in agrupados_bytes.items():
        if not num or num == "0": continue
        
        merger = PdfWriter()
        p_count = 0
        for p_bytes in pages_list:
            try:
                r = PdfReader(io.BytesIO(p_bytes))
                for p in r.pages:
                    merger.add_page(p)
                    p_count += 1
            except: pass
            
        if p_count > 0:
            fname = f"DOC_{num}_{emit}.pdf"
            fpath = session_folder / fname
            with open(fpath, "wb") as fout:
                merger.write(fout)
            resultados.append({"file": fname, "numero": num, "emitente": emit, "paginas": p_count})

    # Salva na Sessão e Recarrega
    st.session_state["resultados"] = resultados
    st.session_state["session_folder"] = str(session_folder)
    st.session_state["logs"] = processed_logs
    st.success(f"Concluído! {len(resultados)} documentos gerados.")
    st.rerun()

# =====================================================================
# PARTE 7/6 — EXIBIÇÃO DE RESULTADOS (TELA PÓS-PROCESSAMENTO)
# =====================================================================
if "resultados" in st.session_state and st.session_state["resultados"]:
    st.markdown("---")
    st.header("📂 Arquivos Gerados")
    
    res_data = st.session_state["resultados"]
    
    # Cria ZIP na memória
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, "w") as zf:
        base_path = Path(st.session_state["session_folder"])
        for r in res_data:
            p = base_path / r["file"]
            if p.exists():
                zf.write(p, arcname=r["file"])
    
    col_d1, col_d2 = st.columns([2, 1])
    with col_d1:
        st.success(f"Processamento finalizado com sucesso! {len(res_data)} arquivos prontos.")
    with col_d2:
        st.download_button(
            label="📦 Baixar Todos (ZIP)",
            data=mem_zip.getvalue(),
            file_name="notas_organizadas.zip",
            mime="application/zip"
        )

    st.dataframe(res_data, use_container_width=True)

# FIM