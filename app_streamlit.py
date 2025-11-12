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
import openai
from openai import OpenAI
import requests
import base64

# =====================================================================
# CONFIGURAÇÃO INICIAL
# =====================================================================
load_dotenv()
st.set_page_config(page_title="Automatizador de Notas Fiscais", page_icon="🧾", layout="wide")

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
.provider-gemini { color: #4285F4; }
.provider-openai { color: #19C37D; }
.provider-deepseek { color: #0D9276; }
.provider-claude { color: #FF6B35; }
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
</style>
""", unsafe_allow_html=True)

st.title("🧠 Automatizador de Notas Fiscais PDF")

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
# MULTI-IA COM FALLBACK (REFATORADO)
# =====================================================================
class MultiAIProvider:
    def __init__(self):
        self.providers = self._setup_providers()
        self.active_provider = None
        self.stats = {p['name']: {'success': 0, 'errors': 0, 'total_time': 0} for p in self.providers}

    # --------------------------
    # Configuração dos provedores
    # --------------------------
    def _setup_providers(self):
        providers = []
        if os.getenv("GOOGLE_API_KEY"):
            try:
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                model = genai.GenerativeModel(os.getenv("MODEL_NAME", "models/gemini-2.5-flash"))
                providers.append({'name': 'Gemini', 'model': model, 'type': 'gemini', 'priority': 1, 'enabled': True})
                st.sidebar.success("✅ Gemini configurado")
            except Exception:
                st.sidebar.warning("⚠️ Gemini não configurado")

        if os.getenv("OPENAI_API_KEY"):
            try:
                openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                providers.append({'name': 'OpenAI', 'client': openai_client, 'type': 'openai', 'model': os.getenv("OPENAI_MODEL", "gpt-4o"), 'priority': 2, 'enabled': True})
                st.sidebar.success("✅ OpenAI configurado")
            except Exception:
                st.sidebar.warning("⚠️ OpenAI não configurado")

        if os.getenv("DEEPSEEK_API_KEY"):
            try:
                providers.append({'name': 'DeepSeek', 'api_key': os.getenv("DEEPSEEK_API_KEY"), 'type': 'deepseek', 'model': os.getenv("DEEPSEEK_MODEL", "deepseek-chat"), 'priority': 3, 'enabled': True})
                st.sidebar.success("✅ DeepSeek configurado")
            except Exception:
                st.sidebar.warning("⚠️ DeepSeek não configurado")

        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                providers.append({'name': 'Claude', 'api_key': os.getenv("ANTHROPIC_API_KEY"), 'type': 'claude', 'model': os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"), 'priority': 4, 'enabled': True})
                st.sidebar.success("✅ Claude configurado")
            except Exception:
                st.sidebar.warning("⚠️ Claude não configurado")

        if not providers:
            st.error("❌ Nenhum provedor de IA configurado.")
            st.stop()

        return sorted(providers, key=lambda x: x['priority'])

    # --------------------------
    # Função auxiliar: extrair texto de PDF
    # --------------------------
    def pdf_para_texto(self, page_stream):
        """Extrai texto de um PDF (uma página) usando PyPDF2"""
        reader = PdfReader(io.BytesIO(page_stream.getvalue()))
        texto = ""
        for p in reader.pages:
            t = p.extract_text()
            if t:
                texto += t + "\n"
        return texto.strip()

    # --------------------------
    # Processamento com fallback
    # --------------------------
    def process_pdf_page(self, prompt_instrucao, page_stream, max_retries=2):
        cache_key = document_cache.get_cache_key(page_stream.getvalue(), prompt_instrucao)
        cached_result = document_cache.get(cache_key)
        if cached_result and st.session_state.get("use_cache", True):
            st.sidebar.info("💾 Usando cache")
            return cached_result['dados'], True, cached_result['tempo'], cached_result['provider']

        last_error = None
        error_count = 0

        for provider in self.providers:
    if not provider.get('enabled', True):
        continue

    inicio = time.time()
    st.sidebar.info(f"🔄 Tentando {provider['name']}...")

    try:
        if provider['type'] == 'gemini':
            dados, tempo = self._call_gemini(provider, prompt_instrucao, page_stream)
        elif provider['type'] == 'openai':
            dados, tempo = self._call_openai(provider, prompt_instrucao, page_stream)
        elif provider['type'] == 'deepseek':
            dados, tempo = self._call_deepseek(provider, prompt_instrucao, page_stream)
        elif provider['type'] == 'claude':
            dados, tempo = self._call_claude(provider, prompt_instrucao, page_stream)
        else:
            continue  # ignora provider inválido

        self.stats[provider['name']]['success'] += 1
        self.stats[provider['name']]['total_time'] += tempo
        self.active_provider = provider['name']

        document_cache.set(cache_key, {'dados': dados, 'tempo': tempo, 'provider': provider['name']})
        return dados, True, tempo, provider['name']

    except Exception as e:
        tempo_falha = round(time.time() - inicio, 2)
        self.stats[provider['name']]['errors'] += 1
        last_error = f"{provider['name']} falhou em {tempo_falha:.2f}s: {e}"
        st.sidebar.warning(f"⚠️ {provider['name']} falhou ({tempo_falha:.2f}s). Tentando próximo...")
        continue  # ← fallback rápido e automático (sem travar nem reprocessar)

# Se nenhum provedor funcionou:
return {"error": f"Todos os provedores falharam. Último erro: {last_error}"}, False, 0, "Nenhum"

                error_count += 1
                last_error = f"{provider['name']} quota error: {str(e)}"
                self.stats[provider['name']]['errors'] += 1
                time.sleep(1)
                continue
            except Exception as e:
                error_count += 1
                last_error = f"{provider['name']} error: {str(e)}"
                self.stats[provider['name']]['errors'] += 1
                if error_count < 2:
                    st.sidebar.warning(f"⚠️ {provider['name']} falhou, tentativa 1, reprocessando...")
                    time.sleep(2)
                    continue
                else:
                    st.sidebar.warning(f"⚠️ {provider['name']} falhou, passando para próximo provedor...")
                    continue

        return {"error": f"Todos os provedores falharam. Último erro: {last_error}"}, False, 0, "Nenhum"

    # --------------------------
    # Chamadas individuais
    # --------------------------
    def _call_gemini(self, provider, prompt, page_stream):
        start = time.time()
        resp = provider['model'].generate_content(
            [prompt, {"mime_type": "application/pdf", "data": page_stream.getvalue()}],
            generation_config={"response_mime_type": "application/json"},
            request_options={'timeout': 60}
        )
        tempo = round(time.time() - start, 2)
        texto = resp.text.strip().lstrip("```json").rstrip("```").strip()
        dados = json.loads(texto)
        return dados, tempo

    def _call_openai(self, provider, prompt, page_stream):
        start = time.time()
        texto_pdf = self.pdf_para_texto(page_stream)
        content = f"{prompt}\n\n{texto_pdf}"

        response = provider['client'].chat.completions.create(
            model=provider['model'],
            messages=[{"role": "user", "content": content}],
            timeout=60
        )
        
        tempo = round(time.time() - start, 2)
        texto = response.choices[0].message.content
        dados = json.loads(texto)
        return dados, tempo

    def _call_deepseek(self, provider, prompt, page_stream):
        start = time.time()
        texto_pdf = self.pdf_para_texto(page_stream)
        content = f"{prompt}\n\n{texto_pdf}"

        client = OpenAI(
            api_key=provider['api_key'],
            base_url="https://api.deepseek.com/v1"
        )

        response = client.chat.completions.create(
            model=provider['model'],
            messages=[{"role": "user", "content": content}],
            timeout=60
        )

        tempo = round(time.time() - start, 2)
        texto = response.choices[0].message.content
        dados = json.loads(texto)
        return dados, tempo

    def _call_claude(self, provider, prompt, page_stream):
        start = time.time()
        texto_pdf = self.pdf_para_texto(page_stream)
        content = f"{prompt}\n\n{texto_pdf}"

        headers = {
            "x-api-key": provider['api_key'],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        data = {
            "model": provider['model'],
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": content}]
        }

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data,
            timeout=60
        )

        if response.status_code != 200:
            raise Exception(f"Claude API error: {response.text}")

        tempo = round(time.time() - start, 2)
        result = response.json()
        texto = result['content'][0]['text']
        dados = json.loads(texto)
        return dados, tempo

    # --------------------------
    # Estatísticas
    # --------------------------
    def get_stats(self):
        return self.stats


# Inicializar o multi-IA
multi_ai = MultiAIProvider()

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

# =====================================================================
# NORMALIZAÇÃO E SUBSTITUIÇÕES
# =====================================================================
SUBSTITUICOES_FIXAS = {
    "COMPANHIA DE AGUA E ESGOTOS DA PARAIBA": "CAGEPA",
    "COMPANHIA DE AGUA E ESGOTOS DA PARAÍBA": "CAGEPA",
    "COMPANHIA DE AGUA E ESGOTO DA PARAIBA": "CAGEPA",
    "CIA DE AGUA E ESGOTO DO CEARA": "CAGECE",
    "COMPANHIA DE AGUA E ESGOTO DO CEARA": "CAGECE",
    "CAGECE": "CAGECE",
    "TRANSPORTE LIDA": "TRANSPORTE_LIDA",
    "TRANSPORTE LIDA LTDA": "TRANSPORTE_LIDA",
    "TRANSPORTELIDA": "TRANSPORTE_LIDA",
    "UNIPAR CARBOCLORO": "UNIPAR_CARBLOCLORO",
    "UNIPAR CARBOCLORO LTDA": "UNIPAR_CARBLOCLORO",
    "UNIPAR_CARBLOCLORO LTDA": "UNIPAR_CARBLOCLORO",
    "EXPRESS TCM": "EXPRESS_TCM",
    "EXPRESS TCM LTDA": "EXPRESS_TCM",
}

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
# SIDEBAR CONFIGURAÇÕES
# =====================================================================
with st.sidebar:
    st.markdown("### 🔧 Configurações Avançadas")
    
    # Configuração de provedores
    st.markdown("#### Provedores de IA")
    for provider in multi_ai.providers:
        enabled = st.checkbox(
            f"{provider['name']}", 
            value=provider.get('enabled', True),
            key=f"provider_{provider['name']}"
        )
        provider['enabled'] = enabled
    
    # Configuração de cache
    st.markdown("#### Otimizações")
    use_cache = st.checkbox("Usar Cache", value=True, key="use_cache")
    
    if st.button("🔄 Limpar Cache"):
        document_cache.clear()
        st.success("Cache limpo!")
    
    # Estatísticas dos provedores
    st.markdown("#### 📊 Estatísticas dos Provedores")
    stats = multi_ai.get_stats()
    for provider_name, stat in stats.items():
        total = stat['success'] + stat['errors']
        if total > 0:
            success_rate = (stat['success'] / total) * 100
            avg_time = stat['total_time'] / stat['success'] if stat['success'] > 0 else 0
            st.write(f"**{provider_name}**:")
            st.write(f"✅ {stat['success']} | ❌ {stat['errors']}")
            st.write(f"📊 {success_rate:.1f}% | ⏱️ {avg_time:.1f}s")
            st.write("---")

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
    
    # Estatísticas por provedor
    if logs:
        st.markdown("#### 🔄 Uso dos Provedores")
        provider_stats = {}
        for log in logs:
            if len(log) > 4 and log[4]:  # provider info
                provider = log[4]
                provider_stats[provider] = provider_stats.get(provider, 0) + 1
        
        for provider, count in provider_stats.items():
            provider_class = f"provider-{provider.lower()}"
            st.markdown(f"<span class='{provider_class}'>**{provider}**: {count} páginas</span>", unsafe_allow_html=True)
    
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
st.markdown("### 📎 Enviar PDFs e processar (uma vez)")
uploaded_files = st.file_uploader("Selecione arquivos PDF", type=["pdf"], accept_multiple_files=True, key="uploader")
col_up_a, col_up_b = st.columns([1,1])
with col_up_a:
    process_btn = st.button("🚀 Processar PDFs")
with col_up_b:
    clear_session = st.button("♻️ Limpar sessão (apagar temporários)")

st.markdown("</div>", unsafe_allow_html=True)

if clear_session:
    if "session_folder" in st.session_state:
        try:
            shutil.rmtree(st.session_state["session_folder"])
        except Exception:
            pass
    for k in ["resultados", "session_folder", "novos_nomes", "processed_logs", "files_meta", "selected_files"]:
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
    st.info(f"🔧 Provedores ativos: {[p['name'] for p in multi_ai.providers if p.get('enabled', True)]}")

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

    for a in arquivos:
        name = a["name"]
        try:
            reader = PdfReader(io.BytesIO(a["bytes"]))
        except Exception:
            processed_logs.append((name, 0, "ERRO_LEITURA", "", "Nenhum"))
            continue

        for idx, page in enumerate(reader.pages):
            b = io.BytesIO()
            w = PdfWriter()
            w.add_page(page)
            w.write(b)
            b.seek(0)

            # CHAMADA MULTI-IA COM FALLBACK
            dados, ok, tempo, provider = multi_ai.process_pdf_page(prompt, b)
            
            page_label = f"{name} (pág {idx+1})"
            if not ok or "error" in dados:
                processed_logs.append((page_label, tempo, "ERRO_IA", dados.get("error", str(dados)), provider))
                progresso += 1
                progress_bar.progress(min(progresso/total_paginas, 1.0))
                progresso_text.markdown(f"<span class='warning-log'>⚠️ {page_label} — ERRO IA [{provider}]</span>", unsafe_allow_html=True)
                resultados_meta.append({
                    "arquivo_origem": name,
                    "pagina": idx+1,
                    "emitente_detectado": dados.get("emitente") if isinstance(dados, dict) else "-",
                    "numero_detectado": dados.get("numero_nota") if isinstance(dados, dict) else "-",
                    "status": "ERRO",
                    "provider": provider
                })
                continue

            # Validar e corrigir dados
            dados = validar_e_corrigir_dados(dados)

            emitente_raw = dados.get("emitente", "") or ""
            numero_raw = dados.get("numero_nota", "") or ""
            cidade_raw = dados.get("cidade", "") or ""

            numero = limpar_numero(numero_raw)
            nome_map = substituir_nome_emitente(emitente_raw, cidade_raw)
            emitente = limpar_emitente(nome_map)

            key = (numero, emitente)
            agrupados_bytes.setdefault(key, []).append(b.getvalue())

            provider_class = f"provider-{provider.lower()}" if provider else ""
            processed_logs.append((page_label, tempo, "OK", f"{numero} / {emitente}", provider))
            resultados_meta.append({
                "arquivo_origem": name,
                "pagina": idx+1,
                "emitente_detectado": emitente_raw,
                "numero_detectado": numero_raw,
                "status": "OK",
                "tempo_s": round(tempo, 2),
                "provider": provider
            })

            progresso += 1
            progress_bar.progress(min(progresso/total_paginas, 1.0))
            progresso_text.markdown(f"<span class='success-log'>✅ {page_label} — OK ({tempo:.2f}s) <span class='{provider_class}'>[{provider}]</span></span>", unsafe_allow_html=True)

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
# PAINEL CORPORATIVO
# =====================================================================
if "resultados" in st.session_state:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📋 Gerenciamento — selecione e aplique ações")
    resultados = st.session_state["resultados"]
    session_folder = Path(st.session_state["session_folder"])
    novos_nomes = st.session_state.get("novos_nomes", {r["file"]: r["file"] for r in resultados})
    files_meta = st.session_state.get("files_meta", {})

    col1, col2, col3, col4 = st.columns([3,2,2,2])
    with col1:
        q = st.text_input("🔎 Buscar arquivo ou emitente", value="", placeholder="parte do nome, emitente ou número")
    with col2:
        sort_by = st.selectbox("Ordenar por", ["Nome (A-Z)", "Nome (Z-A)", "Número (asc)", "Número (desc)"], index=0)
    with col3:
        show_logs = st.checkbox("Mostrar logs detalhados", value=False)
    with col4:
        if st.button("⬇️ Baixar Selecionadas"):
            sel = st.session_state.get("selected_files", [])
            if not sel:
                st.warning("Nenhuma nota selecionada para download.")
            else:
                mem = io.BytesIO()
                with zipfile.ZipFile(mem, "w") as zf:
                    for f in sel:
                        src = session_folder / f
                        if src.exists():
                            arcname = novos_nomes.get(f, f)
                            zf.write(src, arcname=arcname)
                mem.seek(0)
                st.download_button("⬇️ Clique novamente para confirmar download", data=mem, file_name="selecionadas.zip", mime="application/zip")
        if st.button("🗑️ Excluir Selecionadas"):
            sel = st.session_state.get("selected_files", [])
            if not sel:
                st.warning("Nenhuma nota selecionada para exclusão.")
            else:
                count = 0
                for f in sel:
                    src = session_folder / f
                    try:
                        if src.exists():
                            src.unlink()
                    except Exception:
                        pass
                    st.session_state["resultados"] = [r for r in st.session_state["resultados"] if r["file"] != f]
                    if f in st.session_state.get("novos_nomes", {}):
                        st.session_state["novos_nomes"].pop(f, None)
                    if f in st.session_state.get("files_meta", {}):
                        st.session_state["files_meta"].pop(f, None)
                    count += 1
                st.success(f"{count} arquivo(s) excluído(s).")
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

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
    st.markdown("### 🗂 Notas processadas")
    if "selected_files" not in st.session_state:
        st.session_state["selected_files"] = []

    for r in visible:
        fname = r["file"]
        meta = files_meta.get(fname, {})
        cols = st.columns([0.06, 0.48, 0.28, 0.18])
        
        checked = fname in st.session_state.get("selected_files", [])
        cb = cols[0].checkbox("", value=checked, key=f"cb_{fname}")
        
        if cb and fname not in st.session_state["selected_files"]:
            st.session_state["selected_files"].append(fname)
        if (not cb) and fname in st.session_state["selected_files"]:
            st.session_state["selected_files"].remove(fname)

        novos_nomes[fname] = cols[1].text_input(label=fname, value=novos_nomes.get(fname, fname), key=f"rename_input_{fname}")

        emit = meta.get("emitente", r.get("emitente", "-"))
        num = meta.get("numero", r.get("numero", "-"))
        cols[2].markdown(f"<div class='small-note'>{emit}  •  Nº {num}  •  {r.get('pages',1)} pág(s)</div>", unsafe_allow_html=True)

        action_col = cols[3]
        action = action_col.selectbox("", options=["...", "Remover (mover p/ lixeira)", "Baixar este arquivo"], key=f"action_{fname}", index=0)
        
        if action_col.button("⚙️ Gerenciar", key=f"manage_{fname}"):
            st.session_state["_manage_target"] = fname
            st.rerun()

        if action == "Remover (mover p/ lixeira)":
            src = session_folder / fname
            try:
                if src.exists():
                    src.unlink()
            except Exception:
                pass
            st.session_state["resultados"] = [x for x in st.session_state["resultados"] if x["file"] != fname]
            if fname in st.session_state.get("novos_nomes", {}):
                st.session_state["novos_nomes"].pop(fname, None)
            st.success(f"{fname} removido.")
            st.rerun()
        elif action == "Baixar este arquivo":
            src = session_folder / fname
            if src.exists():
                with open(src, "rb") as ff:
                    data = ff.read()
                st.download_button(f"⬇️ Baixar {fname}", data=data, file_name=novos_nomes.get(fname, fname), mime="application/pdf")
            else:
                st.warning("Arquivo não encontrado.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Dashboard analítico
    criar_dashboard_analitico()

    # Mostrar logs se solicitado
    if show_logs and st.session_state.get("processed_logs"):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📝 Logs de processamento (últimas páginas)")
        for entry in st.session_state["processed_logs"][-200:]:
            label, t, status, info, provider = (entry + ("", "", ""))[:5]
            provider_class = f"provider-{provider.lower()}" if provider else ""
            if status == "OK":
                st.markdown(f"<div class='success-log'>✅ {label} — {info} — {t:.2f}s <span class='{provider_class}'>[{provider}]</span></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='warning-log'>⚠️ {label} — {info} <span class='{provider_class}'>[{provider}]</span></div>", unsafe_allow_html=True)
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
            st.download_button("⬇️ Clique para baixar (ZIP)", data=mem, file_name="notas_processadas.zip", mime="application/zip")
    with col_dl_b:
        st.markdown("<div class='small-note'>Dica: edite nomes na lista e use 'Baixar Selecionadas' para baixar apenas o que precisar.</div>", unsafe_allow_html=True)

else:
    st.info("Nenhum arquivo processado ainda. Faça upload e clique em 'Processar PDFs'.")
  # ==============================
# 🔧 GERENCIAR PDF SELECIONADO
# ==============================
if "_manage_target" in st.session_state:
    target_file = st.session_state["_manage_target"]
    target_path = os.path.join(output_dir, target_file)

    st.markdown("---")
    st.subheader(f"⚙️ Gerenciando: `{target_file}`")

    # Mostrar o PDF
    with open(target_path, "rb") as f:
        pdf_bytes = f.read()
        st.download_button("⬇️ Baixar PDF", pdf_bytes, file_name=target_file)
        st.pdf_viewer(target_path)

    # Botão para remover o modo "gerenciar"
    if st.button("⬅️ Voltar"):
        del st.session_state["_manage_target"]
        st.rerun()

    st.markdown("### 🧩 Separar páginas")
    from PyPDF2 import PdfReader, PdfWriter

    pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
    num_pages = len(pdf_reader.pages)

    st.info(f"O PDF tem **{num_pages} páginas**.")
    start_page = st.number_input("Página inicial", min_value=1, max_value=num_pages, value=1)
    end_page = st.number_input("Página final", min_value=1, max_value=num_pages, value=num_pages)

    if st.button("✂️ Separar e salvar nova nota"):
        if start_page <= end_page:
            writer = PdfWriter()
            for i in range(start_page - 1, end_page):
                writer.add_page(pdf_reader.pages[i])
            new_name = f"{Path(target_file).stem}_paginas_{start_page}-{end_page}.pdf"
            new_path = os.path.join(output_dir, new_name)
            with open(new_path, "wb") as nf:
                writer.write(nf)
            st.success(f"✅ Novo PDF salvo: `{new_name}`")
        else:
            st.error("Página inicial não pode ser maior que a final.")

