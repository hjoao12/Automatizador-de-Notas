import os
import io
import time
import json
import zipfile
import uuid
import shutil
import unicodedata
import re
from pathlib import Path
from PyPDF2 import PdfReader, PdfWriter
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
import streamlit as st
from dotenv import load_dotenv

# =====================================================================
# CONFIGURAÇÃO INICIAL
# =====================================================================
load_dotenv()
st.set_page_config(page_title="Automatizador de Notas Fiscais", page_icon="🧾", layout="wide")

# ======= CSS Corporativo Claro (corrigido, seguro e completo) =======
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
.log-ok { color: #0b8457; font-weight: 600; }
.log-warn { color: #f6c85f; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Automatizador de Notas Fiscais PDF")

# Configurações de Paths
TEMP_FOLDER = Path("./temp")
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Configurações de Limites
MAX_TOTAL_PAGES = int(os.getenv("MAX_TOTAL_PAGES", "30")) # Reduzido para evitar quota
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
MIN_RETRY_DELAY = int(os.getenv("MIN_RETRY_DELAY", "10")) # Aumentado
MAX_RETRY_DELAY = int(os.getenv("MAX_RETRY_DELAY", "60")) # Aumentado
REQUEST_DELAY = int(os.getenv("REQUEST_DELAY", "2")) # Delay entre requisições
MODEL_NAME = os.getenv("MODEL_NAME", "models/gemini-1.5-flash") # Modelo mais leve

# Constante do Prompt de Instrução (Para melhor organização)
PROMPT_INSTRUCAO_GEMINI = (
"Você é um especialista em análise de notas fiscais DANFE."
"Extraia APENAS estes 3 campos do documento:"
"1. emitente (nome da empresa que emitiu a nota)"
"2. numero_nota (número da nota fiscal)" 
"3. cidade (cidade do emitente)"
"Responda EXCLUSIVAMENTE em JSON válido: {\"emitente\":\"...\",\"numero_nota\":\"...\",\"cidade\":\"...\"}"
"Se algum campo não for encontrado, use string vazia \"\"."
)

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ Chave GOOGLE_API_KEY não encontrada.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

st.markdown("<div class='muted'>Conectando ao modelo...</div>", unsafe_allow_html=True)
try:
    _ = model.name
    st.success("✅ Google Gemini configurado.")
except Exception:
    st.warning("⚠️ Problema ao conectar com Gemini — verifique a variável de ambiente GOOGLE_API_KEY.")

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
    # Tenta converter para int e depois para string, garantindo a remoção de zeros à esquerda.
    try:
        return str(int(numero))
    except ValueError:
        # Retorna "0" ou o número original limpo se a conversão falhar
        return numero.lstrip("0") or "0"

# =====================================================================
# RETRY GEMINI MELHORADO
# =====================================================================
def calcular_delay(tentativa, error_msg):
    if "retry in" in error_msg.lower():
        try:
            return min(float(re.search(r"retry in (\d+\.?\d*)s", error_msg.lower()).group(1)) + 5, MAX_RETRY_DELAY)
        except:
            pass
    # Backoff exponencial com jitter
    base_delay = MIN_RETRY_DELAY * (2 ** tentativa)
    jitter = base_delay * 0.1  # 10% de jitter
    return min(base_delay + jitter, MAX_RETRY_DELAY)

def chamar_gemini_retry(model, prompt_instrucao, page_stream):
    for tentativa in range(MAX_RETRIES + 1):
        try:
            start = time.time()
            # Timeout mais conservador
            resp = model.generate_content(
                [prompt_instrucao, {"mime_type": "application/pdf", "data": page_stream.getvalue()}],
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.1  # Mais determinístico
                },
                request_options={'timeout': 120}  # Timeout aumentado
            )
            tempo = round(time.time() - start, 2)
            
            if not resp.text:
                raise Exception("Resposta vazia da API")
                
            texto = resp.text.strip().lstrip("```json").rstrip("```").strip()
            try:
                dados = json.loads(texto)
                # Validação básica dos dados
                if not isinstance(dados, dict):
                    raise ValueError("Resposta não é um objeto JSON")
                return dados, True, tempo
            except json.JSONDecodeError as e:
                if tentativa < MAX_RETRIES:
                    st.warning(f"⚠️ Resposta não é JSON válido (tentativa {tentativa + 1}), tentando novamente...")
                    time.sleep(MIN_RETRY_DELAY)
                    continue
                else:
                    dados = {"error": f"JSON inválido: {str(e)}", "_raw": texto[:200]}
                    return dados, False, tempo
                    
        except ResourceExhausted as e:
            delay = calcular_delay(tentativa, str(e))
            st.warning(f"⚠️ Quota excedida (tentativa {tentativa + 1}/{MAX_RETRIES}). Aguardando {delay}s...")
            time.sleep(delay)
        except Exception as e:
            error_msg = str(e).lower()
            if "quota" in error_msg or "rate limit" in error_msg:
                delay = calcular_delay(tentativa, error_msg)
                st.warning(f"⚠️ Limite de taxa excedido (tentativa {tentativa + 1}/{MAX_RETRIES}). Aguardando {delay}s...")
                time.sleep(delay)
            elif tentativa < MAX_RETRIES:
                delay = MIN_RETRY_DELAY
                st.warning(f"⚠️ Erro temporário (tentativa {tentativa + 1}): {str(e)[:100]}... Aguardando {delay}s")
                time.sleep(delay)
            else:
                return {"error": str(e)}, False, 0
                
    return {"error": "Falha máxima de tentativas"}, False, 0

# =====================================================================
# Upload e Processamento
# =====================================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 📎 Enviar PDFs e processar (uma vez)")

# Aviso sobre limites
st.warning(f"⚠️ **Limites atuais:** Máximo de {MAX_TOTAL_PAGES} páginas por processamento para evitar quota excedida.")

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
            # Uso de Path(string) para consistência no rmtree
            shutil.rmtree(Path(st.session_state["session_folder"]))
        except Exception:
            pass
    for k in ["resultados", "session_folder", "novos_nomes", "processed_logs", "files_meta", "selected_files", "_manage_target"]:
        if k in st.session_state:
            del st.session_state[k]
    st.success("Sessão limpa.")
    st.experimental_rerun()

if uploaded_files and process_btn:
    session_id = str(uuid.uuid4())
    # session_folder é um objeto Path
    session_folder = TEMP_FOLDER / session_id
    os.makedirs(session_folder, exist_ok=True)

    # read all files once
    arquivos = []
    for f in uploaded_files:
        try:
            b = f.read()
            arquivos.append({"name": f.name, "bytes": b})
        except Exception:
            st.warning(f"Erro ao ler {f.name}, ignorado.")

    # count pages with validação de limite
    total_paginas = 0
    for a in arquivos:
        try:
            r = PdfReader(io.BytesIO(a["bytes"]))
            total_paginas += len(r.pages)
        except Exception:
            st.warning(f"Arquivo inválido: {a['name']}")

    if total_paginas > MAX_TOTAL_PAGES:
        st.error(f"❌ Limite excedido: {total_paginas} páginas detectadas (máximo: {MAX_TOTAL_PAGES}). Reduza a quantidade de arquivos ou páginas.")
        st.stop()

    st.info(f"📄 Total de páginas a processar: {total_paginas}")

    # prepare structures
    agrupados_bytes = {}
    resultados_meta = []
    processed_logs = []
    progresso = 0
    progress_bar = st.progress(0.0)
    progresso_text = st.empty()
    start_all = time.time()

    # Usa o prompt definido na constante
    prompt = PROMPT_INSTRUCAO_GEMINI

    # Container para logs em tempo real
    log_container = st.container()
    
    successful_pages = 0
    failed_pages = 0

    for a in arquivos:
        name = a["name"]
        try:
            reader = PdfReader(io.BytesIO(a["bytes"]))
        except Exception:
            processed_logs.append((name, 0, "ERRO_LEITURA", "Não foi possível ler o PDF"))
            failed_pages += 1
            continue

        for idx, page in enumerate(reader.pages):
            # Delay entre requisições para evitar rate limiting
            if progresso > 0:
                time.sleep(REQUEST_DELAY)
                
            b = io.BytesIO()
            w = PdfWriter()
            w.add_page(page)
            w.write(b)
            b.seek(0)

            dados, ok, tempo = chamar_gemini_retry(model, prompt, b)
            page_label = f"{name} (pág {idx+1})"
            
            if not ok or "error" in dados:
                error_msg = dados.get("error", str(dados))
                processed_logs.append((page_label, tempo, "ERRO_IA", error_msg))
                failed_pages += 1
                progresso += 1
                progress_bar.progress(min(progresso/total_paginas, 1.0))
                progresso_text.markdown(f"<span class='log-warn'>⚠️ {page_label} — ERRO IA: {error_msg[:80]}...</span>", unsafe_allow_html=True)
                resultados_meta.append({
                    "arquivo_origem": name,
                    "pagina": idx+1,
                    "emitente_detectado": "-",
                    "numero_detectado": "-", 
                    "status": "ERRO"
                })
                continue

            emitente_raw = dados.get("emitente", "") or ""
            numero_raw = dados.get("numero_nota", "") or ""
            cidade_raw = dados.get("cidade", "") or ""

            numero = limpar_numero(numero_raw)
            nome_map = substituir_nome_emitente(emitente_raw, cidade_raw)
            emitente = limpar_emitente(nome_map)

            key = (numero, emitente)
            agrupados_bytes.setdefault(key, []).append(b.getvalue())

            processed_logs.append((page_label, tempo, "OK", f"{numero} / {emitente}"))
            resultados_meta.append({
                "arquivo_origem": name,
                "pagina": idx+1,
                "emitente_detectado": emitente_raw,
                "numero_detectado": numero_raw,
                "status": "OK",
                "tempo_s": round(tempo, 2)
            })

            successful_pages += 1
            progresso += 1
            progress_bar.progress(min(progresso/total_paginas, 1.0))
            progresso_text.markdown(f"<span class='log-ok'>✅ {page_label} — OK ({tempo:.2f}s) → {numero} / {emitente}</span>", unsafe_allow_html=True)

    # write final grouped pdfs to session folder
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

    # persist in session_state
    st.session_state["resultados"] = resultados
    st.session_state["session_folder"] = str(session_folder)
    st.session_state["novos_nomes"] = {r["file"]: r["file"] for r in resultados}
    st.session_state["processed_logs"] = processed_logs
    st.session_state["files_meta"] = files_meta

    tempo_total = round(time.time() - start_all, 2)
    st.success(f"✅ Processamento concluído em {tempo_total}s — {successful_pages} páginas processadas, {failed_pages} falhas")
    st.info(f"📊 Resultado: {len(resultados)} arquivos PDF gerados a partir de {successful_pages} páginas válidas")
    
    st.rerun()

# =====================================================================
# GERENCIAMENTO E DOWNLOAD
# =====================================================================
if "resultados" in st.session_state and st.session_state["resultados"]:
    
    st.markdown("---")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🗂️ Gerenciar e Baixar Arquivos Finais")

    resultados = st.session_state["resultados"]
    session_folder = Path(st.session_state["session_folder"])
    novos_nomes = st.session_state.get("novos_nomes", {})

    st.info(f"Arquivos gerados: **{len(resultados)}** documentos finais.")

    # Tabela de Resultados para Edição de Nomes
    st.markdown("#### 📝 Revisar Nomes dos Arquivos")
    cols = st.columns([1, 2, 2, 1])
    cols[0].markdown("**#**")
    cols[1].markdown("**Número da Nota**")
    cols[2].markdown("**Emitente (Normalizado)**")
    cols[3].markdown("**Páginas**")
    
    for i, r in enumerate(resultados):
        cols = st.columns([1, 2, 2, 1])
        cols[0].write(i + 1)
        cols[1].text(r["numero"])
        cols[2].text(r["emitente"])
        cols[3].write(r["pages"])
    
    # Ação de Download em Massa
    st.markdown("#### 📥 Download em Lote")
    
    try:
        # Cria o ZIP em memória
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for r in resultados:
                nome_original = r["file"]
                caminho_arquivo = session_folder / nome_original
                if caminho_arquivo.exists():
                    with open(caminho_arquivo, "rb") as f:
                        zip_file.writestr(nome_original, f.read())

        st.download_button(
            label="⬇️ Baixar Todos os PDFs (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"notas_fiscais_agrupadas_{time.strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
            help="Baixa todos os arquivos PDF agrupados em um único arquivo ZIP."
        )
        
    except Exception as e:
        st.error(f"Erro ao criar arquivo ZIP: {e}")
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Logs de Processamento
    st.markdown("---")
    st.markdown("### 📋 Logs Detalhados do Processamento")
    
    logs = st.session_state.get("processed_logs", [])
    if logs:
        st.dataframe(
            [{"Arquivo/Pág": l[0], "Tempo (s)": l[1], "Status": l[2], "Detalhe": l[3]} for l in logs],
            use_container_width=True
        )
    else:
        st.write("Nenhum log de processamento encontrado.")
