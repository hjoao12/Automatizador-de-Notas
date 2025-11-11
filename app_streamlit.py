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
</style>
""", unsafe_allow_html=True)

st.title("🧠 Automatizador de Notas Fiscais PDF")


# pequenas variáveis de estilo (corporativo claro)
PRIMARY = "#0f4c81"   # azul petróleo
ACCENT = "#6fb3b8"    # verde-menta claro (agridoce suave)
BG = "#F7FAFC"
CARD_BG = "#FFFFFF"
TEXT_MUTED = "#6b7280"
WARN = "#f6c85f"      # amarelo suave para aviso
ERROR = "#e76f51"     # uso raro (evitar vermelho dominante)

# CSS para deixar o visual corporativo claro e organizado
st.markdown(f"""
    <style>
      :root {{ --primary: {PRIMARY}; --accent: {ACCENT}; --bg: {BG}; --card: {CARD_BG}; --muted:{TEXT_MUTED}; --warn:{WARN}; }}
      .stApp {{ background: var(--bg); color: #0b1220; }}
      .card {{ background: var(--card); padding: 14px; border-radius: 8px; box-shadow: 0 6px 18px rgba(15,76,129,0.06); margin-bottom: 12px; }}
      .title-small {{ color: var(--primary); font-weight: 600; }}
      .muted {{ color: var(--muted); font-size: 13px; }}
      .log-ok {{ color: #0b8457; font-weight: 600; }}
      .log-warn {{ color: var(--warn); font-weight: 600; }}
      .top-actions { display:flex; gap:10px; align-items:center;}
      .small-note { font-size:13px; color:var(--muted); }
      .file-row { padding:10px 8px; border-radius:6px; background: linear-gradient(180deg, rgba(15,76,129,0.02), transparent); margin-bottom:6px; }
      button[title="download-btn"] { background:var(--accent) !important; }
    </style>
""", unsafe_allow_html=True)

TEMP_FOLDER = Path("./temp")
os.makedirs(TEMP_FOLDER, exist_ok=True)

MAX_TOTAL_PAGES = int(os.getenv("MAX_TOTAL_PAGES", "50"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
MIN_RETRY_DELAY = int(os.getenv("MIN_RETRY_DELAY", "5"))
MAX_RETRY_DELAY = int(os.getenv("MAX_RETRY_DELAY", "30"))
MODEL_NAME = os.getenv("MODEL_NAME", "models/gemini-2.0-flash")

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ Chave GOOGLE_API_KEY não encontrada.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

st.markdown("<div class='muted'>Conectando ao modelo...</div>", unsafe_allow_html=True)
try:
    # Apenas para checar; não chama geração
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
    return numero.lstrip("0") or "0"

# =====================================================================
# RETRY GEMINI
# =====================================================================
def calcular_delay(tentativa, error_msg):
    if "retry in" in error_msg.lower():
        try:
            return min(float(re.search(r"retry in (\d+\.?\d*)s", error_msg.lower()).group(1)) + 2, MAX_RETRY_DELAY)
        except:
            pass
    return min(MIN_RETRY_DELAY * (tentativa + 1), MAX_RETRY_DELAY)

def chamar_gemini_retry(model, prompt_instrucao, page_stream):
    for tentativa in range(MAX_RETRIES + 1):
        try:
            start = time.time()
            resp = model.generate_content(
                [prompt_instrucao, {"mime_type": "application/pdf", "data": page_stream.getvalue()}],
                generation_config={"response_mime_type": "application/json"},
                request_options={'timeout': 60}
            )
            tempo = round(time.time() - start, 2)
            texto = resp.text.strip().lstrip("```json").rstrip("```").strip()
            try:
                dados = json.loads(texto)
            except Exception:
                dados = {"error": "Resposta da IA não era JSON", "_raw": texto}
            return dados, True, tempo
        except ResourceExhausted as e:
            delay = calcular_delay(tentativa, str(e))
            st.warning(f"⚠️ Quota excedida (tentativa {tentativa + 1}/{MAX_RETRIES}). Aguardando {delay}s...")
            time.sleep(delay)
        except Exception as e:
            if tentativa < MAX_RETRIES:
                time.sleep(MIN_RETRY_DELAY)
            else:
                return {"error": str(e)}, False, 0
    return {"error": "Falha máxima de tentativas"}, False, 0

# =====================================================================
# Upload e Processamento (mantém painel de progresso + logs coloridos)
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
    # remove temp folder contents and reset session state
    if "session_folder" in st.session_state:
        try:
            shutil.rmtree(st.session_state["session_folder"])
        except Exception:
            pass
    for k in ["resultados", "session_folder", "novos_nomes", "processed_logs", "files_meta"]:
        if k in st.session_state:
            del st.session_state[k]
    st.success("Sessão limpa.")
    st.experimental_rerun()

if uploaded_files and process_btn:
    session_id = str(uuid.uuid4())
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

    # count pages
    total_paginas = 0
    for a in arquivos:
        try:
            r = PdfReader(io.BytesIO(a["bytes"]))
            total_paginas += len(r.pages)
        except Exception:
            st.warning(f"Arquivo inválido: {a['name']}")

    st.info(f"📄 Total de páginas a processar: {total_paginas}")

    # prepare structures
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
            processed_logs.append((name, 0, "ERRO_LEITURA"))
            continue

        for idx, page in enumerate(reader.pages):
            b = io.BytesIO()
            w = PdfWriter()
            w.add_page(page)
            w.write(b)
            b.seek(0)

            dados, ok, tempo = chamar_gemini_retry(model, prompt, b)
            page_label = f"{name} (pág {idx+1})"
            if not ok or "error" in dados:
                processed_logs.append((page_label, tempo, "ERRO_IA", dados.get("error", str(dados))))
                progresso += 1
                progress_bar.progress(min(progresso/total_paginas, 1.0))
                progresso_text.markdown(f"<span class='log-warn'>⚠️ {page_label} — ERRO IA</span>", unsafe_allow_html=True)
                resultados_meta.append({
                    "arquivo_origem": name,
                    "pagina": idx+1,
                    "emitente_detectado": dados.get("emitente") if isinstance(dados, dict) else "-",
                    "numero_detectado": dados.get("numero_nota") if isinstance(dados, dict) else "-",
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

            progresso += 1
            progress_bar.progress(min(progresso/total_paginas, 1.0))
            progresso_text.markdown(f"<span class='log-ok'>✅ {page_label} — OK ({tempo:.2f}s)</span>", unsafe_allow_html=True)

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

    st.success(f"✅ Processamento concluído em {round(time.time() - start_all, 2)}s — {len(resultados)} arquivos gerados.")
    st.rerun()

# =====================================================================
# PAINEL CORPORATIVO (SEM GRUPOS) - seleção múltipla + ações no topo
# =====================================================================
if "resultados" in st.session_state:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📋 Gerenciamento — selecione e aplique ações")
    resultados = st.session_state["resultados"]
    session_folder = Path(st.session_state["session_folder"])
    novos_nomes = st.session_state.get("novos_nomes", {r["file"]: r["file"] for r in resultados})
    files_meta = st.session_state.get("files_meta", {})

    # Top action bar: filtro, busca, ações
    col1, col2, col3, col4 = st.columns([3,2,2,2])
    with col1:
        q = st.text_input("🔎 Buscar arquivo ou emitente", value="", placeholder="parte do nome, emitente ou número")
    with col2:
        sort_by = st.selectbox("Ordenar por", ["Nome (A-Z)", "Nome (Z-A)", "Número (asc)", "Número (desc)"], index=0)
    with col3:
        show_logs = st.checkbox("Mostrar logs detalhados", value=False)
    with col4:
        # action buttons
        if st.button("⬇️ Baixar Selecionadas"):
            sel = st.session_state.get("selected_files", [])
            if not sel:
                st.warning("Nenhuma nota selecionada para download.")
            else:
                # create zip with selected
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
                    # remove from resultados
                    st.session_state["resultados"] = [r for r in st.session_state["resultados"] if r["file"] != f]
                    if f in st.session_state.get("novos_nomes", {}):
                        st.session_state["novos_nomes"].pop(f, None)
                    if f in st.session_state.get("files_meta", {}):
                        st.session_state["files_meta"].pop(f, None)
                    count += 1
                st.success(f"{count} arquivo(s) excluído(s).")
                st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # Build visible list filtered/search/sorted
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

    # display table-like with selection checkboxes and inline rename
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🗂 Notas processadas")
    # remember selected files across reruns
    if "selected_files" not in st.session_state:
        st.session_state["selected_files"] = []

    # we'll show simple table: checkbox | filename (editable) | meta | actions
    for r in visible:
        fname = r["file"]
        meta = files_meta.get(fname, {})
        cols = st.columns([0.06, 0.55, 0.25, 0.14])
        # checkbox - maintain selection state
        checked = fname in st.session_state.get("selected_files", [])
        cb = cols[0].checkbox("", value=checked, key=f"cb_{fname}")
        # handle selection persistence
        if cb and fname not in st.session_state["selected_files"]:
            st.session_state["selected_files"].append(fname)
        if (not cb) and fname in st.session_state["selected_files"]:
            st.session_state["selected_files"].remove(fname)

        # editable name
        novos_nomes[fname] = cols[1].text_input(label=fname, value=novos_nomes.get(fname, fname), key=f"rename_input_{fname}")

        # meta column
        emit = meta.get("emitente", r.get("emitente", "-"))
        num = meta.get("numero", r.get("numero", "-"))
        cols[2].markdown(f"<div class='small-note'>{emit}  •  Nº {num}  •  {r.get('pages',1)} pág(s)</div>", unsafe_allow_html=True)

        # actions dropdown simulation: small selectbox of actions per row
        action = cols[3].selectbox("", options=["...", "Remover (mover p/ lixeira)", "Baixar este arquivo"], key=f"action_{fname}", index=0)
        if action == "Remover (mover p/ lixeira)":
            # remove file
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
            st.experimental_rerun()
        elif action == "Baixar este arquivo":
            src = session_folder / fname
            if src.exists():
                with open(src, "rb") as ff:
                    data = ff.read()
                st.download_button(f"⬇️ Baixar {fname}", data=data, file_name=novos_nomes.get(fname, fname), mime="application/pdf")
            else:
                st.warning("Arquivo não encontrado.")
            # reset action selector back to ...
            st.session_state[f"action_{fname}"] = "..."
        st.markdown("<hr/>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # show logs if requested
    if show_logs and st.session_state.get("processed_logs"):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📝 Logs de processamento (últimas páginas)")
        for entry in st.session_state["processed_logs"][-200:]:
            label, t, status, info = (entry + ("", ""))[:4]
            if status == "OK":
                st.markdown(f"<div class='log-ok'>✅ {label} — {info} — {t:.2f}s</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='log-warn'>⚠️ {label} — {info}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Persist edited names
    st.session_state["novos_nomes"] = novos_nomes

    st.markdown("---")
    # Final download: all (respect novos_nomes)
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

# =====================================================================
# caso não haja resultados ainda
# =====================================================================
else:
    st.info("Nenhum arquivo processado ainda. Faça upload e clique em 'Processar PDFs'.")
