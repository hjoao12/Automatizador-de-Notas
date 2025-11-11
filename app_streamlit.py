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
st.set_page_config(page_title="Automatizador de Notas", page_icon="🧾", layout="wide")
st.title("🧠 Automatizador de Notas Fiscais PDF")

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
st.success("✅ Google Gemini configurado com sucesso!")

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

    # default: normalize and replace spaces by underscore (upper case)
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
            # defensiva: garantir dict
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
# UI: Upload e Processamento (uma vez)
# =====================================================================
st.subheader("📎 Faça upload de um ou mais arquivos PDF")
uploaded_files = st.file_uploader("Selecione arquivos PDF", type=["pdf"], accept_multiple_files=True)

if uploaded_files and st.button("🚀 Processar PDFs"):
    # prepara sessão
    session_id = str(uuid.uuid4())
    session_folder = TEMP_FOLDER / session_id
    os.makedirs(session_folder, exist_ok=True)

    # lemos arquivos 1x
    arquivos = []
    for f in uploaded_files:
        content = f.read()
        arquivos.append({"name": f.name, "bytes": content})

    # conta páginas
    total_paginas = 0
    for a in arquivos:
        try:
            r = PdfReader(io.BytesIO(a["bytes"]))
            total_paginas += len(r.pages)
        except Exception:
            st.warning(f"Arquivo {a['name']} inválido — será ignorado.")

    st.info(f"📄 Total de páginas a processar: {total_paginas}")
    prompt = (
        "Analise a nota fiscal (DANFE). Extraia emitente, número da nota e cidade. "
        "Responda SOMENTE em JSON: {\"emitente\":\"NOME\",\"numero_nota\":\"NUMERO\",\"cidade\":\"CIDADE\"}"
    )

    agrupados_bytes = {}   # chave (numero, emitente) -> list de page bytes
    resultados_meta = []   # lista de dicts com info (novo nome, numero, emitente, paginas)

    progresso = 0
    progress_bar = st.progress(0.0)
    progresso_text = st.empty()
    start_all = time.time()

    for a in arquivos:
        name = a["name"]
        try:
            reader = PdfReader(io.BytesIO(a["bytes"]))
        except Exception:
            st.warning(f"Não foi possível ler {name}, pulando.")
            continue

        for idx, page in enumerate(reader.pages):
            # escreve página isolada em bytes
            b = io.BytesIO()
            w = PdfWriter()
            w.add_page(page)
            w.write(b)
            b.seek(0)

            dados, ok, tempo = chamar_gemini_retry(model, prompt, b)
            if not ok or "error" in dados:
                progresso += 1
                progresso_text.text(f"{name} pág {idx+1} — ERRO IA ({dados.get('error','unknown')})")
                progress_bar.progress(min(progresso/total_paginas, 1.0))
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
            progresso_text.text(f"{name} pág {idx+1} — OK ({tempo:.2f}s)")

    # gera PDFs finais por grupo (numero+emitente)
    resultados = []
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

    # salva no session_state para evitar reprocessar
    st.session_state["resultados"] = resultados
    st.session_state["session_folder"] = str(session_folder)
    st.session_state["grupos"] = {"Sem Grupo": [r["file"] for r in resultados]}
    st.session_state["novos_nomes"] = {r["file"]: r["file"] for r in resultados}

    st.success(f"✅ Processamento concluído em {round(time.time() - start_all, 2)}s — {len(resultados)} arquivos gerados.")
    st.rerun()  # recarrega para mostrar a área de gerenciamento

# =====================================================================
# GERENCIAMENTO (drag & drop com streamlit-sortables ou fallback)
# =====================================================================
if "resultados" in st.session_state:
    st.subheader("🗂️ Gerenciamento das Notas (arraste para agrupar / separar)")
    resultados = st.session_state["resultados"]
    session_folder = Path(st.session_state["session_folder"])
    grupos = st.session_state.get("grupos", {"Sem Grupo": [r["file"] for r in resultados]})
    novos_nomes = st.session_state.get("novos_nomes", {r["file"]: r["file"] for r in resultados})

    # Try import drag-and-drop lib; if not installed, fallback UI
    try:
        from streamlit_sortables import sort_items  # type: ignore

        st.markdown("💡 **Arraste as notas entre os blocos abaixo para agrupar.** Crie novos grupos se necessário.")
        # Build items expected by sort_items multi-containers: list[dict] -> {"group": name, "items": [...]}
        items_for_sort = []
        for group_name, files in grupos.items():
            items_for_sort.append({"group": group_name, "items": files})

        # Call sort_items with multi_containers True; shows vertical stacked containers
        new_structure = sort_items(items_for_sort, key="notas_multi", direction="vertical", multi_containers=True)

        # new_structure is a list of dicts {"group": name, "items": [...]}
        # Convert back to grupos mapping
        updated = {}
        for elem in new_structure:
            gname = elem.get("group") or elem.get("header") or None
            flist = elem.get("items", [])
            if gname is None:
                # fallback: try keys
                keys = [k for k in elem.keys() if k != "items"]
                gname = elem.get(keys[0], "Sem Grupo") if keys else "Sem Grupo"
            updated[gname] = flist

        # ensure "Sem Grupo" exists
        if "Sem Grupo" not in updated:
            updated["Sem Grupo"] = []

        # save updated groups
        st.session_state["grupos"] = updated
        grupos = updated

    except Exception as e:
        # Fallback: no drag-and-drop available — show manual grouping UI
        st.warning("Drag & drop não disponível neste ambiente. Usando modo manual.")
        st.markdown("**Modo manual:** selecione notas e escolha um grupo para movê-las.")
        # Show create group UI
        with st.expander("➕ Gerenciar grupos"):
            new_group = st.text_input("Nome do novo grupo (sem espaços):", key="cg")
            if st.button("Criar grupo", key="btn_create"):
                if new_group and new_group not in grupos:
                    grupos[new_group] = []
                    st.session_state["grupos"] = grupos
                    st.success(f"Grupo '{new_group}' criado.")
        # Manual move: select checkboxes and target group
        all_files = [f["file"] for f in resultados]
        to_move = []
        st.markdown("**Selecione notas para mover:**")
        for fname in all_files:
            if st.checkbox(f"{fname}", key=f"chk_{fname}", value=False):
                to_move.append(fname)
        target = st.selectbox("Mover selecionadas para grupo:", list(grupos.keys()), key="sel_group")
        if st.button("Mover selecionadas"):
            for t in to_move:
                # remove from current
                for g, fl in list(grupos.items()):
                    if t in fl:
                        fl.remove(t)
                grupos[target].append(t)
            st.session_state["grupos"] = grupos
            st.experimental_rerun()

    # Show groups and provide rename / delete controls
    st.markdown("---")
    st.markdown("### ✏️ Renomear / Excluir / Separar notas")
    # Build index for lookup to display metadata
    meta_by_file = {r["file"]: r for r in resultados}
    to_delete = []
    for gname, files in grupos.items():
        st.markdown(f"**📁 {gname}** — {len(files)} notas")
        for f in files:
            cols = st.columns([4, 2, 1])
            with cols[0]:
                newname = st.text_input(f"Nome final para {f}", novos_nomes.get(f, f), key=f"rename_{f}")
                novos_nomes[f] = newname
            with cols[1]:
                st.text(f"{meta_by_file.get(f,{}).get('emitente','-')} ({meta_by_file.get(f,{}).get('numero','-')})")
            with cols[2]:
                if st.button("🗑️", key=f"del_{f}"):
                    to_delete.append((gname, f))
        st.markdown("")

    # apply deletions
    if to_delete:
        for gname, f in to_delete:
            if f in grupos.get(gname, []):
                grupos[gname].remove(f)
            # also remove file physically if exists
            fp = session_folder / f
            try:
                if fp.exists():
                    fp.unlink()
            except Exception:
                pass
        st.session_state["grupos"] = grupos
        st.session_state["novos_nomes"] = {**novos_nomes}
        st.success("Notas excluídas com sucesso.")
        st.experimental_rerun()

    # Save name edits
    st.session_state["novos_nomes"] = {**novos_nomes}

    # Create / Merge groups actions
    st.markdown("---")
    with st.expander("🛠️ Ações de grupo"):
        col_a, col_b = st.columns([2,2])
        with col_a:
            src_group = st.selectbox("Selecionar grupo para separar (mover todos para Sem Grupo):", list(grupos.keys()), key="src_grp")
            if st.button("Separar grupo (mover tudo para Sem Grupo)"):
                items = grupos.get(src_group, []).copy()
                grupos["Sem Grupo"].extend(items)
                grupos[src_group] = []
                st.session_state["grupos"] = grupos
                st.success(f"Grupo {src_group} separado.")
                st.experimental_rerun()
        with col_b:
            merge_from = st.multiselect("Selecionar grupos para mesclar:", [g for g in grupos.keys() if g!="Sem Grupo"], key="merge_groups")
            merge_to = st.text_input("Nome do novo grupo (criar/usar):", key="merge_to")
            if st.button("Mesclar selecionados"):
                if not merge_to:
                    st.warning("Informe um nome para o grupo destino.")
                else:
                    if merge_to not in grupos:
                        grupos[merge_to] = []
                    for mg in merge_from:
                        grupos[merge_to].extend(grupos.get(mg, []))
                        grupos[mg] = []
                    st.session_state["grupos"] = grupos
                    st.success(f"Grupos mesclados em {merge_to}.")
                    st.experimental_rerun()

    # Final: gerar ZIP com nomes editados e grupos
    st.markdown("---")
    if st.button("📦 Gerar ZIP Final"):
        memory_zip = io.BytesIO()
        with zipfile.ZipFile(memory_zip, "w") as zf:
            for gname, files in grupos.items():
                if gname == "Sem Grupo":
                    for fname in files:
                        arcname = novos_nomes.get(fname, fname)
                        src = session_folder / fname
                        if src.exists():
                            zf.write(src, arcname=arcname)
                else:
                    # criar PDF agrupado por esse grupo
                    writer = PdfWriter()
                    total_pages = 0
                    for fname in files:
                        src = session_folder / fname
                        if not src.exists():
                            continue
                        try:
                            r = PdfReader(src)
                            for p in r.pages:
                                writer.add_page(p)
                                total_pages += 1
                        except Exception:
                            continue
                    if total_pages > 0:
                        grouped_name = f"{gname}.pdf"
                        tmp_path = session_folder / grouped_name
                        with open(tmp_path, "wb") as of:
                            writer.write(of)
                        zf.write(tmp_path, arcname=novos_nomes.get(grouped_name, grouped_name))
        memory_zip.seek(0)
        st.download_button("⬇️ Baixar notas finais (ZIP)", data=memory_zip, file_name="notas_processadas.zip", mime="application/zip")
