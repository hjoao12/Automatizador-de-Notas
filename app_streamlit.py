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
    resultados_meta = []   # lista de dicts com info (arquivo_origem, pagina, emitente_detectado, numero_detectado, status)

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
# GERENCIAMENTO SIMPLIFICADO (selecionar + Agrupar Selecionadas)
# =====================================================================
if "resultados" in st.session_state:
    st.subheader("🗂️ Gerenciamento das Notas — Simples e Intuitivo")
    resultados = st.session_state["resultados"]
    session_folder = Path(st.session_state["session_folder"])
    grupos = st.session_state.get("grupos", {"Sem Grupo": [r["file"] for r in resultados]})
    novos_nomes = st.session_state.get("novos_nomes", {r["file"]: r["file"] for r in resultados})

    st.markdown("Use a caixa abaixo para **selecionar notas**. Depois escolha uma ação: **Agrupar Selecionadas**, **Remover Selecionadas do Grupo** ou **Excluir Selecionadas**.")
    st.markdown("Você também pode editar o nome final de cada nota na lista abaixo.")

    # lista de arquivos disponíveis (ordem)
    all_files = [r["file"] for r in resultados]

    # show a simple table-like view with editable names (text_inputs)
    st.markdown("### ✏️ Nomes finais (edite conforme necessário)")
    for f in all_files:
        col1, col2, col3 = st.columns([5,2,1])
        with col1:
            novos_nomes[f] = st.text_input(f"Arquivo: {f}", novos_nomes.get(f, f), key=f"rename_{f}")
        with col2:
            # show meta info
            meta = next((r for r in resultados if r["file"]==f), {})
            st.text(f"{meta.get('emitente','-')} / {meta.get('numero','-')}")
        with col3:
            grp_label = next((g for g, files in grupos.items() if f in files), "Sem Grupo")
            st.text(grp_label)

    st.session_state["novos_nomes"] = novos_nomes

    st.markdown("---")
    st.markdown("### ✅ Seleção e ações em massa")
    selecionadas = st.multiselect("Selecione as notas para a ação:", options=all_files, default=[])

    col_a, col_b, col_c = st.columns([2,2,2])
    with col_a:
        group_name = st.text_input("Nome do grupo para agrupar (ex: FATURAS_JUL)", key="group_name_input")
        if st.button("➕ Agrupar Selecionadas"):
            if not selecionadas:
                st.warning("Nenhuma nota selecionada.")
            else:
                target = group_name.strip() or f"Grupo_{int(time.time())}"
                if target not in grupos:
                    grupos[target] = []
                # move selecionadas para target (removendo de qualquer outro grupo)
                for s in selecionadas:
                    for g, fl in grupos.items():
                        if s in fl:
                            fl.remove(s)
                    grupos[target].append(s)
                st.session_state["grupos"] = grupos
                st.success(f"{len(selecionadas)} nota(s) agrupada(s) em '{target}'.")
    with col_b:
        if st.button("↩️ Remover Selecionadas do Grupo (Mover para Sem Grupo)"):
            if not selecionadas:
                st.warning("Nenhuma nota selecionada.")
            else:
                for s in selecionadas:
                    for g, fl in grupos.items():
                        if s in fl:
                            fl.remove(s)
                grupos.setdefault("Sem Grupo", [])
                grupos["Sem Grupo"].extend(selecionadas)
                # dedupe
                grupos["Sem Grupo"] = list(dict.fromkeys(grupos["Sem Grupo"]))
                st.session_state["grupos"] = grupos
                st.success(f"{len(selecionadas)} nota(s) movida(s) para 'Sem Grupo'.")
    with col_c:
        if st.button("🗑️ Excluir Selecionadas"):
            if not selecionadas:
                st.warning("Nenhuma nota selecionada.")
            else:
                for s in selecionadas:
                    # remove from groups
                    for g, fl in list(grupos.items()):
                        if s in fl:
                            fl.remove(s)
                    # remove physical file
                    src = session_folder / s
                    try:
                        if src.exists():
                            src.unlink()
                    except Exception:
                        pass
                    # remove from resultados list
                    st.session_state["resultados"] = [r for r in st.session_state["resultados"] if r["file"] != s]
                    # remove name entry
                    if s in st.session_state.get("novos_nomes", {}):
                        st.session_state["novos_nomes"].pop(s, None)
                st.session_state["grupos"] = grupos
                st.success(f"{len(selecionadas)} nota(s) excluída(s).")

    st.markdown("---")
    st.markdown("### 📂 Grupos existentes (clique para expandir)")
    # show groups and contents
    for gname, files in grupos.items():
        with st.expander(f"{gname} — {len(files)} notas", expanded=False):
            if not files:
                st.write("_(vazio)_")
            else:
                for f in files:
                    meta = next((r for r in resultados if r["file"]==f), {})
                    cols = st.columns([6,2,1])
                    with cols[0]:
                        st.text(f)
                    with cols[1]:
                        st.text(f"{meta.get('emitente','-')}")
                    with cols[2]:
                        if st.button("Remover", key=f"rm_{gname}_{f}"):
                            # move to Sem Grupo
                            grupos[gname].remove(f)
                            grupos.setdefault("Sem Grupo", []).append(f)
                            st.session_state["grupos"] = grupos
                            st.experimental_rerun()

    st.markdown("---")
    # group rename / delete
    st.markdown("### ⚙️ Gerenciar grupos")
    col1, col2 = st.columns(2)
    with col1:
        sel_group = st.selectbox("Selecionar grupo para renomear:", [g for g in grupos.keys() if g!="Sem Grupo"], index=0 if any(g!="Sem Grupo" for g in grupos.keys()) else 0)
        new_name_for_group = st.text_input("Novo nome do grupo:", key="rename_group_input")
        if st.button("Renomear grupo"):
            if sel_group and new_name_for_group:
                grupos[new_name_for_group] = grupos.pop(sel_group)
                st.session_state["grupos"] = grupos
                st.success(f"Grupo '{sel_group}' renomeado para '{new_name_for_group}'.")
    with col2:
        del_group = st.selectbox("Selecionar grupo para excluir (moverá as notas para 'Sem Grupo'):", [g for g in grupos.keys() if g!="Sem Grupo"], index=0 if any(g!="Sem Grupo" for g in grupos.keys()) else 0)
        if st.button("Excluir grupo"):
            if del_group and del_group in grupos:
                itens = grupos.pop(del_group)
                grupos.setdefault("Sem Grupo", []).extend(itens)
                # dedupe
                grupos["Sem Grupo"] = list(dict.fromkeys(grupos["Sem Grupo"]))
                st.session_state["grupos"] = grupos
                st.success(f"Grupo '{del_group}' excluído e suas notas movidas para 'Sem Grupo'.")

    st.session_state["grupos"] = grupos
    st.session_state["novos_nomes"] = novos_nomes

    st.markdown("---")
    # Final: gerar ZIP com nomes editados e grupos
    if st.button("📦 Baixar ZIP final (respeita agrupamentos e nomes)"):
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
                        # use group name as arcname (but allow override via novos_nomes if user edited)
                        zf.write(tmp_path, arcname=novos_nomes.get(grouped_name, grouped_name))
        memory_zip.seek(0)
        st.download_button("⬇️ Baixar notas finais (ZIP)", data=memory_zip, file_name="notas_processadas.zip", mime="application/zip")
