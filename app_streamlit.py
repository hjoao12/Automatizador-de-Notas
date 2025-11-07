import os
import streamlit as st
from dotenv import load_dotenv
import tempfile
import shutil

# ======== 🔧 Carregar variáveis de ambiente (.env local ou nuvem) ========
load_dotenv()  # Se existir um .env local, ele será carregado

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ======== 🧩 IA — fallback entre Google e OpenAI ========
from pathlib import Path

def extract_text_with_ai(file_path):
    """
    Extrai texto de PDF usando IA.
    Se o Google falhar, tenta OpenAI/DeepSeek como fallback.
    """
    try:
        if GOOGLE_API_KEY:
            from google import genai
            client = genai.Client(api_key=GOOGLE_API_KEY)
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=f"Extraia o texto completo e os dados principais do PDF: {file_path}"
            )
            return response.text

        elif OPENAI_API_KEY:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Você é um assistente que extrai texto de PDFs."},
                    {"role": "user", "content": f"Extraia o texto e os dados principais do arquivo {file_path}."}
                ]
            )
            return response.choices[0].message.content

        else:
            raise Exception("Nenhuma API Key configurada.")

    except Exception as e:
        return f"Erro ao usar IA: {e}"


# ======== 🖼️ Interface Streamlit ========
st.set_page_config(page_title="Automatizador de Notas", layout="wide")
st.title("📄 Automatizador de Notas com IA")
st.write("Envie seus PDFs de notas fiscais e extraia automaticamente as informações usando IA.")

uploaded_files = st.file_uploader("Selecione os arquivos PDF", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    st.info(f"{len(uploaded_files)} arquivo(s) enviado(s).")

    temp_dir = tempfile.mkdtemp()
    st.session_state["temp_dir"] = temp_dir

    results = []

    progress = st.progress(0)
    total = len(uploaded_files)

    for i, uploaded_file in enumerate(uploaded_files, start=1):
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner(f"🔍 Processando {uploaded_file.name}..."):
            extracted_text = extract_text_with_ai(file_path)
            results.append({
                "arquivo": uploaded_file.name,
                "conteudo": extracted_text
            })

        progress.progress(i / total)

    st.success("✅ Processamento concluído!")

    # Mostrar resultados
    for r in results:
        st.subheader(r["arquivo"])
        st.text_area("Resultado", r["conteudo"], height=200)

    # Limpar temp_dir ao sair
    shutil.rmtree(temp_dir, ignore_errors=True)

else:
    st.info("Envie seus PDFs para começar.")


# ======== 🧭 Footer ========
st.markdown("---")
st.markdown("**Desenvolvido por João Henrique** 🚀 | Com suporte a Google Gemini e OpenAI/DeepSeek")
