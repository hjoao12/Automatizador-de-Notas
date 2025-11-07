import os
import streamlit as st
import tempfile
import shutil
from dotenv import load_dotenv

# ======== 🔧 Configuração ========
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# ======== 🧠 Função IA com fallback ========
def extract_text_with_ai(file_path):
    """
    Lê o PDF com IA — tenta Google Gemini, depois OpenAI, depois DeepSeek.
    """
    # ======== 1️⃣ Google Gemini ========
    try:
        if GOOGLE_API_KEY:
            import google.generativeai as genai
            genai.configure(api_key=GOOGLE_API_KEY)

            # Faz upload do PDF real
            uploaded_file = genai.upload_file(file_path)
            model = genai.GenerativeModel("gemini-1.5-flash-latest")

            prompt = (
                "Extraia todas as informações relevantes da nota fiscal (CNPJ, Razão Social, "
                "Número da Nota, Data, Valor Total e Itens). Resuma de forma estruturada."
            )

            response = model.generate_content([prompt, uploaded_file])
            return response.text.strip()

    except Exception as e:
        st.warning(f"⚠️ Falha no Google Gemini: {e}")

    # ======== 2️⃣ OpenAI GPT-4o ========
    try:
        if OPENAI_API_KEY:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)

            with open(file_path, "rb") as f:
                pdf_bytes = f.read()

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Você é um assistente que extrai dados de notas fiscais em PDF."},
                    {"role": "user", "content": "Extraia os dados principais do seguinte arquivo PDF."},
                ],
                files=[{"name": os.path.basename(file_path), "content": pdf_bytes}]
            )
            return response.choices[0].message.content.strip()

    except Exception as e:
        st.warning(f"⚠️ Falha no OpenAI: {e}")

    # ======== 3️⃣ DeepSeek ========
    try:
        if DEEPSEEK_API_KEY:
            import requests

            with open(file_path, "rb") as f:
                pdf_bytes = f.read()

            headers = {
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/pdf"
            }

            response = requests.post(
                "https://api.deepseek.com/v1/parse-pdf",
                headers=headers,
                data=pdf_bytes
            )

            if response.status_code == 200:
                return response.text.strip()
            else:
                raise Exception(f"Erro DeepSeek: {response.status_code} {response.text}")

    except Exception as e:
        st.warning(f"⚠️ Falha no DeepSeek: {e}")

    return "❌ Nenhum modelo conseguiu processar o PDF."


# ======== 💻 Interface ========
st.set_page_config(page_title="Automatizador de Notas", layout="wide")
st.title("📄 Automatizador de Notas com IA")
st.write("Envie PDFs de notas fiscais e deixe a IA extrair automaticamente as informações.")

uploaded_files = st.file_uploader("Selecione os arquivos PDF", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    st.info(f"{len(uploaded_files)} arquivo(s) enviado(s).")

    temp_dir = tempfile.mkdtemp()
    results = []

    progress = st.progress(0)
    total = len(uploaded_files)

    for i, uploaded_file in enumerate(uploaded_files, start=1):
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner(f"🔍 Processando {uploaded_file.name}..."):
            result = extract_text_with_ai(file_path)
            results.append({"arquivo": uploaded_file.name, "conteudo": result})

        progress.progress(i / total)

    st.success("✅ Processamento concluído!")

    for r in results:
        st.subheader(f"📄 {r['arquivo']}")
        st.text_area(f"Resultado_{r['arquivo']}", r["conteudo"], height=250, key=r["arquivo"])

    shutil.rmtree(temp_dir, ignore_errors=True)

else:
    st.info("Envie seus PDFs para começar.")

st.markdown("---")
st.markdown("**Desenvolvido por João Henrique 🚀** — Suporte a Google Gemini, OpenAI e DeepSeek")
