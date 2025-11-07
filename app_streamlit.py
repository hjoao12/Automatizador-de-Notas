import os
import streamlit as st
import tempfile
import shutil
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# ======== 🧩 Função de fallback IA ========
def extract_text_with_ai(file_path):
    """
    Extrai dados da nota fiscal usando fallback entre:
    1. Google Gemini
    2. OpenAI GPT-4o
    3. DeepSeek
    """
    # 1️⃣ Google Gemini
    try:
        if GOOGLE_API_KEY:
            import google.generativeai as genai
            genai.configure(api_key=GOOGLE_API_KEY)

            uploaded_file = genai.upload_file(file_path)
            model = genai.GenerativeModel("gemini-2.5-flash")

            prompt = (
                "Extraia os principais dados da nota fiscal (número, empresa, data, valor, CNPJ, itens). "
                "Resuma de forma estruturada e clara."
            )

            response = model.generate_content([prompt, uploaded_file])
            return response.text.strip()
    except Exception as e:
        st.warning(f"⚠️ Falha no Google Gemini: {e}")

    # 2️⃣ OpenAI GPT-4o (usando Responses API — suporta arquivos)
    try:
        if OPENAI_API_KEY:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)

            file_obj = client.files.create(file=open(file_path, "rb"), purpose="assistants")

            response = client.responses.create(
                model="gpt-4o-mini",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Extraia os principais dados desta nota fiscal em PDF:"},
                            {"type": "input_file", "file_id": file_obj.id}
                        ]
                    }
                ]
            )
            return response.output_text
    except Exception as e:
        st.warning(f"⚠️ Falha no OpenAI: {e}")

    # 3️⃣ DeepSeek — fallback de texto via PyMuPDF
    try:
        if DEEPSEEK_API_KEY:
            import fitz  # PyMuPDF
            import requests

            text = ""
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()

            if not text.strip():
                raise Exception("PDF vazio ou ilegível.")

            headers = {
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }

            data = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "Você extrai dados de notas fiscais."},
                    {"role": "user", "content": f"Extraia e formate os dados principais:\n\n{text[:8000]}"}
                ]
            }

            r = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=data)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
            else:
                raise Exception(f"Erro DeepSeek: {r.status_code} {r.text}")

    except Exception as e:
        st.warning(f"⚠️ Falha no DeepSeek: {e}")

    return "❌ Nenhum modelo conseguiu processar o PDF."


# ======== 🖥️ Interface Streamlit ========
st.set_page_config(page_title="Automatizador de Notas com IA", layout="wide")
st.title("📄 Automatizador de Notas com IA")
st.write("Envie PDFs de notas fiscais e extraia automaticamente as informações com fallback inteligente (Gemini → OpenAI → DeepSeek).")

uploaded_files = st.file_uploader("Selecione os arquivos PDF", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    st.info(f"{len(uploaded_files)} arquivo(s) enviado(s).")
    temp_dir = tempfile.mkdtemp()
    results = []
    progress = st.progress(0)

    for i, uploaded_file in enumerate(uploaded_files, start=1):
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner(f"🔍 Processando {uploaded_file.name}..."):
            result = extract_text_with_ai(file_path)
            results.append({"arquivo": uploaded_file.name, "conteudo": result})

        progress.progress(i / len(uploaded_files))

    st.success("✅ Processamento concluído!")

    for r in results:
        st.subheader(r["arquivo"])
        st.text_area(f"Resultado_{r['arquivo']}", r["conteudo"], height=250, key=r["arquivo"])

    shutil.rmtree(temp_dir, ignore_errors=True)

else:
    st.info("Envie seus PDFs para começar.")

st.markdown("---")
st.markdown("**Desenvolvido por João Henrique 🚀** — com suporte a Google Gemini, OpenAI e DeepSeek")
