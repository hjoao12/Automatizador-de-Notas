import os
import streamlit as st
from dotenv import load_dotenv
import tempfile
import shutil
from pathlib import Path
import requests

# ========== 🔧 Carregar variáveis de ambiente ==========
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# ========== 🧠 IA com fallback ==========
def extract_text_with_ai(file_path):
    """
    Extrai texto de PDF usando IA com fallback automático:
      1️⃣ Google Gemini
      2️⃣ OpenAI (ChatGPT)
      3️⃣ DeepSeek
    """
    # --- 1️⃣ Tenta Google Gemini ---
    if GOOGLE_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GOOGLE_API_KEY)
            model = genai.GenerativeModel("gemini-2.5-flash")
            prompt = f"Extraia o texto completo e os principais dados do PDF: {file_path}"
            response = model.generate_content(prompt)
            if response.text:
                return response.text
        except Exception as e:
            st.warning(f"⚠️ Falha no Google Gemini: {e}")

    # --- 2️⃣ Tenta OpenAI ---
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Você é um assistente que extrai texto e dados de PDFs."},
                    {"role": "user", "content": f"Extraia o texto e as informações principais do arquivo {file_path}."}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            st.warning(f"⚠️ Falha no OpenAI: {e}")

    # --- 3️⃣ Tenta DeepSeek ---
    if DEEPSEEK_API_KEY:
        try:
            url = "https://api.deepseek.com/chat/completions"
            headers = {
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "Você é um assistente que lê PDFs e extrai informações relevantes."},
                    {"role": "user", "content": f"Extraia o texto e as informações do arquivo {file_path}."}
                ]
            }
            response = requests.post(url, headers=headers, json=payload)
            data = response.json()
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            st.warning(f"⚠️ Falha no DeepSeek: {e}")

    # --- Nenhum serviço funcionou ---
    return "❌ Nenhum serviço de IA pôde processar o arquivo."


# ========== 🖥️ Interface Streamlit ==========
st.set_page_config(page_title="Automatizador de Notas com IA", layout="wide")
st.title("📄 Automatizador de Notas com IA (Google + OpenAI + DeepSeek)")
st.write("Envie seus PDFs de notas fiscais e extraia automaticamente as informações usando IA com fallback inteligente.")

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
            results.append({"arquivo": uploaded_file.name, "conteudo": extracted_text})

        progress.progress(i / total)

    st.success("✅ Processamento concluído!")

    # Mostrar resultados com IDs únicos
    for idx, r in enumerate(results):
        st.subheader(r["arquivo"])
        st.text_area("Resultado", r["conteudo"], height=200, key=f"resultado_{idx}")

    shutil.rmtree(temp_dir, ignore_errors=True)
else:
    st.info("Envie seus PDFs para começar.")

st.markdown("---")
st.markdown("**Desenvolvido por João Henrique** 🚀 | IA com fallback: Google → OpenAI → DeepSeek")
