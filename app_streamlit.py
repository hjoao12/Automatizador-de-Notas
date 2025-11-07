if uploaded_files and st.button("🚀 Processar PDFs"):
    session_id = str(uuid.uuid4())
    session_folder = TEMP_FOLDER / session_id
    os.makedirs(session_folder, exist_ok=True)

    resultados = []
    start_global = time.time()

    prompt = (
        "Analise a nota fiscal. Extraia emitente, número da nota e cidade. "
        "Responda SOMENTE em JSON: {\"emitente\":\"NOME\",\"numero_nota\":\"NUMERO\",\"cidade\":\"CIDADE\"}"
    )

    # 🔧 Calcula total de páginas antes
    total_paginas = 0
    for f in uploaded_files:
        try:
            leitor = PdfReader(io.BytesIO(f.read()))
            total_paginas += len(leitor.pages)
            f.seek(0)
        except:
            pass

    progress_bar = st.progress(0)
    progresso = 0
    progresso_texto = st.empty()

    st.info(f"📄 Total de páginas a processar: {total_paginas}")

    for file_index, file in enumerate(uploaded_files):
        file_name = file.name
        pdf_bytes = io.BytesIO(file.read())

        try:
            leitor = PdfReader(pdf_bytes)
        except Exception as e:
            st.error(f"Erro ao ler {file_name}: {e}")
            continue

        for i, page in enumerate(leitor.pages):
            start_page_time = time.time()
            page_stream = io.BytesIO()
            writer = PdfWriter()
            writer.add_page(page)
            writer.write(page_stream)
            page_stream.seek(0)

            dados, ok, tempo = chamar_gemini_retry(model, prompt, page_stream)
            if ok and "error" not in dados:
                emitente = dados.get("emitente", "")
                numero = dados.get("numero_nota", "")
                cidade = dados.get("cidade", "")
                numero_limpo = limpar_numero(numero)
                nome_map = substituir_nome_emitente(emitente, cidade)
                emitente_limpo = limpar_emitente(nome_map)
                novo_nome = f"DOC {numero_limpo}_{emitente_limpo}.pdf"
                destino = session_folder / novo_nome
                with open(destino, "wb") as f_out:
                    f_out.write(page_stream.read())
                status_msg = "✅ Sucesso"
            else:
                status_msg = f"❌ {dados.get('error', 'Erro desconhecido')}"
                novo_nome = "-"

            tempo_pagina = round(time.time() - start_page_time, 2)
            progresso += 1
            progresso_texto.markdown(
                f"⏱ **Página {progresso}/{total_paginas} — {file_name} ({i+1}/{len(leitor.pages)})** → {status_msg} ({tempo_pagina}s)"
            )
            progress_bar.progress(min(progresso / total_paginas, 1.0))

            resultados.append({
                "original": file_name,
                "novo": novo_nome,
                "tempo": tempo_pagina,
                "status": status_msg
            })

    tempo_total = round(time.time() - start_global, 2)
    st.success(f"🏁 Processamento concluído em {tempo_total}s ({len(resultados)} páginas).")

    # 📦 Download final
    memory_zip = io.BytesIO()
    with zipfile.ZipFile(memory_zip, 'w') as zf:
        for f in os.listdir(session_folder):
            zf.write(session_folder / f, arcname=f)
    memory_zip.seek(0)
    st.download_button(
        "⬇️ Baixar arquivos processados",
        data=memory_zip,
        file_name="notas_processadas.zip",
        mime="application/zip"
    )

    # 📋 Exibe resultados na tela
    st.subheader("📜 Resultados detalhados")
    st.dataframe(resultados)
