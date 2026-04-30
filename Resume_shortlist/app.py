import streamlit as st
import os
import chromadb
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

# ================= LOCAL EMBEDDING =================
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ================= CHROMA DB =================
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("resumes")

# ================= PDF LOADER =================
def load_pdf_resume(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    text = " ".join(page.page_content for page in docs)

    metadata = {
        "source": os.path.basename(pdf_path),
        "pages": len(docs),
        "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return text, metadata

# ================= UI =================
st.set_page_config(page_title="📄 Resume RAG", layout="wide")
st.title("📄 Resume Shortlisting using RAG")

# ================= UPLOAD =================
st.header("1️⃣ Upload Resume PDFs")

files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

if files:
    for file in files:
        temp_path = os.path.join("temp", file.name)
        os.makedirs("temp", exist_ok=True)

        with open(temp_path, "wb") as f:
            f.write(file.read())

        text, meta = load_pdf_resume(temp_path)

        with st.spinner("🔄 Generating embedding..."):
            embedding = embed_model.embed_documents([text])[0]

        collection.add(
            documents=[text],
            metadatas=[meta],
            embeddings=[embedding],
            ids=[file.name]
        )

        os.remove(temp_path)

    st.success("✅ Resumes uploaded and indexed")

# ================= JOB DESCRIPTION =================
st.header("2️⃣ Job Description")

jd = st.text_area("Paste job description", height=180)
top_n = st.slider("Top N resumes", 1, 10, 3)

if st.button("🔍 Shortlist") and jd:
    with st.spinner("🔍 Finding best matches..."):
        jd_embedding = embed_model.embed_query(jd)

        results = collection.query(
            query_embeddings=[jd_embedding],
            n_results=top_n
        )

    st.subheader("📊 Shortlisted Resumes")

    for i, (doc, meta) in enumerate(
        zip(results["documents"][0], results["metadatas"][0]), start=1
    ):
        st.markdown(f"### #{i} {meta['source']}")
        st.caption(f"Pages: {meta['pages']} | Uploaded: {meta['uploaded_at']}")
        st.write(doc[:500] + " ...")

# ================= LIST =================
st.header("3️⃣ List All Resumes")

data = collection.get()

if not data["ids"]:
    st.info("No resumes available")
else:
    # Create mapping for resume content
    resume_map = {
        meta["source"]: doc
        for doc, meta in zip(data["documents"], data["metadatas"])
    }

    selected_resume = st.radio(
        "📌 Select a resume",
        list(resume_map.keys())
    )

    st.success(f"📄 Selected Resume: {selected_resume}")

    # ✅ SHOW FULL CONTENT HERE
    st.text_area(
        "📄 Resume Content",
        resume_map[selected_resume],
        height=400
    )

# ================= DELETE =================
st.header("4️⃣ Delete Resume")

if data["ids"]:
    to_delete = st.selectbox("Select resume to delete", data["ids"])

    if st.button("🗑️ Delete Selected Resume"):
        collection.delete(ids=[to_delete])
        st.success(f"✅ {to_delete} deleted successfully")
        st.rerun()