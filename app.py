import os
import glob
import time
import streamlit as st
import chromadb
from pypdf import PdfReader
from google import genai
from google.genai import types

MODELS = ["gemini-flash-latest", "gemini-flash-lite-latest"]

SYSTEM_PROMPT = """You are a troubleshooting assistant for industrial machinery.
Rules:
1. Answer ONLY using the manual excerpts provided.
2. Give numbered, step-by-step instructions.
3. Cite the source like (file name, page X) after each step.
4. Start with a SAFETY note: power off, lockout/tagout, wear PPE.
5. If the excerpts do not contain the answer, say: "I could not find this in the manuals. Please contact a senior technician."
"""

st.set_page_config(page_title="Machine Troubleshooter", page_icon="🔧")
st.title("🔧 Industrial Machinery Troubleshooting Assistant")
st.caption("Demo project. Questions are sent to Google's Gemini API. Do not enter private information.")


def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.environ.get("GEMINI_API_KEY")


API_KEY = get_api_key()
if not API_KEY:
    st.error("API key is missing. Add GEMINI_API_KEY in the app secrets.")
    st.stop()

client = genai.Client(api_key=API_KEY)


def chunk_text(text, size=800, overlap=150):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


@st.cache_resource(show_spinner="Preparing the manual database (first start only)...")
def load_collection():
    db = chromadb.PersistentClient(path="db")
    collection = db.get_or_create_collection("manuals")
    if collection.count() == 0:
        # Finds PDFs inside the manuals folder OR next to app.py
        for pdf_path in glob.glob("manuals/*.pdf") + glob.glob("*.pdf"):
            name = os.path.basename(pdf_path)
            reader = PdfReader(pdf_path)
            ids, docs, metas = [], [], []
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                for i, chunk in enumerate(chunk_text(text)):
                    if len(chunk.strip()) < 50:
                        continue
                    ids.append(f"{name}-p{page_num}-c{i}")
                    docs.append(chunk)
                    metas.append({"source": name, "page": page_num})
            for s in range(0, len(ids), 100):
                collection.add(
                    ids=ids[s:s + 100],
                    documents=docs[s:s + 100],
                    metadatas=metas[s:s + 100],
                )
    return collection


collection = load_collection()


def ask(contents, config=None):
    last_error = None
    for model in MODELS:
        for attempt in range(3):
            try:
                return client.models.generate_content(
                    model=model, contents=contents, config=config
                ).text
            except Exception as e:
                last_error = e
                text = str(e)
                if "503" in text or "429" in text or "UNAVAILABLE" in text:
                    time.sleep(3 * (attempt + 1))
                else:
                    raise
    raise last_error


def retrieve(query, k=5):
    result = collection.query(query_texts=[query], n_results=k)
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    return "\n\n".join(
        f"[{m['source']}, page {m['page']}]\n{d}" for d, m in zip(docs, metas)
    )


def image_part(uploaded):
    return types.Part.from_bytes(data=uploaded.getvalue(), mime_type=uploaded.type)


uploaded = st.file_uploader(
    "Optional: upload a photo of the machine or error screen",
    type=["png", "jpg", "jpeg"],
)

if "history" not in st.session_state:
    st.session_state.history = []

for role, text in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)

question = st.chat_input("Describe the problem (e.g. Pump not delivering water, error E14)")

if question:
    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching manuals..."):
                search_text = question
                parts = []

                if uploaded is not None:
                    parts.append(image_part(uploaded))
                    desc = ask([
                        image_part(uploaded),
                        "In 1-2 sentences, describe the machine part, any visible damage, and read any error code or text shown.",
                    ])
                    search_text = question + " " + desc

                context = retrieve(search_text)
                parts.append(
                    f"MANUAL EXCERPTS:\n{context}\n\nTECHNICIAN QUESTION: {question}"
                )

                reply = ask(
                    parts,
                    config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
                )

            st.markdown(reply)
            with st.expander("Retrieved manual excerpts"):
                st.text(context)
            st.session_state.history.append(("assistant", reply))

        except Exception:
            st.error("Sorry, the AI service is busy or unavailable. Please wait a minute and try again.")