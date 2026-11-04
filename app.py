import os
import glob
import time
import base64
import streamlit as st
import chromadb
from pypdf import PdfReader
from google import genai
from google.genai import types

MODELS = ["gemini-flash-latest", "gemini-flash-lite-latest"]

SYSTEM_PROMPT = """You are a troubleshooting assistant for industrial machinery.
Rules:
1. Start with a one-line SAFETY note: power off, lockout/tagout, wear PPE.
2. Then answer using ONLY the manual excerpts provided. If any excerpt is relevant to the question, even partly, use it to give numbered, step-by-step checks.
3. Cite the source like (file name, page X) after each step.
4. Only if NONE of the excerpts relate to the question, reply with exactly: "I could not find this in the manuals. Please contact a senior technician." In that case give no safety note and no steps.
5. Never combine a refusal with troubleshooting steps.
6. Use the previous conversation only to understand follow-up questions.
"""
CONFIG = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.2)

EXAMPLES = [
    "Pump is not delivering water",
    "Pump is making a loud noise",
    "Motor is overheating",
]

st.set_page_config(page_title="Machine Troubleshooter", page_icon="🔧")

st.markdown(
    """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.block-container {max-width: 800px; padding-top: 2rem;}
section[data-testid="stSidebar"] .stButton button {
    width: 100%; text-align: left; justify-content: flex-start;
}
</style>
""",
    unsafe_allow_html=True,
)


def set_background(candidates=("background.jpg", "background.jpeg", "background.png")):
    """Uses a picture named background.jpg / .jpeg / .png if it sits next to app.py."""
    for path in candidates:
        if os.path.exists(path):
            mime = "image/png" if path.endswith(".png") else "image/jpeg"
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            st.markdown(
                f"""
<style>
.stApp {{
    background-image: linear-gradient(rgba(0,0,0,0.75), rgba(0,0,0,0.75)),
                      url("data:{mime};base64,{data}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}
header[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stBottom"] > div {{ background: transparent; }}
section[data-testid="stSidebar"] {{ background-color: rgba(0, 0, 0, 0.45); }}
</style>
""",
                unsafe_allow_html=True,
            )
            return


set_background()


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
    """Normal (non-streaming) call, used to describe images."""
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


def stream_answer(contents, config):
    """Streaming call: yields text pieces so the answer types out."""
    last_error = None
    for model in MODELS:
        for attempt in range(3):
            started = False
            try:
                for chunk in client.models.generate_content_stream(
                    model=model, contents=contents, config=config
                ):
                    if chunk.text:
                        started = True
                        yield chunk.text
                return
            except Exception as e:
                last_error = e
                if started:
                    raise
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


def recent_history(messages, n=4):
    if not messages:
        return ""
    lines = []
    for m in messages[-n:]:
        who = "Technician" if m["role"] == "user" else "Assistant"
        lines.append(f"{who}: {m['content'][:600]}")
    return "PREVIOUS CONVERSATION:\n" + "\n".join(lines) + "\n\n"


# ---------- chat state ----------
def new_chat():
    cid = str(int(time.time() * 1000))
    st.session_state.chats[cid] = {"title": "New chat", "messages": []}
    st.session_state.current = cid


if "chats" not in st.session_state:
    st.session_state.chats = {}
    st.session_state.current = None
    new_chat()

# ---------- sidebar ----------
with st.sidebar:
    st.markdown("## 🔧 Machine Troubleshooter")
    if st.button("➕ New chat"):
        new_chat()
        st.rerun()
    st.markdown("**Chats**")
    for cid, c in reversed(list(st.session_state.chats.items())):
        kind = "primary" if cid == st.session_state.current else "secondary"
        if st.button(c["title"][:28], key=f"chat_{cid}", type=kind):
            st.session_state.current = cid
            st.rerun()
    st.caption(
        "Demo project. Questions are sent to Google's Gemini API. "
        "Do not enter private information. Chats are cleared when you close or refresh the page."
    )

chat = st.session_state.chats[st.session_state.current]

# ---------- welcome screen ----------
if not chat["messages"]:
    st.markdown(
        "<h2 style='text-align:center; margin-top:3rem;'>How can I help with your machine today?</h2>",
        unsafe_allow_html=True,
    )
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex, key=f"ex_{ex}"):
            st.session_state.pending = ex
            st.rerun()

# ---------- show past messages ----------
for m in chat["messages"]:
    with st.chat_message(m["role"], avatar="🔧" if m["role"] == "assistant" else None):
        if m.get("image"):
            st.image(m["image"], width=250)
        st.markdown(m["content"])
        if m.get("context"):
            with st.expander("Sources from the manual"):
                st.text(m["context"])

# ---------- input ----------
prompt = st.chat_input(
    "Message the troubleshooter (use the + to attach a photo)",
    accept_file=True,
    file_type=["png", "jpg", "jpeg"],
)

question, image_bytes, mime = None, None, None
pending = st.session_state.pop("pending", None)
if pending:
    question = pending
elif prompt:
    question = (prompt.text or "").strip()
    if prompt.files:
        f = prompt.files[0]
        image_bytes, mime = f.getvalue(), f.type
    if not question and image_bytes:
        question = "What could be wrong with this machine?"

# ---------- answer ----------
if question:
    is_first = not chat["messages"]
    if is_first:
        chat["title"] = question[:40]
    history = recent_history(chat["messages"])
    chat["messages"].append({"role": "user", "content": question, "image": image_bytes})

    with st.chat_message("user"):
        if image_bytes:
            st.image(image_bytes, width=250)
        st.markdown(question)

    with st.chat_message("assistant", avatar="🔧"):
        try:
            with st.spinner("Searching manuals..."):
                search_text = question
                parts = []
                if image_bytes:
                    img = types.Part.from_bytes(data=image_bytes, mime_type=mime)
                    parts.append(img)
                    desc = ask([
                        img,
                        "In 1-2 sentences, describe the machine part, any visible damage, and read any error code or text shown.",
                    ])
                    search_text = question + " " + desc
                context = retrieve(search_text)
                parts.append(
                    f"{history}MANUAL EXCERPTS:\n{context}\n\nTECHNICIAN QUESTION: {question}"
                )

            reply = st.write_stream(stream_answer(parts, CONFIG))
            with st.expander("Sources from the manual"):
                st.text(context)
            chat["messages"].append(
                {"role": "assistant", "content": reply, "context": context}
            )
        except Exception:
            chat["messages"].pop()
            st.error("Sorry, the AI service is busy or unavailable. Please wait a minute and try again.")

    if is_first and chat["messages"]:
        st.rerun()
