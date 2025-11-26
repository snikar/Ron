import streamlit as st
from datetime import datetime

from core.config import (
    APP_TITLE,
    OPENAI_MODELS,
    GEMINI_MODELS,
    LOCAL_MODELS,
)
from core.routing import Router

from memory.memory_manager import MemoryManager
from memory.parsers import ParserEngine
from importers.chatgpt_html_reader import ChatGPTHTMLReader
from importers.html_chunker import HTMLChunker

from models.openai_brain import OpenAIBrain
from models.gemini_brain import GeminiBrain
from models.local_brain import LocalBrain


# ---------------------------------------------------------
#                STREAMLIT INITIALIZATION
# ---------------------------------------------------------

st.set_page_config(page_title=APP_TITLE, page_icon="🤖", layout="wide")
st.title(APP_TITLE)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "memory_manager" not in st.session_state:
    st.session_state["memory_manager"] = MemoryManager()

if "write_memory" not in st.session_state:
    st.session_state["write_memory"] = True


memory = st.session_state["memory_manager"]
parser_engine = ParserEngine()
html_reader = ChatGPTHTMLReader()
html_chunker = HTMLChunker()


# ---------------------------------------------------------
#                    SIDEBAR CONTROLS
# ---------------------------------------------------------

st.sidebar.header("⚙️ Settings")

all_models = OPENAI_MODELS + GEMINI_MODELS + LOCAL_MODELS

selected_model = st.sidebar.selectbox(
    "Select Model",
    options=all_models,
    index=0
)

write_toggle = st.sidebar.checkbox(
    "Write to Memory",
    value=st.session_state["write_memory"],
)

st.session_state["write_memory"] = write_toggle


# ---------------------------------------------------------
#             IMPORT CHATGPT EXPORT (chat.html)
# ---------------------------------------------------------

st.sidebar.subheader("📥 Import ChatGPT Export")

uploaded_html = st.sidebar.file_uploader(
    "Upload chat.html",
    type=["html"],
    key="html_upload",
)

if uploaded_html:
    with st.sidebar:
        st.write("Processing ChatGPT export...")

    try:
        blocks = html_reader.read_html(uploaded_html.read())
        chunks = html_chunker.chunk_blocks(blocks)

        for c in chunks:
            memory.add_memory_entry(
                text=c,
                source="chatgpt_export",
                write=write_toggle
            )

        st.sidebar.success(f"Imported {len(chunks)} chunks from ChatGPT export.")

    except Exception as e:
        st.sidebar.error(f"Error importing HTML: {e}")


# ---------------------------------------------------------
#        FILE UPLOAD (PDF, Image, DOCX, Excel, etc.)
# ---------------------------------------------------------

st.sidebar.subheader("📄 Upload File for Ingestion")

uploaded_file = st.sidebar.file_uploader(
    "Upload Document",
    type=["pdf", "png", "jpg", "jpeg", "docx", "txt", "md", "csv", "xlsx"],
)

if uploaded_file:
    try:
        raw_text = parser_engine.parse_file(uploaded_file.read(), uploaded_file.name)

        # Chunk + embed + store
        chunks = memory.chunker.chunk_text(raw_text)

        for c in chunks:
            memory.add_memory_entry(
                text=c,
                source=f"file:{uploaded_file.name}",
                write=write_toggle
            )

        st.sidebar.success(f"Ingested {len(chunks)} text chunks.")

    except Exception as e:
        st.sidebar.error(f"File ingestion error: {e}")


# ---------------------------------------------------------
#                 ROUTER (MODEL SELECTION)
# ---------------------------------------------------------

router = Router(memory_manager=memory, write_memory=write_toggle)
brain = router.get_brain(selected_model)


# ---------------------------------------------------------
#                     CHAT INTERFACE
# ---------------------------------------------------------

st.subheader("💬 Jeff Chat")

# Display chat history
for role, content in st.session_state["chat_history"]:
    if role == "user":
        st.markdown(f"**You:** {content}")
    else:
        st.markdown(f"**Jeff ({selected_model}):** {content}")

# Input box
user_message = st.text_input("Type your message:", "")

if st.button("Send") and user_message.strip():
    # Log user message
    st.session_state["chat_history"].append(("user", user_message))

    try:
        reply = brain.generate_reply(user_message)
    except Exception as e:
        reply = f"[ERROR] {e}"

    # Log reply
    st.session_state["chat_history"].append(("assistant", reply))

    # Write memory if enabled
    if write_toggle:
        memory.add_memory_entry(
            text=user_message,
            source="user_message",
            write=True,
        )
        memory.add_memory_entry(
            text=reply,
            source="assistant_reply",
            write=True,
        )

    st.experimental_rerun()
