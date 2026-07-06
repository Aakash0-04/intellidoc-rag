"""Professional Streamlit chat interface."""
from __future__ import annotations

import uuid

import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="RAG Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #f0f2f6; }
    .sub-header { font-size: 0.9rem; color: #9ca3af; margin-top: -0.5rem; }
    .chat-user { background: #1f2937; border-radius: 12px; padding: 12px 16px; margin: 8px 0; border-left: 3px solid #ef4444; }
    .chat-assistant { background: #111827; border-radius: 12px; padding: 12px 16px; margin: 8px 0; border-left: 3px solid #10b981; }
    .source-box { background: #1f2937; border-radius: 8px; padding: 10px 14px; margin-top: 8px; font-size: 0.8rem; color: #9ca3af; }
    .stButton>button { border-radius: 8px; font-weight: 500; }
    .upload-area { background: #1f2937; border-radius: 12px; padding: 16px; margin-bottom: 16px; }
    .sidebar-title { font-size: 1.1rem; font-weight: 600; color: #f0f2f6; margin-bottom: 12px; }
    div[data-testid="stChatMessage"] { padding: 0; }
</style>
""", unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">📁 Documents</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload PDF or DOCX",
            type=["pdf", "docx"],
            label_visibility="collapsed",
        )
        
        if uploaded and st.button("🚀 Process Document", type="primary", use_container_width=True):
            with st.spinner("Extracting & indexing..."):
                files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                try:
                    resp = requests.post(f"{API_BASE}/upload", files=files, timeout=180)
                    resp.raise_for_status()
                    data = resp.json()
                    st.session_state.uploaded_files.append(data["filename"])
                    st.success(f"✅ {data['chunks_indexed']} chunks indexed")
                except Exception as e:
                    st.error(f"Failed: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    if st.session_state.uploaded_files:
        st.markdown("**Indexed Files:**")
        for f in st.session_state.uploaded_files:
            st.markdown(f"• {f}")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            requests.post(f"{API_BASE}/clear", data={"session_id": st.session_state.session_id})
            st.session_state.messages = []
            st.rerun()
    with col2:
        st.caption(f"Session: `{st.session_state.session_id}`")

# ── Main Chat Area ─────────────────────────────────────────
st.markdown('<div class="main-header">📚 Document RAG Chat</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload documents and ask questions with cited sources</div>', unsafe_allow_html=True)
st.divider()

# Display messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📄 Sources"):
                    for src in msg["sources"]:
                        page = src.get("page") or "N/A"
                        st.markdown(
                            f'<div class="source-box">'
                            f'📄 <b>{src["source"]}</b> &nbsp;|&nbsp; '
                            f'📄 Page {page} &nbsp;|&nbsp; '
                            f'Type: {src.get("type", "text")}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

# Chat input
if prompt := st.chat_input("Ask about your documents...", key="chat_input"):
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Retrieving & generating..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/chat",
                    data={"message": prompt, "session_id": st.session_state.session_id},
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                
                st.markdown(data["answer"])
                
                if data.get("sources"):
                    with st.expander("📄 Sources"):
                        for src in data["sources"]:
                            page = src.get("page") or "N/A"
                            st.markdown(
                                f'<div class="source-box">'
                                f'📄 <b>{src["source"]}</b> &nbsp;|&nbsp; '
                                f'📄 Page {page} &nbsp;|&nbsp; '
                                f'Type: {src.get("type", "text")}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["answer"],
                    "sources": data.get("sources", []),
                })
                
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Sorry, an error occurred: {e}",
                    "sources": [],
                })