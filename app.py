"""ROLE C - Streamlit UI. Run: streamlit run app.py"""
import json, os, random, re
import streamlit as st
from dotenv import load_dotenv

from src.ingest import load_documents, load_uploaded, chunk_pages
from src.retriever import Retriever
from src.llm import chat
from src.prompts import build_messages, quiz_messages, MODE_HINTS, NOT_FOUND
from src.guard import check_grounding
from src.ui_helpers import CSS, terms, annotate, source_card

load_dotenv()
HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
st.set_page_config(page_title="Saathi", page_icon="📚", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)

# ---------------- sidebar ----------------
with st.sidebar:
    st.subheader("Model")
    llm_model = st.text_input("Llama model (Ollama)", os.getenv("LLM_MODEL", "llama3.2:3b"))
    mode = st.selectbox("Answer style", list(MODE_HINTS))
    st.subheader("Your notes")
    use_rag = st.toggle("Answer from my notes", True,
                        help="Turn off to see what Llama says without your notes.")
    compare = st.toggle("Compare with Llama alone", False, disabled=not use_rag,
                        help="Shows both answers side by side.")
    backend = st.selectbox("Search method", ["tfidf (no model)", "ollama embeddings"])
    k = st.slider("Passages to retrieve", 1, 8, 4)
    size = st.slider("Passage length (words)", 60, 300, 120, 10)
    overlap = st.slider("Overlap (words)", 0, 80, 30, 5)
    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.update(log=[], history=[])
    st.markdown('<span class="ok-mark">Green underline</span>: backed by your notes<br>'
                '<span class="bad-mark">Red wavy</span>: not found in your notes<br>'
                '<mark>Yellow</mark>: words from your question', unsafe_allow_html=True)
    st.caption("Everything runs on this computer. Nothing is sent online.")

# ---------------- hero ----------------
st.markdown(f"""<div class="hero"><h1>Saathi</h1>
<p>Ask questions about your own notes. Every answer shows where it came from, and says so when your notes don't have it.</p>
<span class="pill">Runs offline</span><span class="pill">{llm_model}</span></div>""", unsafe_allow_html=True)

# ---------------- notes ----------------
indexed = st.session_state.get("sig") is not None
with st.expander("Your notes" if indexed else "Start here: add your notes", expanded=not indexed):
    files = st.file_uploader("Drop PDF, TXT or MD files (several are fine)",
                             type=["pdf", "txt", "md"], accept_multiple_files=True)
    if st.button("Use the sample notes instead"):
        st.session_state["use_sample"] = True
if files:
    st.session_state["use_sample"] = False

source = "upload" if files else ("sample" if st.session_state.get("use_sample") else None)
if source is None:
    st.info("Add your notes above to begin. Saathi reads them and answers only from them.")
    st.stop()

sig = (source, tuple((f.name, f.size) for f in files or []), backend, size, overlap)
if st.session_state.get("sig") != sig:
    with st.spinner("Reading your notes..."):
        pages = load_uploaded(files) if source == "upload" else load_documents("data")
        if not pages:
            st.session_state["sig"] = None
            st.error("No text could be read from these files. Scanned PDFs need OCR first, "
                     "or paste the text into a .txt file.")
            st.stop()
        chunks = chunk_pages(pages, size, overlap)
        try:
            r = Retriever("tfidf" if backend.startswith("tfidf") else "ollama",
                          os.getenv("EMBED_MODEL", "nomic-embed-text"), HOST).build(chunks)
        except Exception as e:
            st.error(f"Could not build the search index: {e}. Try the tfidf search method.")
            st.stop()
        st.session_state.update(retriever=r, chunks=chunks, history=[], log=[], quiz=[], sig=sig,
                                n_pages=len(pages), names=sorted({p["source"] for p in pages}),
                                quiz_id=0)
        st.rerun()

st.markdown(f'<span class="pill ok">Ready</span><span class="pill">{len(st.session_state["names"])} file(s)</span>'
            f'<span class="pill">{st.session_state["n_pages"]} page(s)</span>'
            f'<span class="pill">{len(st.session_state["chunks"])} passages</span>'
            f'<br><small style="color:#5B6678">{", ".join(st.session_state["names"])}</small>',
            unsafe_allow_html=True)


# ---------------- rendering ----------------
def rag_block(e, open_sources):
    guard = check_grounding(e["ans"], e["hits"])
    st.markdown(annotate(e["ans"], guard), unsafe_allow_html=True)
    bad = sum(not g["supported"] for g in guard)
    if NOT_FOUND[:20] in e["ans"]:
        st.markdown('<span class="pill warn">Not in your notes</span>', unsafe_allow_html=True)
    elif guard and bad == 0:
        st.markdown(f'<span class="pill ok">All {len(guard)} sentence(s) backed by your notes</span>',
                    unsafe_allow_html=True)
    elif bad:
        st.markdown(f'<span class="pill bad">{bad} of {len(guard)} sentence(s) not backed by your notes</span>',
                    unsafe_allow_html=True)
    with st.expander(f"Sources ({len(e['hits'])} passages)", expanded=open_sources):
        top, qt = max((h["score"] for h in e["hits"]), default=0), terms(e["q"])
        st.markdown("".join(source_card(i, h, top, qt) for i, h in enumerate(e["hits"], 1)),
                    unsafe_allow_html=True)


def plain_block(text):
    st.markdown(f'<div class="answer plain">{text}</div>', unsafe_allow_html=True)
    st.markdown('<span class="pill warn">Not checked against your notes</span>', unsafe_allow_html=True)


def render_entry(e, open_sources=False):
    if e["base"] is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**With your notes**")
            rag_block(e, open_sources)
        with c2:
            st.markdown("**Llama alone**")
            plain_block(e["base"])
    elif e["hits"] is not None:
        rag_block(e, open_sources)
    else:
        plain_block(e["ans"])


tab_chat, tab_quiz = st.tabs(["Ask", "Quiz me"])

# ---------------- chat ----------------
with tab_chat:
    log = st.session_state["log"]
    for i, e in enumerate(log):
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.write(e["q"])
        with st.chat_message("assistant", avatar="📚"):
            render_entry(e, open_sources=(i == len(log) - 1))
    if not log:
        st.caption("Try one of these, or type your own question below.")
        starters = ["What are the main topics in these notes?",
                    "List the key definitions I should remember.",
                    "What is the capital of Australia?"]
        cols = st.columns(3)
        for c, s, lab in zip(cols, starters, ["Main topics", "Key definitions", "Test: not in my notes"]):
            if c.button(lab, use_container_width=True, help=s):
                st.session_state["pending"] = s
    q = st.chat_input("Ask a question about your notes")
    if not q:
        q = st.session_state.pop("pending", None)
    if q:
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.write(q)
        with st.chat_message("assistant", avatar="📚"):
            with st.spinner("Llama is reading your notes..." if use_rag else "Llama is thinking..."):
                hits = st.session_state["retriever"].search(q, k) if use_rag else None
                ans = chat(build_messages(q, hits, mode, st.session_state["history"]), llm_model, HOST)
                base = chat(build_messages(q, None), llm_model, HOST) if (use_rag and compare) else None
            entry = {"q": q, "ans": ans, "hits": hits, "base": base}
            render_entry(entry, open_sources=True)
        log.append(entry)
        st.session_state["history"] += [{"role": "user", "content": q},
                                        {"role": "assistant", "content": ans}]

# ---------------- quiz ----------------
with tab_quiz:
    n = st.slider("Questions", 1, 5, 3)
    if st.button("Make a quiz from my notes", type="primary"):
        sample = random.sample(st.session_state["chunks"], min(3, len(st.session_state["chunks"])))
        with st.spinner("Writing questions..."):
            raw = chat(quiz_messages(sample, n), llm_model, HOST, temperature=0.4)
        try:
            st.session_state["quiz"] = json.loads(re.search(r"\[.*\]", raw, re.S).group(0))
            st.session_state["quiz_id"] += 1
        except Exception:
            st.session_state["quiz"] = []
            st.error("Llama's reply wasn't a valid quiz. Click again; small models sometimes slip.")
    score = answered = 0
    for i, item in enumerate(st.session_state.get("quiz", [])):
        with st.container(border=True):
            st.markdown(f"**{i + 1}. {item['q']}**")
            pick = st.radio("Answer", item["options"], key=f"q{st.session_state['quiz_id']}_{i}",
                            index=None, label_visibility="collapsed")
            if pick is not None:
                answered += 1
                ok = item["options"].index(pick) == item["answer_index"]
                score += ok
                (st.success if ok else st.error)(("Correct. " if ok else "Not quite. ") + item.get("why", ""))
    if answered:
        st.markdown(f'<span class="pill">Score: {score} of {answered} answered</span>', unsafe_allow_html=True)
