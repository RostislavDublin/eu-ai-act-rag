"""
EU AI Act RAG Assistant - Streamlit application.

Tabs:
  Chat             - ask questions, see answers with source chunks
  Corpus           - view loaded documents, upload new ones (authenticated only)
  Evaluation       - run RAG Triad eval dataset
  Quality Dashboard - radar + bar charts from last evaluation run

Access control:
  APP_PASSWORD gate  - basic protection for the whole app (optional)
  Role toggle button - inside the app, in the sidebar:
      Public User        -> clearance = "public"    (EU AI Act only)
      Authenticated User -> clearance = "restricted" (EU AI Act + internal docs)
  Document upload is available to Authenticated User only.
  DIAL_TOKEN is only used for API calls, not for determining clearance.
"""
import io
import json
import os
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EU AI Act RAG Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Credential helpers: Streamlit Secrets (Cloud) or .env (local)
# ---------------------------------------------------------------------------

def _secret(key: str, default: str = "") -> str:
    try:
        val = st.secrets.get(key)
        if val:
            return str(val)
    except Exception:
        pass
    return os.environ.get(key, default)


def _load_env_once() -> None:
    """Load .env file for local development (no-op on Streamlit Cloud)."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path, override=False)
        except ImportError:
            pass


_load_env_once()

# ---------------------------------------------------------------------------
# APP_PASSWORD gate (optional - protects against quota exhaustion)
# ---------------------------------------------------------------------------

APP_PASSWORD = _secret("APP_PASSWORD")

if APP_PASSWORD:
    if "app_authed" not in st.session_state:
        st.session_state["app_authed"] = False

    if not st.session_state["app_authed"]:
        st.title("EU AI Act RAG Assistant")

        col_login, col_checklist = st.columns([3, 2])

        with col_login:
            # --- API / model info panel ---
            _openai_key = _secret("OPENAI_API_KEY")
            if _openai_key:
                _provider_label = "OpenAI API"
                _embed_model = _secret("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
                _chat_model = _secret("OPENAI_CHAT_MODEL", "gpt-4o")
            else:
                _provider_label = "EPAM DIAL"
                _embed_model = _secret("DIAL_EMBEDDING_MODEL", "text-embedding-005")
                _chat_model = _secret("DIAL_CHAT_MODEL", "gpt-4o")

            with st.container(border=True):
                st.markdown(f"**Provider:** {_provider_label}")
                st.markdown(f"**Embeddings:** `{_embed_model}`")
                st.markdown(f"**Generation / Judging:** `{_chat_model}`")

            pwd = st.text_input("Enter access password:", type="password")
            if st.button("Enter"):
                if pwd == APP_PASSWORD:
                    st.session_state["app_authed"] = True
                    st.rerun()
                else:
                    st.error("Incorrect password.")

        with col_checklist:
            with st.container(border=True):
                st.markdown("**Practical Task Requirements**")
                _check = "<span style='color:#2ca02c; font-weight:bold'>&#10003;</span>"
                _cross = "<span style='color:#999'>&#9744;</span>"
                st.markdown(
                    f"{_check} Document analysis app with data extraction and visualization "
                    "(Streamlit, 4 tabs: Chat, Corpus, Evaluation, Dashboard)<br>"
                    f"{_check} Evaluation metrics defined and demonstrated "
                    "(RAG Triad: Faithfulness, Answer Relevancy, Context Precision; "
                    "18-question eval dataset, LLM-as-judge)<br>"
                    f"{_check} LLM via EPAM DIAL API (with OpenAI fallback)<br>"
                    f"{_check} Ninja: Corpus update without Vector DB rebuild "
                    "(SHA-256 dedup, incremental indexing)<br>"
                    f"{_check} Ninja: Access-control-aware RAG "
                    "(public/restricted clearance filtering at retrieval, eval, and upload)<br>"
                    f"{_cross} Ninja: Multi-modal RAG for graphical content "
                    "(not implemented)<br>"
                    f"{_check} Ninja: RAG evaluation with precision, faithfulness, groundedness "
                    "(position-weighted Context Precision, Faithfulness via claim decomposition)<br>"
                    f"{_check} Hybrid retrieval pipeline "
                    "(Semantic + BM25 via Reciprocal Rank Fusion)",
                    unsafe_allow_html=True,
                )

        st.stop()

# ---------------------------------------------------------------------------
# Guard: DIAL credentials required
# ---------------------------------------------------------------------------

DIAL_TOKEN = _secret("DIAL_TOKEN") or _secret("OPENAI_API_KEY")
if not DIAL_TOKEN:
    st.warning(
        "No API credentials found. "
        "Set DIAL_TOKEN (EPAM DIAL) or OPENAI_API_KEY (OpenAI) in .env, then restart."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Role / clearance (session state - driven by toggle button in sidebar)
# Starts as "public"; user can switch to "restricted" inside the app.
# ---------------------------------------------------------------------------

if "user_role" not in st.session_state:
    st.session_state["user_role"] = "public"


def _clearance() -> str:
    return st.session_state["user_role"]


# ---------------------------------------------------------------------------
# Pipeline initialization (cached per Streamlit server process)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Initializing RAG pipeline...")
def _init_pipeline(base_url: str, token: str, chat_model: str, embed_model: str, provider: str = "dial"):
    """
    Build pipeline components and pre-load documents from data/.
    Cached: runs once per Streamlit server process.
    """
    from rag.pipeline import build_pipeline_from_env, VectorStore, HybridRetriever
    from rag.evaluation import RagTriadEvaluator

    client, store, retriever, generator = build_pipeline_from_env(
        base_url=base_url,
        token=token,
        chat_model=chat_model,
        embed_model=embed_model,
        provider=provider,
    )
    evaluator = RagTriadEvaluator(client)

    data_dir = Path(__file__).parent / "data"
    if data_dir.exists():
        for fpath in sorted(data_dir.glob("*.txt")):
            clearance = "restricted" if "restricted" in fpath.name else "public"
            text = fpath.read_text(encoding="utf-8", errors="replace")
            store.add_document(text, fpath.name, clearance=clearance)
        # Rebuild BM25 with all chunks visible to restricted users (superset)
        retriever.rebuild_bm25("restricted")

    return client, store, retriever, generator, evaluator


def _get_pipeline():
    openai_key = _secret("OPENAI_API_KEY")
    if openai_key:
        return _init_pipeline(
            base_url="",
            token=openai_key,
            chat_model=_secret("OPENAI_CHAT_MODEL", "gpt-4o"),
            embed_model=_secret("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            provider="openai",
        )
    return _init_pipeline(
        base_url=_secret("DIAL_BASE_URL", "https://ai-proxy.lab.epam.com"),
        token=_secret("DIAL_TOKEN"),
        chat_model=_secret("DIAL_CHAT_MODEL", "gpt-4o"),
        embed_model=_secret("DIAL_EMBEDDING_MODEL", "text-embedding-005"),
        provider="dial",
    )


try:
    client, store, retriever, generator, evaluator = _get_pipeline()
except Exception as exc:
    st.error(f"Failed to initialize pipeline: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar: role toggle + corpus stats
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Access")

    is_authenticated = _clearance() == "restricted"
    role_label = "Authenticated User" if is_authenticated else "Public User"
    toggle_label = "Switch to Public User" if is_authenticated else "Switch to Authenticated User"

    st.markdown(
        f"**Current role:** "
        f"<span style='color:{'#d62728' if is_authenticated else '#2ca02c'}; font-weight:bold'>"
        f"{role_label}</span>",
        unsafe_allow_html=True,
    )

    if st.button(toggle_label, width='stretch'):
        st.session_state["user_role"] = "public" if is_authenticated else "restricted"
        # Clear chat history on role switch to avoid mixing contexts
        st.session_state["chat_history"] = []
        st.rerun()

    st.divider()
    st.header("Corpus")
    total = store.total_chunks()
    st.metric("Total chunks", total)

    summary = store.corpus_summary()
    if summary:
        for doc in summary:
            badge = "[R]" if doc["clearance"] == "restricted" else "[P]"
            label_color = "red" if doc["clearance"] == "restricted" else "green"
            visible = doc["clearance"] == "public" or _clearance() == "restricted"
            dim = "" if visible else " *(hidden)*"
            st.caption(
                f"<span style='color:{label_color}'>{badge}</span> "
                f"{doc['source']} ({doc['chunks']} chunks){dim}",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No documents loaded.")

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

tab_chat, tab_corpus, tab_eval, tab_dashboard = st.tabs(
    ["Chat", "Corpus", "Evaluation", "Quality Dashboard"]
)

# ============================================================
# TAB: Chat
# ============================================================

with tab_chat:
    # ----------------------------------------------------------------
    # CSS: pin chat input to viewport bottom + page padding.
    # ----------------------------------------------------------------
    st.markdown(
        """
<style>
/* ---- Responsive layout: reduce wasted padding ---- */
.stMainBlockContainer {
    padding-bottom: 0 !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}

/* Keep inner keyed blocks sized to wrapper height so page scroll is not used.
   Actual wrapper height is calculated dynamically by JS below. */
.st-key-chat-box,
.st-key-info-panel {
    height: 100% !important;
    min-height: 0 !important;
}

/* ---- Pin chat input to viewport bottom ---- */
[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 0 !important;
    left: 21rem !important;
    right: 0 !important;
    z-index: 9999 !important;
    background: var(--background-color, #ffffff) !important;
    padding: 0.5rem 0 !important;
    border-top: 1px solid rgba(49, 51, 63, 0.10) !important;
    box-shadow: 0 -3px 10px rgba(0, 0, 0, 0.06) !important;
}
section.main .block-container {
    padding-bottom: 110px !important;
}
</style>
""",
        unsafe_allow_html=True,
    )

    if "scroll_chat_to_bottom" not in st.session_state:
        st.session_state["scroll_chat_to_bottom"] = False

    st.html(
        """
<script>
// ---------------------------------------------------------------------------
// Panel height sync for all tabs (Chat, Corpus, Evaluation, Dashboard).
//
// Each tab has two columns: main content and info panel. Both use
// st.container(height="stretch") which creates a stLayoutWrapper in the DOM.
// This script constrains every wrapper to fill exactly the remaining viewport
// height, enabling independent scrolling per panel without page overflow.
//
// Key technique - "clear-reflow-apply" on every sync cycle:
//   1. Strip inline height/overflow from ALL wrappers (visible and hidden).
//   2. Force a layout reflow so getBoundingClientRect returns natural sizes.
//   3. Apply explicit pixel height only to wrappers in the active tab.
// This prevents stale measurements from a previously-active tab polluting
// the layout when the user switches back.
//
// Triggers: MutationObserver (DOM changes), resize, tab clicks (delayed).
// ---------------------------------------------------------------------------
(() => {
    window.__ragShouldAutoScroll = __AUTO_SCROLL__;
    if (__AUTO_SCROLL__) {
        window.__ragAutoScrollTimer = setTimeout(() => {
            window.__ragShouldAutoScroll = false;
        }, 800);
    }

    // Re-use the existing sync function if the script runs again (Streamlit rerenders).
    if (window.__ragPanelHeightSync) {
        if (__AUTO_SCROLL__) {
            window.__ragShouldAutoScroll = true;
            // Keep flag alive long enough for MutationObserver-triggered syncs
            // to catch the newly rendered chat messages.
            clearTimeout(window.__ragAutoScrollTimer);
            window.__ragAutoScrollTimer = setTimeout(() => {
                window.__ragShouldAutoScroll = false;
            }, 800);
        }
        window.__ragPanelHeightSync();
        return;
    }

    const RESERVE_MARGIN = 12;
    const SERVICE_MESSAGE_GAP = 40;

    // Registry of all managed panels.
    // 'chat' reserve accounts for the fixed chat input bar at the bottom.
    // 'pad' reserve is a small bottom padding for non-chat tabs.
    const ALL_SELECTORS = [
        { sel: '.st-key-chat-box',    reserve: 'chat' },
        { sel: '.st-key-info-panel',  reserve: 'chat' },
        { sel: '.st-key-corpus-main', reserve: 'pad' },
        { sel: '.st-key-corpus-info', reserve: 'pad' },
        { sel: '.st-key-eval-main',   reserve: 'pad' },
        { sel: '.st-key-eval-info',   reserve: 'pad' },
        { sel: '.st-key-dash-main',   reserve: 'pad' },
        { sel: '.st-key-dash-info',   reserve: 'pad' },
    ];

    // Find the stLayoutWrapper ancestor for a given st-key container.
    const findWrapper = (sel) => {
        const block = document.querySelector(sel);
        if (!block) return null;
        return block.closest('[data-testid="stLayoutWrapper"]');
    };

    // Remove all inline sizing so Streamlit's default flex layout takes over.
    const clearSizing = (el) => {
        el.style.height = '';
        el.style.maxHeight = '';
        el.style.overflowY = '';
        el.style.overflowX = '';
        el.style.overscrollBehavior = '';
    };

    // Set explicit pixel height = viewport bottom minus element top minus reserve.
    const applySizing = (el, reserve) => {
        const top = el.getBoundingClientRect().top;
        if (top <= 0) return; // element not laid out yet or off-screen
        const available = Math.max(180, Math.floor(window.innerHeight - top - reserve));
        el.style.height = `${available}px`;
        el.style.maxHeight = `${available}px`;
        el.style.overflowY = 'auto';
        el.style.overflowX = 'hidden';
        el.style.overscrollBehavior = 'contain';
    };

    const syncPanelHeights = () => {
        // --- Position the fixed chat input to match the chat column width ---
        const chatInput = document.querySelector('[data-testid="stChatInput"]');
        const chatBlock = document.querySelector('.st-key-chat-box');
        const chatWrapper = chatBlock
            ? chatBlock.closest('[data-testid="stLayoutWrapper"]')
            : null;

        if (chatInput && chatWrapper) {
            const rect = chatWrapper.getBoundingClientRect();
            const left = Math.max(0, Math.round(rect.left));
            const right = Math.max(0, Math.round(window.innerWidth - rect.right));
            const width = Math.max(0, Math.round(rect.width));
            chatInput.style.left = `${left}px`;
            chatInput.style.right = `${right}px`;
            chatInput.style.width = `${width}px`;
        } else if (chatInput) {
            chatInput.style.left = '21rem';
            chatInput.style.right = '0';
            chatInput.style.width = 'auto';
        }

        // Chat panels need extra bottom reserve for the fixed input bar.
        const chatReserve = chatInput
            ? Math.ceil(chatInput.getBoundingClientRect().height) + RESERVE_MARGIN + SERVICE_MESSAGE_GAP
            : 136;
        const BOTTOM_PAD = 16;

        // STEP 1: Clear ALL wrapper inline styles (prevents stale sizes on hidden tabs)
        ALL_SELECTORS.forEach(({ sel }) => {
            const w = findWrapper(sel);
            if (w) clearSizing(w);
        });

        // STEP 2: Force layout reflow for clean measurements
        void document.body.offsetHeight;

        // STEP 3: Apply sizing only to wrappers in the currently visible tab
        ALL_SELECTORS.forEach(({ sel, reserve }) => {
            const w = findWrapper(sel);
            if (!w) return;
            const r = w.getBoundingClientRect();
            if (r.height === 0 || r.top <= 0) return; // hidden tab or not rendered
            const px = reserve === 'chat' ? chatReserve : BOTTOM_PAD;
            applySizing(w, px);

            // Chat history box gets a visible border and auto-scroll to bottom.
            if (sel === '.st-key-chat-box') {
                w.style.boxSizing = 'border-box';
                w.style.border = '1px solid rgba(49, 51, 63, 0.20)';
                w.style.borderRadius = '0.5rem';
                if (window.__ragShouldAutoScroll) {
                    w.scrollTop = w.scrollHeight;
                }
            }
        });
    };

    // Debounce via requestAnimationFrame to avoid layout thrashing.
    let raf = null;
    const schedule = () => {
        if (raf !== null) cancelAnimationFrame(raf);
        raf = requestAnimationFrame(() => {
            raf = null;
            syncPanelHeights();
        });
    };

    window.__ragPanelHeightSync = schedule;
    window.addEventListener('resize', schedule, { passive: true });

    // Re-sync on any DOM mutation (Streamlit adds/removes elements dynamically).
    const observer = new MutationObserver(schedule);
    observer.observe(document.body, { childList: true, subtree: true });

    // Tab switch: Streamlit swaps tab content asynchronously, so we re-sync
    // after short delays to catch the moment the new tab's DOM is laid out.
    document.addEventListener('click', (e) => {
        if (e.target.closest('[role="tab"]')) {
            setTimeout(syncPanelHeights, 80);
            setTimeout(syncPanelHeights, 250);
        }
    }, true);

    schedule();
})();
</script>
""".replace("__AUTO_SCROLL__", "true" if st.session_state["scroll_chat_to_bottom"] else "false"),
        unsafe_allow_javascript=True,
    )

    if st.session_state["scroll_chat_to_bottom"]:
        st.session_state["scroll_chat_to_bottom"] = False

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    col_chat_main, col_chat_info = st.columns([3, 1])

    with col_chat_info:
        with st.container(border=True, height="stretch", key="info-panel"):
            st.markdown("**How retrieval works**")
            st.markdown(
                """
**Step 1 - Clearance filter**

All chunks are pre-filtered by role before any search. Public User sees only public chunks; Authenticated User sees all.

**Step 2 - Semantic search**

The query is embedded into a vector. ChromaDB returns the top-10 nearest chunks by cosine distance.

Filter: chunks with `dist >= 0.85` are dropped - they are geometrically far from the query even if they ranked first.

**Step 3 - BM25 keyword search**

The query is tokenized. Okapi BM25 scores every chunk by term frequency and inverse document frequency.

Filter: chunks are dropped if their score is below `top_score × 0.25`. This is a relative threshold - it adapts to the corpus. If the best BM25 match scores 10, chunks below 2.5 are excluded. This prevents a chunk that matched 1-2 incidental keywords from entering the ranking at position 1.

Note: if the top BM25 score is 0 (no query term appears anywhere in the corpus), all BM25 results are dropped.

**Step 4 - RRF fusion**

The surviving chunks from both methods are merged by rank position:

`rrf = 1/(60+sem_rank) + 1/(60+bm25_rank)`

A chunk present in both lists scores highest. A chunk from only one method still contributes via its own rank. The final top-5 by RRF score are passed to the model.

Important: RRF uses only rank, not the raw BM25 score or cosine distance. `rrf=0.032` means "best available in the filtered set" - it says nothing about absolute relevance.

**Step 5 - Generation**

The model receives the top-5 chunks as context and is instructed to answer only from that context. If the relevant information was filtered out (wrong clearance, distance too large, no keyword overlap), the model will say it cannot answer.

**Score labels per chunk:**
- `sem(rank=N, dist=D)` - rank in semantic results + cosine distance (0=identical, 1=orthogonal)
- `bm25(rank=N, score=S)` - rank in BM25 results + raw Okapi score (corpus-relative, not normalized)
- `rrf=` - final fused score
- `sem:-` / `bm25:-` - chunk was absent from that method's filtered results
"""
            )

    with col_chat_main:
        role_display = "Authenticated User" if _clearance() == "restricted" else "Public User"
        col_caption, col_clear = st.columns([5, 1])
        col_caption.caption(f"Hybrid retrieval (semantic + BM25 via RRF) - role: **{role_display}**")
        with col_clear:
            if st.session_state["chat_history"]:
                if st.button("Clear", key="clear_chat", width='stretch'):
                    st.session_state["chat_history"] = []
                    st.rerun()

        chat_box = st.container(height="stretch", key="chat-box")
        with chat_box:
            # Keep container DOM materialized even when history is empty,
            # so JS can apply wrapper border immediately.
            st.markdown("<div style='height:0; overflow:hidden'></div>", unsafe_allow_html=True)

            for entry in st.session_state["chat_history"]:
                with st.chat_message("user"):
                    st.write(entry["question"])
                with st.chat_message("assistant"):
                    st.write(entry["answer"])
                    if entry.get("chunks"):
                        with st.expander(f"Source chunks ({len(entry['chunks'])})"):
                            for i, c in enumerate(entry["chunks"], 1):
                                sem_r   = c.get("sem_rank")
                                bm25_r  = c.get("bm25_rank")
                                rrf_s   = c.get("rrf_score")
                                sem_d   = c.get("sem_distance")
                                bm25_s  = c.get("bm25_score")

                                sem_parts = [f"rank={sem_r}"]
                                if sem_d is not None:
                                    sem_parts.append(f"dist={sem_d:.4f}")
                                sem_tag = "sem(" + ", ".join(sem_parts) + ")" if sem_r else "sem:-"

                                bm25_parts = [f"rank={bm25_r}"]
                                if bm25_s is not None:
                                    bm25_parts.append(f"score={bm25_s:.2f}")
                                bm25_tag = "bm25(" + ", ".join(bm25_parts) + ")" if bm25_r else "bm25:-"

                                rrf_tag = f"rrf={rrf_s:.5f}" if rrf_s is not None else ""

                                scores_line = "  |  ".join(filter(None, [sem_tag, bm25_tag, rrf_tag]))

                                st.markdown(
                                    f"**{i}. {c['source']}** [{c['clearance']}]  "
                                    f"<span style='font-size:0.78em; color:gray'>{scores_line}</span>",
                                    unsafe_allow_html=True,
                                )
                                st.text(c["text"])

    question = st.chat_input("Ask about the EU AI Act...")
    if question:
        if store.total_chunks() == 0:
            st.warning("Corpus is empty. Upload documents in the Corpus tab first.")
        else:
            with st.spinner("Retrieving and generating..."):
                try:
                    clearance = _clearance()
                    retriever.rebuild_bm25(clearance)
                    chunks = retriever.retrieve(question, clearance=clearance, top_n=5)
                    answer, _ = generator.generate(question, chunks)
                except Exception as exc:
                    answer = f"Error: {exc}"
                    chunks = []

            st.session_state["chat_history"].append(
                {"question": question, "answer": answer, "chunks": chunks}
            )
            st.session_state["scroll_chat_to_bottom"] = True
            st.rerun()

# ============================================================
# TAB: Corpus
# ============================================================

with tab_corpus:
    col_corp_main, col_corp_info = st.columns([3, 1])

    with col_corp_info:
        with st.container(border=True, height="stretch", key="corpus-info"):
            st.markdown("**About the knowledge base**")
            st.markdown(
                """
**Vector store:** ChromaDB in-memory. Each document is split into ~2000-character chunks with 200-character overlap.

**Embeddings:** each chunk is embedded at index time via the configured embedding model and stored alongside its text in the vector store.

**BM25 index** is rebuilt before each query to include only chunks visible to the current user role. This ensures clearance filtering applies to keyword search as well as semantic search.

**Deduplication:** SHA-256 hash of each chunk text. Re-uploading the same document skips already-indexed chunks.

**Clearance levels:**
- `[P]` Public - visible to all users
- `[R]` Restricted - visible to Authenticated Users only

Documents loaded from `data/` at startup. Files with `restricted` in the name are indexed as restricted.
"""
            )

    with col_corp_main:
      with st.container(height="stretch", key="corpus-main"):
        st.header("Document Corpus")

        summary = store.corpus_summary()
        if summary:
            st.subheader("Loaded documents")
            for doc in summary:
                badge_color = "red" if doc["clearance"] == "restricted" else "green"
                bg_color = "#fff0f0" if doc["clearance"] == "restricted" else "#f0fff0"
                st.markdown(
                    f"**{doc['source']}** &nbsp; "
                    f"<span style='color:gray; font-size:0.9em'>{doc['chunks']} chunks</span>"
                    f"&nbsp;&nbsp;"
                    f"<span style='color:{badge_color}; font-weight:bold; background:{bg_color};"
                    f" padding:2px 8px; border-radius:4px; font-size:0.82em'>"
                    f"{doc['clearance'].upper()}</span>",
                    unsafe_allow_html=True,
                )
                with st.expander("Show chunks and terms"):
                    doc_tab_chunks, doc_tab_terms = st.tabs(["Chunks", "Terms"])

                    doc_chunks = store.get_document_chunks(doc["source"], _clearance())

                    with doc_tab_chunks:
                        if not doc_chunks:
                            st.caption("No chunks accessible with current role.")
                        else:
                            for ch in doc_chunks:
                                st.markdown(f"**#{ch['chunk_index'] + 1}**")
                                st.text(ch["text"])
                                st.divider()

                    with doc_tab_terms:
                        if not doc_chunks:
                            st.caption("No chunks accessible with current role.")
                        else:
                            import re as _re
                            from collections import Counter as _Counter
                            all_text = " ".join(ch["text"] for ch in doc_chunks)
                            tokens = _re.findall(r"\w+", all_text.lower())
                            stopwords = {
                                "the", "a", "an", "of", "to", "in", "and", "or", "for",
                                "is", "are", "be", "was", "were", "that", "this", "it",
                                "on", "at", "by", "as", "with", "not", "its", "their",
                                "from", "shall", "which", "have", "has", "been",
                            }
                            counts = _Counter(
                                t for t in tokens if len(t) > 2 and t not in stopwords
                            )
                            top = counts.most_common(40)
                            if top:
                                max_count = top[0][1]
                                rows = []
                                for term, count in top:
                                    bar = "█" * max(1, round(count / max_count * 20))
                                    rows.append({"term": term, "count": count, "freq bar": bar})
                                st.dataframe(rows, width='stretch', hide_index=True)
        else:
            st.info("Corpus is empty.")

        st.divider()

        if _clearance() == "restricted":
            st.subheader("Upload a document")
            st.caption(
                "Supported: .txt, .pdf - Incremental indexing: existing documents are skipped (SHA-256 dedup)."
            )

            uploaded = st.file_uploader("Choose a file", type=["txt", "pdf"])

            clearance_choice = st.radio(
                "Document clearance",
                ["public", "restricted"],
                horizontal=True,
                help="Public documents are visible to all users. Restricted are only visible to authenticated users.",
            )

            if uploaded and st.button("Index document"):
                with st.spinner("Reading and indexing..."):
                    try:
                        if uploaded.name.endswith(".pdf"):
                            try:
                                import pypdf
                                reader = pypdf.PdfReader(io.BytesIO(uploaded.read()))
                                text = "\n\n".join(
                                    page.extract_text() or "" for page in reader.pages
                                )
                            except ImportError:
                                st.error("pypdf not installed. Run: pip install pypdf")
                                text = ""
                        else:
                            text = uploaded.read().decode("utf-8", errors="replace")

                        if text.strip():
                            added, skipped = store.add_document(
                                text, uploaded.name, clearance=clearance_choice
                            )
                            retriever.rebuild_bm25(_clearance())
                            if added > 0:
                                st.success(f"Indexed {added} chunks from '{uploaded.name}'.")
                            else:
                                st.info(
                                    f"Document '{uploaded.name}' already indexed ({skipped} chunks)."
                                )
                            st.rerun()
                        else:
                            st.warning("Document appears to be empty.")
                    except Exception as exc:
                        st.error(f"Failed to index: {exc}")
        else:
            st.info("Document upload is available to Authenticated Users only. Use the toggle in the sidebar.")

# ============================================================
# TAB: Evaluation
# ============================================================

with tab_eval:
    col_eval_main, col_eval_info = st.columns([3, 1])

    with col_eval_info:
        with st.container(border=True, height="stretch", key="eval-info"):
            st.markdown("**About the evaluation**")
            st.markdown(
                """
**RAG Triad** - three independent LLM-as-judge metrics. No ground-truth answers required.

**Faithfulness** - does the answer stay within the retrieved context? Detects hallucination. Score 1.0 = fully grounded.

**Answer Relevancy** - does the answer address the question? Detects vague or off-topic responses.

**Context Precision@K** - are relevant chunks ranked at the top? Penalises retrieval noise and poor ordering.

**P@K formula:**
```
P@K = (1/|R|) * sum(P@k * rel_k)
```
where P@k = (# relevant in top-k) / k, rel_k = 1 if chunk k is relevant.

Example with 2 relevant chunks out of 5 retrieved:
- Relevant at rank 1,2: P@K = 1.0
- Relevant at rank 4,5: P@K = 0.33

**Judge model:** the same LLM used for generation also evaluates its own outputs via structured prompts.

**Thresholds:**
- > 0.8 - good
- 0.6-0.8 - acceptable
- < 0.6 - needs improvement

Low faithfulness = hallucination risk. Low context precision = retrieval is noisy or poorly ranked.
"""
            )

    with col_eval_main:
      with st.container(height="stretch", key="eval-main"):
        st.header("RAG Triad Evaluation")
        st.caption(
            "LLM-as-judge scoring: Faithfulness, Answer Relevancy, Context Precision."
        )

        eval_path = Path(__file__).parent / "eval_questions.json"
        if not eval_path.exists():
            st.warning("eval_questions.json not found.")
            st.stop()

        with open(eval_path) as f:
            eval_questions: list[dict] = json.load(f)

        clearance = _clearance()
        visible_questions = [
            q for q in eval_questions
            if q.get("clearance", "public") == "public" or clearance == "restricted"
        ]

        role_display = "Authenticated User" if clearance == "restricted" else "Public User"
        st.write(f"{len(visible_questions)} questions available for role: **{role_display}**")

        if st.button("Run Evaluation", type="primary", disabled=(store.total_chunks() == 0)):
            results = []
            progress = st.progress(0)
            status_text = st.empty()

            for idx, q in enumerate(visible_questions):
                status_text.text(f"Evaluating {idx + 1}/{len(visible_questions)}: {q['question'][:60]}...")
                try:
                    retriever.rebuild_bm25(clearance)
                    chunks = retriever.retrieve(q["question"], clearance=clearance, top_n=5)
                    answer, _ = generator.generate(q["question"], chunks)
                    score = evaluator.evaluate(q["question"], answer, chunks)
                    results.append(
                        {
                            "id": q.get("id", f"q{idx+1:02d}"),
                            "label": q["question"][:50],
                            "question": q["question"],
                            "answer": answer,
                            "clearance": q.get("clearance", "public"),
                            "chunks": chunks,
                            **score.as_dict(),
                        }
                    )
                except Exception as exc:
                    results.append(
                        {
                            "id": q.get("id", f"q{idx+1:02d}"),
                            "label": q["question"][:50],
                            "question": q["question"],
                            "answer": f"Error: {exc}",
                            "clearance": q.get("clearance", "public"),
                            "chunks": [],
                            "faithfulness": 0.0,
                            "answer_relevancy": 0.0,
                            "context_precision": 0.0,
                            "average": 0.0,
                        }
                    )
                progress.progress((idx + 1) / len(visible_questions))

            status_text.empty()
            st.session_state["eval_results"] = results
            st.success(f"Evaluation complete. {len(results)} questions scored.")
            st.rerun()

        if "eval_results" in st.session_state:
            results = st.session_state["eval_results"]
            avg_f = sum(r["faithfulness"] for r in results) / len(results)
            avg_r = sum(r["answer_relevancy"] for r in results) / len(results)
            avg_p = sum(r["context_precision"] for r in results) / len(results)

            c1, c2, c3 = st.columns(3)
            c1.metric("Avg Faithfulness", f"{avg_f:.2f}")
            c2.metric("Avg Answer Relevancy", f"{avg_r:.2f}")
            c3.metric("Avg Context Precision", f"{avg_p:.2f}")

            st.divider()

            for r in results:
                f_val = r['faithfulness']
                r_val = r['answer_relevancy']
                p_val = r['context_precision']
                avg_val = r['average']

                def _score_color(v: float) -> str:
                    if v >= 0.8: return "#2ca02c"
                    if v >= 0.6: return "#ff7f0e"
                    return "#d62728"

                score_html = (
                    f"<span style='color:{_score_color(f_val)}'>F:{f_val:.2f}</span> &nbsp;"
                    f"<span style='color:{_score_color(r_val)}'>R:{r_val:.2f}</span> &nbsp;"
                    f"<span style='color:{_score_color(p_val)}'>P:{p_val:.2f}</span> &nbsp;"
                    f"<span style='color:{_score_color(avg_val)}; font-weight:bold'>avg:{avg_val:.2f}</span>"
                )

                with st.expander(f"[{r['id']}] {r['question'][:70]}"):
                    # scores summary
                    st.markdown(score_html, unsafe_allow_html=True)
                    st.caption(
                        "F = Faithfulness (grounding) | R = Answer Relevancy | P = Context Precision"
                    )
                    st.divider()

                    # answer
                    st.markdown("**Answer**")
                    st.write(r["answer"])

                    # retrieved chunks with scores
                    chunks_used = r.get("chunks", [])
                    if chunks_used:
                        st.divider()
                        st.markdown(f"**Retrieved chunks ({len(chunks_used)})**")
                        for i, c in enumerate(chunks_used, 1):
                            sem_r  = c.get("sem_rank")
                            bm25_r = c.get("bm25_rank")
                            rrf_s  = c.get("rrf_score")
                            sem_d  = c.get("sem_distance")
                            bm25_s = c.get("bm25_score")

                            sem_parts = [f"rank={sem_r}"]
                            if sem_d is not None:
                                sem_parts.append(f"dist={sem_d:.4f}")
                            sem_tag = "sem(" + ", ".join(sem_parts) + ")" if sem_r else "sem:-"

                            bm25_parts = [f"rank={bm25_r}"]
                            if bm25_s is not None:
                                bm25_parts.append(f"score={bm25_s:.2f}")
                            bm25_tag = "bm25(" + ", ".join(bm25_parts) + ")" if bm25_r else "bm25:-"

                            rrf_tag = f"rrf={rrf_s:.5f}" if rrf_s is not None else ""
                            scores_line = "  |  ".join(filter(None, [sem_tag, bm25_tag, rrf_tag]))

                            st.markdown(
                                f"**{i}. {c['source']}** [{c['clearance']}]  "
                                f"<span style='font-size:0.78em; color:gray'>{scores_line}</span>",
                                unsafe_allow_html=True,
                            )
                            st.text(c["text"])

# ============================================================
# TAB: Quality Dashboard
# ============================================================

with tab_dashboard:
    col_dash_main, col_dash_info = st.columns([3, 1])

    with col_dash_info:
        with st.container(border=True, height="stretch", key="dash-info"):
            st.markdown("**About the charts**")
            st.markdown(
                """
**Three metrics form the RAG Triad:**

**Faithfulness** - answer grounded in context. Low = hallucination risk.

**Answer Relevancy** - answer addresses the question. Low = vague or off-topic.

**Context Precision** - retrieved context is relevant. Low = retrieval noise.

**Radar chart** (per question) - shows the metric profile of each individual question. Good answers have a large, even triangle.

**Bar chart** - per-question average score. Identifies which questions are hardest for the system.

**Average radar** - overall system health. Target: all axes > 0.8.

**Interpreting low scores:**
- Low faithfulness - model ignores context or makes up facts
- Low context precision - BM25/semantic retrieval brings in unrelated chunks
- Low answer relevancy - question is out of scope for the corpus
"""
            )

    with col_dash_main:
      with st.container(height="stretch", key="dash-main"):
        st.header("Quality Dashboard")

        if "eval_results" not in st.session_state:
            st.info("Run evaluation first (Evaluation tab).")
        else:
            from rag.visualization import radar_chart, bar_chart, average_radar

            results = st.session_state["eval_results"]
            chart_data = [
                {
                    "label": r["label"],
                    "faithfulness": r["faithfulness"],
                    "answer_relevancy": r["answer_relevancy"],
                    "context_precision": r["context_precision"],
                }
                for r in results
            ]

            col_l, col_r = st.columns(2)

            with col_l:
                st.subheader("Per-question radar")
                st.plotly_chart(radar_chart(chart_data), width='stretch')

            with col_r:
                st.subheader("Average scores")
                st.plotly_chart(average_radar(chart_data), width='stretch')

            st.subheader("Per-question bar chart")
            st.plotly_chart(bar_chart(chart_data), width='stretch')

            # Summary table
            st.subheader("Score table")
            table_rows = [
                {
                    "ID": r["id"],
                    "Question": r["question"][:60],
                    "Faithfulness": f"{r['faithfulness']:.2f}",
                    "Answer Relevancy": f"{r['answer_relevancy']:.2f}",
                    "Context Precision": f"{r['context_precision']:.2f}",
                    "Average": f"{r['average']:.2f}",
                }
                for r in results
            ]
            st.dataframe(table_rows, width='stretch')
