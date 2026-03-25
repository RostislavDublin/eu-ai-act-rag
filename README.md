# EU AI Act RAG Assistant

Interactive Q&A application for the EU Artificial Intelligence Act. Ask questions about the regulation and get grounded answers with source citations, backed by a hybrid retrieval pipeline and automated quality evaluation.

## Quick start

### Hosted version

The app is deployed on Streamlit Community Cloud. Open the link provided with the submission - no local setup needed.

### Local run

```bash
cp .env-template .env        # fill in your API credentials
pip install -r requirements.txt
streamlit run app.py
```

The app loads the EU AI Act dataset from `data/` on startup and builds the vector index automatically. If `APP_PASSWORD` is set in the environment, the app will prompt for a password before granting access.

## User journey

### 1. Login

If the app is password-protected, enter the access password on the login screen. The right panel shows a checklist of implemented features for quick reference.

### 2. Chat tab - ask questions

Type a question about the EU AI Act in the input box at the bottom. The assistant retrieves relevant passages from the corpus and generates a grounded answer. The left sidebar shows your current access role and retrieval pipeline details.

Try switching roles with the "Switch to Public User" / "Switch to Authenticated User" button in the sidebar. Public users see only public documents; authenticated users also see restricted internal documents. This demonstrates access-controlled RAG at retrieval time.

Example questions to try:
- "What are the requirements for high-risk AI systems?"
- "What penalties does the EU AI Act impose for non-compliance?"
- "How does the Act define an AI system?"

### 3. Corpus tab - explore the knowledge base

See which documents are loaded, how many chunks each contains, and the clearance level (Public / Restricted) for each. Upload additional .txt or .pdf files - they are chunked and indexed incrementally without rebuilding the entire index (SHA-256 deduplication).

### 4. Evaluation tab - measure answer quality

Run the built-in evaluation dataset (12 questions with expected answers). The app scores each answer on three dimensions of the RAG Triad:
- **Faithfulness** - is the answer grounded in retrieved context? (no hallucination)
- **Answer Relevancy** - does the answer address the question? (no off-topic drift)
- **Context Precision** - did the retriever pull the right chunks? (retrieval quality)

Each metric is scored by an LLM-as-judge with a structured rubric. Retrieved chunks are shown alongside scores for transparency.

### 5. Quality Dashboard tab - visualize metrics

After running evaluation, view the results as interactive charts:
- **Radar chart** - shows average scores across all three RAG Triad dimensions
- **Per-question bar chart** - grouped bars showing individual metric scores for each evaluated question

## Architecture

```
documents -> chunking (2000 chars, 200 overlap) -> embeddings -> ChromaDB (in-memory)

user question -> [semantic search (cosine)] + [BM25 keyword search] -> RRF fusion -> top-K chunks
             -> prompt + context -> LLM -> answer
             -> LLM-as-judge evaluation -> quality metrics
```

Key components:
- **Hybrid retrieval**: semantic search (ChromaDB cosine similarity) combined with BM25 keyword search, fused via Reciprocal Rank Fusion
- **Access control**: each chunk carries a clearance metadata tag; retrieval filters by user role before search
- **Incremental indexing**: new documents are added without rebuilding the vector store
- **RAG Triad evaluation**: automated quality scoring using LLM-as-judge

## Configuration

Copy `.env-template` to `.env` and fill in credentials. Two LLM provider options:

**Option A - EPAM DIAL proxy:**
- `DIAL_BASE_URL`, `DIAL_TOKEN`, `DIAL_CHAT_MODEL`, `DIAL_EMBEDDING_MODEL`

**Option B - OpenAI API:**
- `OPENAI_API_KEY`, `OPENAI_CHAT_MODEL`, `OPENAI_EMBEDDING_MODEL`

If both are set, OpenAI takes priority. Set `APP_PASSWORD` to protect the app with a password gate.

For Streamlit Cloud deployment, add the same variables in Settings > Secrets (TOML format).

## Project structure

```
.env-template          # credential template (safe to commit)
.env                   # real credentials (gitignored)
requirements.txt       # Python dependencies
app.py                 # Streamlit entry point
rag/
  pipeline.py          # chunking, embedding, retrieval, generation
  evaluation.py        # RAG Triad LLM-as-judge scoring
  visualization.py     # radar chart, bar charts (Plotly)
data/                  # EU AI Act text files
eval_questions.json    # evaluation Q&A pairs with expected answers
```

This mirrors a real SSO pattern stripped to essentials: token presence = identity claim, metadata filter = enforcement point.

### 3. Full RAG Triad + precision/recall metrics

Context Precision and Context Recall added to evaluation alongside Faithfulness and Answer Relevancy. All four axes on the radar chart.

### Optional (if time allows)

- [ ] Reranking step (two-stage retrieval)
- [ ] Multi-modal RAG (extract tables/diagrams from PDF via GPT-4o vision)
