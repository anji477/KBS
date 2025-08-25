## Knowledge Base Search - 5 Minute Presentation

### 1) Problem & Goal
- Build a lightweight KB search to quickly find info from product/company docs.

### 2) Demo Flow
1. Upload/build index from a sample file (CSV/JSON/TXT/PDF)
2. Run 2–3 searches; show results ranked by TF-IDF
3. Show last 5 recent searches
4. Trigger a miss and show suggestions ("Check spelling", "Try related terms")

### 3) Architecture
- CLI in Python
- Loaders for CSV/JSON/TXT/PDF
- TF-IDF vectorization with title boost
- Simple JSON cache in `~/.kbs_cache` for index and recents

### 4) Challenges
- Title/key detection across heterogeneous data (CSV/JSON)
- PDF text extraction quality varies by source
- Balancing speed vs. accuracy with a generic TF-IDF (no domain tuning)

### 5) Possible Enhancements
- Add semantic embeddings (e.g., Sentence Transformers) with hybrid search
- Synonym/typo handling via fuzzy matching and query expansion
- Simple web UI and multi-file indexing 