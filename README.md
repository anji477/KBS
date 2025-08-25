## Knowledge Base Search (CLI)

A minimal CLI tool to upload a knowledge base file (CSV/JSON/TXT/PDF) and run TF-IDF search over titles/keys and content, with last-5 recent searches.

### Features
- Upload/build index from a file: CSV, JSON, TXT, PDF
- Search queries across titles/keys and content
- Remembers last 5 searches per file
- Helpful suggestions when no results are found

### Setup

1. Create a virtual environment and install dependencies
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Usage

- Upload/build index for a file
```bash
python -m kbs.cli upload path\to\file.csv
```

- Search within that file
```bash
python -m kbs.cli search path\to\file.csv "your query"
```

- Show recent searches for that file
```bash
python -m kbs.cli recent path\to\file.csv
```

Indexes and recent searches are cached at `~/.kbs_cache`.

### Demo Script (5 min)
1. Show uploading a sample CSV/JSON/TXT/PDF and building index
2. Run 2-3 searches; highlight ranked results and title/content matching
3. Show `recent` remembering last 5 searches
4. Show no-result case and suggestions

### Notes
- CSV heuristic: first column or columns named `title|name|subject|key|id` used as title.
- JSON heuristic: array of objects preferred; falls back to wrapping primitives.
- PDF extraction uses PyPDF2; quality depends on the source PDF. 