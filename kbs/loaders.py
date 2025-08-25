import json
import os
from typing import List, Dict, Any, Optional

import pandas as pd
from PyPDF2 import PdfReader


Document = Dict[str, Any]


def _safe_str(value: Any) -> str:
	if value is None:
		return ""
	if isinstance(value, (dict, list)):
		return json.dumps(value, ensure_ascii=False)
	return str(value)


def load_txt(path: str) -> List[Document]:
	with open(path, "r", encoding="utf-8", errors="ignore") as f:
		text = f.read()
	return [
		{
			"id": os.path.basename(path),
			"title": os.path.basename(path),
			"content": text,
		}
	]


def load_pdf(path: str) -> List[Document]:
	reader = PdfReader(path)
	texts: List[str] = []
	for page in reader.pages:
		try:
			texts.append(page.extract_text() or "")
		except Exception:
			texts.append("")
	text = "\n".join(texts)
	return [
		{
			"id": os.path.basename(path),
			"title": os.path.basename(path),
			"content": text,
		}
	]


def load_csv(path: str) -> List[Document]:
	df = pd.read_csv(path)
	return _rows_to_documents(df)


def load_json(path: str) -> List[Document]:
	with open(path, "r", encoding="utf-8", errors="ignore") as f:
		data = json.load(f)
	# If single object, make a list
	if isinstance(data, dict):
		data = [data]
	# If it's a list of primitives, wrap
	rows: List[Dict[str, Any]]
	if isinstance(data, list):
		if all(not isinstance(x, dict) for x in data):
			rows = [{"value": x} for x in data]
		else:
			rows = [x if isinstance(x, dict) else {"value": x} for x in data]
	else:
		rows = [{"value": data}]
	df = pd.DataFrame(rows)
	return _rows_to_documents(df)


def _rows_to_documents(df: pd.DataFrame) -> List[Document]:
	# Heuristics for title/key
	title_candidates = [
		"title",
		"name",
		"subject",
		"key",
		"id",
	]
	lower_cols = {c.lower(): c for c in df.columns}
	picked_title_col: Optional[str] = None
	for cand in title_candidates:
		if cand in lower_cols:
			picked_title_col = lower_cols[cand]
			break
	if picked_title_col is None and len(df.columns) > 0:
		picked_title_col = df.columns[0]

	docs: List[Document] = []
	for idx, row in df.iterrows():
		title_val = _safe_str(row.get(picked_title_col)) if picked_title_col else f"row_{idx}"
		content_parts: List[str] = []
		for col in df.columns:
			if col == picked_title_col:
				continue
			content_parts.append(f"{col}: {_safe_str(row.get(col))}")
		content = "\n".join(content_parts)
		docs.append({
			"id": f"row_{idx}",
			"title": title_val if title_val else f"row_{idx}",
			"content": content,
		})
	return docs


def load_file(path: str) -> List[Document]:
	ext = os.path.splitext(path)[1].lower()
	if ext == ".txt":
		return load_txt(path)
	if ext == ".pdf":
		return load_pdf(path)
	if ext == ".csv":
		return load_csv(path)
	if ext == ".json":
		return load_json(path)
	raise ValueError(f"Unsupported file type: {ext}") 