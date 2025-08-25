import hashlib
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from rapidfuzz import fuzz
import re


@dataclass
class KBIndex:
	vectorizer: TfidfVectorizer
	matrix: Any
	documents: List[Dict[str, Any]]


CACHE_DIR = os.path.join(os.path.expanduser("~"), ".kbs_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _hash_path(path: str) -> str:
	return hashlib.sha256(os.path.abspath(path).encode("utf-8")).hexdigest()[:16]


def build_index(documents: List[Dict[str, Any]]) -> KBIndex:
	# Combine title and content, give slight weight to title via token duplication prefix
	corpus: List[str] = []
	for d in documents:
		title = (d.get("title") or "")
		content = (d.get("content") or "")
		combined = ((title + " ") * 2) + content
		corpus.append(combined)
	vectorizer = TfidfVectorizer(stop_words="english")
	matrix = vectorizer.fit_transform(corpus)
	return KBIndex(vectorizer=vectorizer, matrix=matrix, documents=documents)


def save_index(file_path: str, index: KBIndex) -> str:
	key = _hash_path(file_path)
	pkl_path = os.path.join(CACHE_DIR, f"{key}.json")
	# Store as JSON for portability (vectorizer vocab + idf + docs + matrix CSR)
	from scipy.sparse import csr_matrix
	import numpy as np
	mat: csr_matrix = index.matrix.tocsr()
	payload = {
		"vectorizer": {
			"vocabulary_": index.vectorizer.vocabulary_,
			"idf_": index.vectorizer.idf_.tolist(),
			"stop_words": "english",
		},
		"matrix": {
			"data": mat.data.tolist(),
			"indices": mat.indices.tolist(),
			"indptr": mat.indptr.tolist(),
			"shape": list(mat.shape),
		},
		"documents": index.documents,
	}
	with open(pkl_path, "w", encoding="utf-8") as f:
		json.dump(payload, f)
	return pkl_path


def load_index(file_path: str) -> KBIndex:
	key = _hash_path(file_path)
	pkl_path = os.path.join(CACHE_DIR, f"{key}.json")
	if not os.path.exists(pkl_path):
		raise FileNotFoundError("Index not found. Please upload/build first.")
	with open(pkl_path, "r", encoding="utf-8") as f:
		payload = json.load(f)
	from scipy.sparse import csr_matrix
	import numpy as np
	vectorizer = TfidfVectorizer(stop_words=payload["vectorizer"]["stop_words"])  # type: ignore
	vectorizer.vocabulary_ = {k: int(v) for k, v in payload["vectorizer"]["vocabulary_"].items()}  # type: ignore
	vectorizer.idf_ = np.array(payload["vectorizer"]["idf_"])  # type: ignore
	mat = csr_matrix((
		payload["matrix"]["data"],
		payload["matrix"]["indices"],
		payload["matrix"]["indptr"],
	), shape=tuple(payload["matrix"]["shape"]))
	documents = payload["documents"]
	return KBIndex(vectorizer=vectorizer, matrix=mat, documents=documents)


def search(index: KBIndex, query: str, top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
	if not query.strip():
		return []
	q_vec = index.vectorizer.transform([query])
	scores = linear_kernel(q_vec, index.matrix).flatten()
	ranked_indices = scores.argsort()[::-1][:top_k]
	results: List[Tuple[float, Dict[str, Any]]] = []
	for i in ranked_indices:
		results.append((float(scores[i]), index.documents[int(i)]))
	return results


def _normalize(values: List[float]) -> List[float]:
	if not values:
		return values
	v_min = min(values)
	v_max = max(values)
	if v_max == v_min:
		return [0.0 for _ in values]
	return [(v - v_min) / (v_max - v_min) for v in values]


def search_hybrid(index: KBIndex, query: str, top_k: int = 5, tfidf_weight: float = 0.7) -> List[Tuple[float, Dict[str, Any]]]:
	"""Hybrid search: TF-IDF cosine + fuzzy string match on title/content.

	- Computes TF-IDF cosine similarity for all docs
	- Computes RapidFuzz token_set_ratio against title and content (max of both)
	- Normalizes both to [0,1] and blends with tfidf_weight
	"""
	if not query.strip():
		return []

	# TF-IDF scores
	q_vec = index.vectorizer.transform([query])
	tfidf_scores_list = linear_kernel(q_vec, index.matrix).flatten().tolist()

	# Fuzzy scores (0..100) -> normalize to 0..1 later
	fuzzy_raw: List[float] = []
	for doc in index.documents:
		title = (doc.get("title") or "").strip()
		content = (doc.get("content") or "").strip()
		# Prefer title match if present
		score_title = fuzz.token_set_ratio(query, title) if title else 0
		# Use a cheaper partial ratio on content to avoid heavy cost
		score_content = fuzz.partial_ratio(query, content[:1000]) if content else 0
		fuzzy_raw.append(float(max(score_title, score_content)))

	# Normalize both vectors to [0,1]
	tfidf_norm = _normalize(tfidf_scores_list)
	fuzzy_norm = _normalize([v / 100.0 for v in fuzzy_raw])

	# Blend
	alpha = max(0.0, min(1.0, tfidf_weight))
	blended: List[float] = [
		(alpha * t) + ((1.0 - alpha) * f)
		for t, f in zip(tfidf_norm, fuzzy_norm)
	]

	# Rank and return
	ranked_indices = sorted(range(len(blended)), key=lambda i: blended[i], reverse=True)[:top_k]
	results: List[Tuple[float, Dict[str, Any]]] = []
	for i in ranked_indices:
		results.append((float(blended[i]), index.documents[int(i)]))
	return results


# ---- Smart query parsing and filtered ranking ----
_FIELD_ALIASES = {
	"employee_id": ["employee_id", "id", "emp_id"],
	"name": ["name", "employee", "employee_name"],
	"department": ["department", "dept"],
	"role": ["role", "title", "position"],
	"last_leave_date": ["last_leave_date", "last_leave", "leave_date"],
	"leave_balance_days": ["leave_balance_days", "leave_balance", "balance"],
	"total_leaves_available_days": ["total_leaves_available_days", "leaves_available", "available"],
	"total_leaves_taken_days": ["total_leaves_taken_days", "leaves_taken", "taken"],
}


def _match_field_name(token_key: str) -> str:
	key = token_key.strip().lower()
	for canon, aliases in _FIELD_ALIASES.items():
		if key == canon or key in aliases:
			return canon
	return key


_FIELD_REGEX_CACHE: Dict[str, re.Pattern] = {}


def _extract_field_from_content(doc: Dict[str, Any], field: str) -> str:
	"""Extract a field value from the doc content lines like 'field: value'."""
	pattern = _FIELD_REGEX_CACHE.get(field)
	if pattern is None:
		pattern = re.compile(rf"^\s*{re.escape(field)}\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE)
		_FIELD_REGEX_CACHE[field] = pattern
	content = (doc.get("content") or "")
	m = pattern.search(content)
	return (m.group(1).strip() if m else "")


def _doc_matches_filters(doc: Dict[str, Any], filters: List[Tuple[str, str, str]]) -> bool:
	"""filters: list of (field, op, value); op in {eq, contains, >, >=, <, <=}"""
	for field, op, value in filters:
		field_l = field.lower()
		if field_l == "name":
			doc_val = (doc.get("title") or "").strip()
		else:
			doc_val = _extract_field_from_content(doc, field)
		if op in ("eq", "contains"):
			needle = value.lower()
			if op == "eq":
				if doc_val.lower() != needle:
					return False
			else:
				if needle not in doc_val.lower():
					return False
		else:
			# numeric comparison if possible
			try:
				doc_num = float(re.findall(r"[-+]?[0-9]*\.?[0-9]+", doc_val)[0]) if doc_val else float("nan")
				val_num = float(value)
			except Exception:
				return False
			if op == ">" and not (doc_num > val_num):
				return False
			if op == ">=" and not (doc_num >= val_num):
				return False
			if op == "<" and not (doc_num < val_num):
				return False
			if op == "<=" and not (doc_num <= val_num):
				return False
	return True


def _parse_smart_query(query: str) -> Tuple[str, List[Tuple[str, str, str]]]:
	"""Return (free_text, filters). Supports tokens like:
	- employee_id:E018 (eq)
	- department:Platform (contains)
	- role:"API Engineer" (contains with quotes)
	- leave_balance_days>=10 (numeric comparisons)
	Other text becomes free_text.
	"""
	filters: List[Tuple[str, str, str]] = []
	free_parts: List[str] = []
	# token patterns
	pattern = re.compile(r"(\w+)(:|>=|<=|>|<)(\S+|\"[^\"]+\")")
	used_spans: List[Tuple[int, int]] = []
	for m in pattern.finditer(query):
		key_raw, op_raw, val_raw = m.group(1), m.group(2), m.group(3)
		field = _match_field_name(key_raw)
		value = val_raw.strip().strip('"')
		if op_raw == ":":
			# contains by default; if value is quoted and exact match requested by user, they can type name:"Alice Johnson"
			op = "contains"
		elif op_raw in (">=", "<=", ">", "<"):
			op = op_raw
		else:
			op = "contains"
		filters.append((field, op, value))
		used_spans.append(m.span())
	# collect remaining text as free text
	last = 0
	for start, end in used_spans:
		if last < start:
			free_parts.append(query[last:start].strip())
		last = end
	if last < len(query):
		free_parts.append(query[last:].strip())
	free_text = " ".join([p for p in free_parts if p])
	return free_text, filters


def search_smart(index: KBIndex, query: str, top_k: int = 5, tfidf_weight: float = 0.7) -> List[Tuple[float, Dict[str, Any]]]:
	free_text, filters = _parse_smart_query(query)
	base_query = free_text if free_text else query
	# compute hybrid scores against entire corpus
	q_vec = index.vectorizer.transform([base_query])
	tfidf_scores_list = linear_kernel(q_vec, index.matrix).flatten().tolist()
	# fuzzy
	fuzzy_raw: List[float] = []
	for doc in index.documents:
		title = (doc.get("title") or "").strip()
		content = (doc.get("content") or "").strip()
		score_title = fuzz.token_set_ratio(base_query, title) if title else 0
		score_content = fuzz.partial_ratio(base_query, content[:1000]) if content else 0
		fuzzy_raw.append(float(max(score_title, score_content)))
	# normalize and blend
	tfidf_norm = _normalize(tfidf_scores_list)
	fuzzy_norm = _normalize([v / 100.0 for v in fuzzy_raw])
	alpha = max(0.0, min(1.0, tfidf_weight))
	blended: List[float] = [
		(alpha * t) + ((1.0 - alpha) * f)
		for t, f in zip(tfidf_norm, fuzzy_norm)
	]
	# apply strict filters mask
	masked_scores: List[float] = []
	for score, doc in zip(blended, index.documents):
		if filters and not _doc_matches_filters(doc, filters):
			masked_scores.append(float("-inf"))
		else:
			masked_scores.append(score)
	# rank
	ranked_indices = sorted(range(len(masked_scores)), key=lambda i: masked_scores[i], reverse=True)[:top_k]
	results: List[Tuple[float, Dict[str, Any]]] = []
	for i in ranked_indices:
		if masked_scores[i] == float("-inf"):
			continue
		results.append((float(masked_scores[i]), index.documents[int(i)]))
	return results


def recent_searches_path(file_path: str) -> str:
	key = _hash_path(file_path)
	return os.path.join(CACHE_DIR, f"{key}_recent.json")


def add_recent_search(file_path: str, query: str) -> None:
	path = recent_searches_path(file_path)
	if os.path.exists(path):
		with open(path, "r", encoding="utf-8") as f:
			data = json.load(f)
		recent: List[str] = data.get("recent", [])
	else:
		recent = []
	recent = [q for q in recent if q != query]
	recent.insert(0, query)
	recent = recent[:5]
	with open(path, "w", encoding="utf-8") as f:
		json.dump({"recent": recent}, f)


def get_recent_searches(file_path: str) -> List[str]:
	path = recent_searches_path(file_path)
	if not os.path.exists(path):
		return []
	with open(path, "r", encoding="utf-8") as f:
		data = json.load(f)
	return data.get("recent", []) 