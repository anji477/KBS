import hashlib
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from rapidfuzz import fuzz


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