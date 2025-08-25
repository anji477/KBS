import os
import tempfile
from typing import List, Dict, Any

import streamlit as st

from kbs import loaders
from kbs.search import build_index, save_index, load_index, search, search_hybrid, add_recent_search, get_recent_searches


st.set_page_config(page_title="KB Search", page_icon="🔎", layout="centered")

st.title("Knowledge Base Search")

uploaded_file = st.file_uploader("Upload a CSV / JSON / TXT / PDF", type=["csv", "json", "txt", "pdf"])

if uploaded_file is not None:
	# Save uploaded file to a temp path to reuse existing loader/index code
	suffix = os.path.splitext(uploaded_file.name)[1]
	with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
		tmp.write(uploaded_file.getbuffer())
		tmp_path = tmp.name
	st.session_state["uploaded_path"] = tmp_path
	st.success(f"Uploaded: {uploaded_file.name}")

if "uploaded_path" in st.session_state:
	file_path = st.session_state["uploaded_path"]
	if st.button("Build Index"):
		docs = loaders.load_file(file_path)
		idx = build_index(docs)
		save_index(file_path, idx)
		st.success("Index built and cached.")

	st.subheader("Search")
	query = st.text_input("Enter your search query")
	employee_filter = st.text_input("Filter by employee (optional)")
	employee_id_filter = st.text_input("Filter by Employee ID (e.g., E018)")
	use_ai = st.toggle("AI-enhanced search", value=True, help="Blend TF-IDF with fuzzy matching for better recall")
	if st.button("Search") and query.strip():
		try:
			idx = load_index(file_path)
		except FileNotFoundError:
			# Auto-build index if missing
			docs = loaders.load_file(file_path)
			idx = build_index(docs)
			save_index(file_path, idx)
			st.info("Index was missing and has been built automatically.")
		results = (search_hybrid(idx, query, top_k=5) if use_ai else search(idx, query, top_k=5))
		if employee_filter.strip():
			needle = employee_filter.strip().lower()
			results = [
				(score, doc)
				for score, doc in results
				if needle in (doc.get("title") or "").lower() or needle in (doc.get("content") or "").lower()
			]
		# Strict post-filter by Employee ID if provided (matches 'employee_id: E###' in content)
		if employee_id_filter.strip():
			needle_id = employee_id_filter.strip().lower()
			results = [
				(score, doc)
				for score, doc in results
				if f"employee_id: {needle_id}" in (doc.get("content") or "").lower()
			]
		add_recent_search(file_path, query)
		if not results or all(score <= 1e-9 for score, _ in results):
			st.warning("No results found. Suggestions: Check spelling, Try related terms")
		else:
			for score, doc in results:
				st.markdown(f"**{doc.get('title') or doc.get('id')}**  ")
				preview = (doc.get("content") or "").strip().replace("\n", " ")
				if len(preview) > 300:
					preview = preview[:297] + "..."
				st.caption(f"score={score:.4f}")
				st.write(preview)
				st.divider()

	st.subheader("Recent searches")
	recents = get_recent_searches(file_path)
	if recents:
		st.write(recents)
	else:
		st.caption("No recent searches yet.")

st.caption("Indexes and recents are cached in ~/.kbs_cache")
