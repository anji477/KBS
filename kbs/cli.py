import argparse
import os
from typing import List

from . import loaders
from .search import build_index, save_index, load_index, search, add_recent_search, get_recent_searches


SUGGESTIONS = [
	"Check spelling",
	"Try related terms",
]


def cmd_upload(args: argparse.Namespace) -> None:
	path = args.file
	if not os.path.exists(path):
		raise SystemExit(f"File not found: {path}")
	print(f"Loading: {path}")
	docs = loaders.load_file(path)
	print(f"Loaded {len(docs)} documents. Building index...")
	idx = build_index(docs)
	cache_path = save_index(path, idx)
	print(f"Index saved to: {cache_path}")
	recent = get_recent_searches(path)
	if recent:
		print("Recent searches:")
		for q in recent:
			print(f" - {q}")


def cmd_search(args: argparse.Namespace) -> None:
	path = args.file
	query = args.query
	try:
		idx = load_index(path)
	except FileNotFoundError as e:
		raise SystemExit(str(e))
	results = search(idx, query, top_k=args.top_k)
	add_recent_search(path, query)
	if not results or all(score <= 1e-9 for score, _ in results):
		print("No results found.")
		print("Suggestions:")
		for s in SUGGESTIONS:
			print(f" - {s}")
		return
	for rank, (score, doc) in enumerate(results, start=1):
		title = doc.get("title") or doc.get("id")
		preview = (doc.get("content") or "").strip().replace("\n", " ")
		if len(preview) > 160:
			preview = preview[:157] + "..."
		print(f"{rank}. {title}  [score={score:.4f}]")
		print(f"   {preview}")


def cmd_recent(args: argparse.Namespace) -> None:
	path = args.file
	recent = get_recent_searches(path)
	if not recent:
		print("No recent searches.")
		return
	print("Recent searches:")
	for q in recent:
		print(f" - {q}")


def build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(prog="kbs", description="Knowledge Base Search CLI")
	sub = p.add_subparsers(dest="command", required=True)

	p_upload = sub.add_parser("upload", help="Upload/build index for a file")
	p_upload.add_argument("file", help="Path to CSV/JSON/TXT/PDF file")
	p_upload.set_defaults(func=cmd_upload)

	p_search = sub.add_parser("search", help="Search within an uploaded file")
	p_search.add_argument("file", help="Path to previously uploaded file")
	p_search.add_argument("query", help="Search query")
	p_search.add_argument("--top-k", type=int, default=5, help="Number of results to show")
	p_search.set_defaults(func=cmd_search)

	p_recent = sub.add_parser("recent", help="Show recent searches for a file")
	p_recent.add_argument("file", help="Path to previously uploaded file")
	p_recent.set_defaults(func=cmd_recent)

	return p


def main(argv: List[str] | None = None) -> None:
	parser = build_parser()
	args = parser.parse_args(argv)
	args.func(args)


if __name__ == "__main__":
	main() 