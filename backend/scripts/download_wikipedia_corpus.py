"""Download a public Wikipedia knowledge corpus (CC BY-SA) for Groundline RAG."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httpx

WIKI_TITLES = [
    "Artificial_intelligence",
    "Machine_learning",
    "Deep_learning",
    "Neural_network_(machine_learning)",
    "Large_language_model",
    "Generative_artificial_intelligence",
    "Transformer_(deep_learning_architecture)",
    "Attention_(machine_learning)",
    "Reinforcement_learning",
    "Supervised_learning",
    "Unsupervised_learning",
    "Natural_language_processing",
    "Computer_vision",
    "Retrieval-augmented_generation",
    "Prompt_engineering",
    "Hallucination_(artificial_intelligence)",
    "Foundation_model",
    "Multimodal_learning",
    "Word_embedding",
    "Vector_database",
    "Algorithm",
    "Data_structure",
    "Database",
    "Knowledge_graph",
    "Semantic_search",
    "Information_retrieval",
    "Search_engine",
    "Distributed_computing",
    "Cloud_computing",
    "Operating_system",
    "Computer_network",
    "Cybersecurity",
    "Cryptography",
    "Blockchain",
    "Statistics",
    "Probability",
    "Bayes'_theorem",
    "Linear_algebra",
    "Gradient_descent",
    "Overfitting",
    "Cross-validation_(statistics)",
    "Feature_engineering",
    "Dimensionality_reduction",
    "Cluster_analysis",
    "Recommender_system",
    "Scientific_method",
    "Physics",
    "Quantum_computing",
    "Biology",
    "Genetics",
    "Climate_change",
    "Astronomy",
    "Medicine",
    "Economics",
    "Philosophy_of_mind",
    "Cognitive_science",
    "Human-computer_interaction",
    "Software_engineering",
    "Open-source_software",
    "Wikipedia",
    "Open_data",
    "Python_(programming_language)",
    "JavaScript",
    "SQL",
    "Application_programming_interface",
    "Microservices",
    "Docker_(software)",
    "Kubernetes",
    "Git",
    "DevOps",
    "Data_science",
]

USER_AGENT = "GroundlineRAG/1.0 (educational; https://github.com/local/groundline)"
API = "https://en.wikipedia.org/w/api.php"


def _safe_name(title: str) -> str:
    return re.sub(r"[^\w\-]+", "_", title).strip("_")


def fetch_batch(client: httpx.Client, titles: list[str]) -> list[dict]:
    params = {
        "action": "query",
        "prop": "extracts|info",
        "explaintext": 1,
        "exsectionformat": "plain",
        "redirects": 1,
        "titles": "|".join(titles),
        "format": "json",
        "formatversion": 2,
    }
    for attempt in range(6):
        resp = client.get(API, params=params)
        if resp.status_code == 429:
            wait = 2 ** attempt + 1
            print(f"  rate-limited; sleeping {wait}s…")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", [])
        out: list[dict] = []
        for page in pages:
            if page.get("missing"):
                continue
            text = (page.get("extract") or "").strip()
            if len(text) < 400:
                continue
            if len(text) > 12000:
                text = text[:12000] + "\n\n[Article truncated for corpus size.]"
            title = page.get("title", "Unknown")
            wiki_slug = title.replace(" ", "_")
            out.append(
                {
                    "title": title,
                    "pageid": page.get("pageid"),
                    "filename": f"wikipedia_{_safe_name(title)}.txt",
                    "text": (
                        f"Title: {title}\n"
                        f"Source: https://en.wikipedia.org/wiki/{wiki_slug}\n"
                        f"License: Creative Commons Attribution-ShareAlike (Wikipedia)\n\n"
                        f"{text}"
                    ),
                    "chars": len(text),
                }
            )
        return out
    return []


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "corpus" / "wikipedia"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "manifest.json"

    existing = {p.name for p in out_dir.glob("wikipedia_*.txt")}
    pending = []
    for title in WIKI_TITLES:
        # Rough pre-check; final filename uses resolved page title.
        guess = f"wikipedia_{_safe_name(title.replace('_', ' '))}.txt"
        # Always attempt missing titles; resume skips exact existing filenames after resolve.
        pending.append(title)

    articles: list[dict] = []
    # Keep already downloaded
    for path in sorted(out_dir.glob("wikipedia_*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        title_line = text.splitlines()[0] if text else path.stem
        title = title_line.replace("Title: ", "", 1) if title_line.startswith("Title:") else path.stem
        articles.append(
            {
                "title": title,
                "filename": path.name,
                "chars": len(text),
                "pageid": None,
                "path": str(path),
            }
        )

    with httpx.Client(timeout=90.0, headers={"User-Agent": USER_AGENT}) as client:
        batch_size = 5
        for i in range(0, len(pending), batch_size):
            batch = pending[i : i + batch_size]
            print(f"Fetching batch {i // batch_size + 1}: {batch}")
            try:
                fetched = fetch_batch(client, batch)
            except Exception as exc:  # noqa: BLE001
                print(f"  batch failed: {exc}")
                time.sleep(3)
                continue
            for article in fetched:
                path = out_dir / article["filename"]
                if path.exists():
                    continue
                path.write_text(article["text"], encoding="utf-8")
                articles.append(
                    {
                        "title": article["title"],
                        "filename": article["filename"],
                        "chars": article["chars"],
                        "pageid": article["pageid"],
                        "path": str(path),
                    }
                )
                print(f"  saved {article['title']} ({article['chars']} chars)")
            time.sleep(1.2)

    # Deduplicate by filename
    by_name = {a["filename"]: a for a in articles}
    unique = list(by_name.values())
    meta_path.write_text(
        json.dumps({"source": "Wikipedia", "license": "CC BY-SA", "count": len(unique), "articles": unique}, indent=2),
        encoding="utf-8",
    )
    print(f"Done: {len(unique)} articles → {out_dir}")


if __name__ == "__main__":
    main()
