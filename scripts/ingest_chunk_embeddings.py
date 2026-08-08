"""
Local chunk-embeddings ingestion for the Massive + Lakebase Databricks App.

WHY THIS SCRIPT EXISTS
----------------------
The Spark notebook `notebooks/ingest_ticker_news_embeddings.py` computes two
kinds of embeddings and writes them into Lakebase (pgvector):

  1. Title/description embeddings  -> ticker_news_embeddings
  2. Full-article-body chunk embeddings -> ticker_news_chunk_embeddings

Step (1) works fine on Databricks Free-edition Serverless. Step (2) does NOT,
because building chunk embeddings requires fetching each article's *body* from
its publisher's website (requests.get(article_url) + trafilatura). Databricks
Free-edition Serverless blocks outbound traffic to arbitrary external domains,
and Free edition can't create a classic (non-serverless) cluster that would be
allowed to. So the chunk table ends up empty.

The app's semantic-search endpoint (`POST /api/search` in app.py) reads ONLY
from `ticker_news_chunk_embeddings`. An empty chunk table means search returns
nothing -- regardless of how good the endpoint is. So the chunk table MUST be
populated for the demo to work.

The fix is to decouple *where the embeddings are computed* from *where the app
runs*. The vectors are just rows in Postgres; nothing about them depends on the
machine that produced them. This script runs the exact same fetch -> chunk ->
embed -> upsert pipeline as the notebook, but from ANY machine that has:

  - outbound internet (to fetch article bodies), and
  - network access to the Lakebase Postgres endpoint (public over the internet).

Your laptop satisfies both, for free, with no new cloud account. See
`docs/chunk_embeddings_local_ingestion.md` for the full write-up of the
blockers, the decision, and the Colab / AWS alternatives.

USAGE
-----
    # from databricks-lakebase-app-day-2/
    pip install -r scripts/requirements-ingest.txt
    python scripts/ingest_chunk_embeddings.py

    # useful flags
    python scripts/ingest_chunk_embeddings.py --limit-articles 20 --dry-run
    python scripts/ingest_chunk_embeddings.py --ticker AAPL --create-table

CONNECTION
----------
The Lakebase connection URL is resolved in this order:
  1. LAKEBASE_URL env var (e.g. from a local .env) -- preferred for portability
     (Colab / AWS / any box without the Databricks CLI).
  2. Falls back to the Databricks secret scope via the same helper the app uses
     (lakebase._lakebase_url()), which needs the Databricks CLI configured.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone

import psycopg2
import requests
import trafilatura
from psycopg2.extras import execute_values


def _repair_ssl_env() -> None:
    """Work around a stale SSL_CERT_FILE / SSL_CERT_DIR.

    Some environments (notably conda activation scripts) export SSL_CERT_FILE
    or SSL_CERT_DIR pointing at a CA bundle that doesn't exist on disk. httpx
    (used by huggingface_hub to download the embedding model) and requests then
    crash with FileNotFoundError before making any network call. If the
    referenced path is missing, repoint to certifi's bundle so TLS works.
    """
    cert_file = os.environ.get("SSL_CERT_FILE")
    cert_dir = os.environ.get("SSL_CERT_DIR")
    stale = (cert_file and not os.path.isfile(cert_file)) or (
        cert_dir and not os.path.isdir(cert_dir)
    )
    if not stale:
        return
    try:
        import certifi

        os.environ["SSL_CERT_FILE"] = certifi.where()
        os.environ.pop("SSL_CERT_DIR", None)
    except ImportError:
        os.environ.pop("SSL_CERT_FILE", None)
        os.environ.pop("SSL_CERT_DIR", None)


# Must run before any TLS is set up (model download, article fetches).
_repair_ssl_env()

try:
    # Optional: only needed if you keep secrets in a local .env file.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv not installed -> rely on real env vars.
    pass

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s"
)
logger = logging.getLogger("ingest-chunks")


# ---------------------------------------------------------------------------
# Configuration (env vars mirror the app / notebook; CLI flags override them)
# ---------------------------------------------------------------------------
NEWS_TABLE_NAME = os.environ.get("NEWS_TABLE_NAME", "ticker_news_documents")
CHUNK_EMBEDDINGS_TABLE_NAME = os.environ.get(
    "CHUNK_EMBEDDINGS_TABLE_NAME", "ticker_news_chunk_embeddings"
)
EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))

# Many publishers 403 the default python-requests User-Agent. A browser-like UA
# meaningfully raises the fetch success rate -- this is one concrete improvement
# over the original notebook, which sent no UA and so lost more articles to 403s.
FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def resolve_lakebase_url() -> str:
    """Prefer LAKEBASE_URL (portable); fall back to the Databricks secret scope
    via the same helper the app uses. The fallback needs the Databricks CLI
    configured; the env-var path works anywhere (laptop, Colab, AWS)."""
    url = os.environ.get("LAKEBASE_URL")
    if url:
        logger.info("Using LAKEBASE_URL from environment.")
        return url

    logger.info(
        "LAKEBASE_URL not set; falling back to Databricks secret scope "
        "(requires the Databricks CLI to be configured)."
    )
    try:
        # Import lazily so the env-var path doesn't require databricks-sdk.
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import lakebase  # noqa: E402  (local module, resolved at runtime)

        return lakebase._lakebase_url()
    except Exception as exc:  # noqa: BLE001 -- surface a clear, actionable error
        logger.error(
            "Could not resolve a Lakebase connection URL. Set LAKEBASE_URL in "
            "your environment (or .env), or configure the Databricks CLI. "
            "Underlying error: %s",
            exc,
        )
        raise SystemExit(1)


def connect(lakebase_url: str):
    """Open a psycopg2 connection to Lakebase over the public Postgres endpoint."""
    conn = psycopg2.connect(lakebase_url, connect_timeout=15)
    return conn


def ensure_chunk_table(conn, embedding_dim: int) -> None:
    """Create the pgvector extension + chunk table if they don't exist yet.

    Mirrors sql/03_setup_chunk_embeddings_table.sql, but resolves the vector
    dimension from the loaded model so it can never drift out of sync.
    """
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {CHUNK_EMBEDDINGS_TABLE_NAME} (
                id TEXT PRIMARY KEY,
                article_id TEXT NOT NULL,
                ticker TEXT NOT NULL,
                chunk_index INT NOT NULL,
                chunk_text TEXT NOT NULL,
                embedding VECTOR({embedding_dim}) NOT NULL,
                model_name TEXT NOT NULL,
                embedded_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )
        conn.commit()
    logger.info("Ensured %s exists (VECTOR(%d)).", CHUNK_EMBEDDINGS_TABLE_NAME, embedding_dim)

    # The HNSW index is a search-speed optimization, not a correctness
    # requirement -- inserts and cosine search both work without it (just a
    # slower scan). Creating it requires OWNING the table, which fails if the
    # table was created earlier under a different role. So make it best-effort:
    # try in its own transaction and, on any DB error, roll back and continue.
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{CHUNK_EMBEDDINGS_TABLE_NAME}_embedding
                ON {CHUNK_EMBEDDINGS_TABLE_NAME}
                USING hnsw (embedding vector_cosine_ops);
                """
            )
            conn.commit()
        logger.info("Ensured HNSW index on %s.", CHUNK_EMBEDDINGS_TABLE_NAME)
    except psycopg2.Error as exc:
        conn.rollback()
        logger.warning(
            "Skipping HNSW index (not required for correctness): %s. "
            "If you own the table, create it once with sql/03_setup_chunk_embeddings_table.sql.",
            str(exc).strip(),
        )


def load_articles(conn, ticker: str | None, limit_articles: int | None) -> list[dict]:
    """Read articles that have a fetchable URL from ticker_news_documents."""
    sql = f"""
        SELECT id, ticker, article_url
        FROM {NEWS_TABLE_NAME}
        WHERE article_url IS NOT NULL AND article_url <> ''
    """
    params: list = []
    if ticker:
        sql += " AND ticker = %s"
        params.append(ticker.upper())
    sql += " ORDER BY published_utc DESC NULLS LAST"
    if limit_articles:
        sql += " LIMIT %s"
        params.append(limit_articles)

    with conn.cursor() as cur:
        cur.execute(sql, tuple(params))
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    logger.info("Loaded %d articles with URLs from %s.", len(rows), NEWS_TABLE_NAME)
    return rows


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping character windows (same scheme as the notebook)."""
    chunks: list[str] = []
    step = max(1, size - overlap)
    for start in range(0, len(text), step):
        piece = text[start : start + size].strip()
        if piece:
            chunks.append(piece)
        if start + size >= len(text):
            break
    return chunks


def fetch_and_chunk(
    articles: list[dict], size: int, overlap: int, delay: float
) -> tuple[list[dict], dict]:
    """Fetch each article body, extract clean text, and split into chunks.

    Returns (chunk_rows, stats). Failures (paywall, timeout, 403, dead link) are
    skipped per-article rather than failing the whole run.
    """
    session = requests.Session()
    session.headers.update(FETCH_HEADERS)

    chunk_rows: list[dict] = []
    stats = {"fetched": 0, "failed": 0, "empty": 0, "articles_with_chunks": 0}

    for i, article in enumerate(articles):
        if i > 0 and delay > 0:
            time.sleep(delay)  # be polite to publishers
        url = article["article_url"]
        try:
            resp = session.get(url, timeout=20)
            resp.raise_for_status()
            body = trafilatura.extract(resp.text)
        except Exception as exc:  # noqa: BLE001
            stats["failed"] += 1
            logger.warning("  [skip] %s (%s)", url[:80], exc)
            continue

        if not body:
            stats["empty"] += 1
            continue

        stats["fetched"] += 1
        pieces = chunk_text(body, size, overlap)
        if pieces:
            stats["articles_with_chunks"] += 1
        for idx, piece in enumerate(pieces):
            chunk_rows.append(
                {
                    "id": f"{article['id']}_{idx}",
                    "article_id": article["id"],
                    "ticker": article["ticker"],
                    "chunk_index": idx,
                    "chunk_text": piece,
                }
            )

        if (i + 1) % 10 == 0:
            logger.info(
                "  progress: %d/%d articles processed, %d chunks so far",
                i + 1,
                len(articles),
                len(chunk_rows),
            )

    return chunk_rows, stats


def embed_chunks(model, chunk_rows: list[dict], batch_size: int = 32) -> None:
    """Compute an embedding for each chunk in place (adds an 'embedding' key)."""
    texts = [c["chunk_text"] for c in chunk_rows]
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors = model.encode(batch, show_progress_bar=False)
        for row, vec in zip(chunk_rows[start : start + batch_size], vectors):
            row["embedding"] = vec.tolist()
        if (start + batch_size) % 128 == 0:
            logger.info("  embedded %d/%d chunks", min(start + batch_size, len(texts)), len(texts))


def upsert_chunks(conn, chunk_rows: list[dict], model_name: str) -> int:
    """Batch-upsert chunk vectors. Embeddings go in directly as pgvector literals
    ('[v1,v2,...]'::vector) -- no array-then-cast two-step needed, matching how
    app.py's search endpoint reads/writes vectors."""
    embedded_at = datetime.now(timezone.utc)
    values = [
        (
            r["id"],
            r["article_id"],
            r["ticker"],
            int(r["chunk_index"]),
            r["chunk_text"],
            "[" + ",".join(str(float(x)) for x in r["embedding"]) + "]",
            model_name,
            embedded_at,
        )
        for r in chunk_rows
    ]

    sql = f"""
        INSERT INTO {CHUNK_EMBEDDINGS_TABLE_NAME}
            (id, article_id, ticker, chunk_index, chunk_text, embedding, model_name, embedded_at)
        VALUES %s
        ON CONFLICT (id) DO NOTHING
    """
    template = "(%s, %s, %s, %s, %s, %s::vector, %s, %s)"

    with conn.cursor() as cur:
        execute_values(cur, sql, values, template=template, page_size=100)
        conn.commit()
        inserted = cur.rowcount
    return inserted


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ticker", help="Only ingest chunks for this one ticker (e.g. AAPL).")
    p.add_argument("--limit-articles", type=int, help="Cap the number of articles processed.")
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help=f"Chars per chunk (default {CHUNK_SIZE}).")
    p.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP, help=f"Overlap chars (default {CHUNK_OVERLAP}).")
    p.add_argument("--fetch-delay", type=float, default=0.5, help="Seconds to sleep between article fetches (default 0.5).")
    p.add_argument("--create-table", action="store_true", help="Create pgvector extension + chunk table if missing.")
    p.add_argument("--dry-run", action="store_true", help="Fetch + chunk + embed but do NOT write to Lakebase.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Import here so `--help` works without the (heavy) torch import.
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embedding_dim = model.get_sentence_embedding_dimension()
    logger.info("Model loaded -> %d-dim vectors.", embedding_dim)

    lakebase_url = resolve_lakebase_url()
    conn = connect(lakebase_url)
    try:
        if args.create_table:
            ensure_chunk_table(conn, embedding_dim)

        articles = load_articles(conn, args.ticker, args.limit_articles)
        if not articles:
            logger.warning(
                "No articles with URLs found in %s. Run POST /news/sync (or the "
                "notebook's news-fetch cells) first to populate it.",
                NEWS_TABLE_NAME,
            )
            return

        logger.info("Fetching + chunking article bodies (this hits publisher sites)...")
        chunk_rows, stats = fetch_and_chunk(
            articles, args.chunk_size, args.chunk_overlap, args.fetch_delay
        )
        logger.info(
            "Fetch summary: %d ok, %d failed, %d empty; %d chunks from %d articles.",
            stats["fetched"], stats["failed"], stats["empty"],
            len(chunk_rows), stats["articles_with_chunks"],
        )

        if not chunk_rows:
            logger.warning(
                "No chunks produced. If ALL fetches failed you may be on a network "
                "that blocks outbound web traffic -- that's the exact Serverless "
                "blocker this script exists to route around. Try from a normal "
                "internet connection, Colab, or AWS (see docs/)."
            )
            return

        logger.info("Embedding %d chunks...", len(chunk_rows))
        embed_chunks(model, chunk_rows)

        if args.dry_run:
            logger.info("[dry-run] Skipping write. Would upsert %d chunks.", len(chunk_rows))
            return

        logger.info("Upserting %d chunks into %s...", len(chunk_rows), CHUNK_EMBEDDINGS_TABLE_NAME)
        inserted = upsert_chunks(conn, chunk_rows, EMBEDDING_MODEL_NAME)
        logger.info(
            "Done. Inserted %d new chunks (%d were duplicates, skipped via ON CONFLICT).",
            inserted, len(chunk_rows) - inserted,
        )
        logger.info(
            "Verify with:  SELECT COUNT(*) FROM %s;  then try the app's /search page.",
            CHUNK_EMBEDDINGS_TABLE_NAME,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
