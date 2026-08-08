# Chunk Embeddings: The Blocker, the Decision, and How to Ingest Them Locally

> **Read this if:** you're setting up the vector-search feature and the
> `ticker_news_chunk_embeddings` table is empty, or you hit
> *"All article URLs failed to fetch"* when running the ingestion notebook on
> Databricks.
>
> This is written as a walkthrough, not just a how-to. It explains **what broke,
> why, how we diagnosed it, the options we considered, why we chose the one we
> did, and how to do the other two** — so you can make the same call for
> yourself instead of just copying commands.

---

## 1. TL;DR

- The app's semantic search (`POST /api/search`) reads **only** from
  `ticker_news_chunk_embeddings`. If that table is empty, search returns nothing.
- Populating it requires **fetching each article's full body from its publisher's
  website**. Databricks **Free-edition Serverless blocks outbound traffic to
  arbitrary external domains**, and Free edition can't create a classic cluster
  that would be allowed to. So the chunk step fails there.
- **This is a networking problem, not a compute problem.** The embedding model
  itself runs fine on Serverless (the title/description embeddings work).
- **Fix:** run the fetch → chunk → embed → upsert pipeline **anywhere with normal
  internet**, writing vectors into Lakebase over its public Postgres endpoint.
  Your **laptop** does this for free. Colab and AWS are alternatives.
- Run it with:
  ```bash
  pip install -r scripts/requirements-ingest.txt
  python scripts/ingest_chunk_embeddings.py --create-table
  ```

---

## 2. Background: what "chunk embeddings" are and why the app needs them

The ingestion pipeline produces **two** kinds of vectors:

| Table | One vector per… | Source text | Used for |
|---|---|---|---|
| `ticker_news_embeddings` | article | title + description | coarse discovery |
| `ticker_news_chunk_embeddings` | ~800-char passage of the article body | full article body (fetched from `article_url`) | fine-grained RAG retrieval |

The chunk table is what makes retrieval *precise* — instead of matching a whole
article on its headline, you match the exact passage that answers the query.

**Crucially, the app's search endpoint only queries the chunk table.** See
[`app.py`](../app.py) — `vector_search()` runs:

```sql
SELECT ... , 1 - (embedding <=> %s::vector) AS similarity_score
FROM ticker_news_chunk_embeddings
ORDER BY embedding <=> %s::vector
LIMIT %s
```

No chunk rows ⇒ no search results, no matter how correct the endpoint is. That's
why this table is a hard dependency for the demo.

---

## 3. The blocker (what actually failed, and why)

Running `notebooks/ingest_ticker_news_embeddings.py` on Databricks Free edition:

- ✅ **Fetch news metadata** from the Massive API → works.
- ✅ **Title/description embeddings** (`sentence-transformers` on Serverless) → works.
- ❌ **Chunk embeddings** → the notebook prints:

  ```
  ⚠️ WARNING: All article URLs failed to fetch!
  This is expected on Databricks Serverless due to DNS resolution limitations.
  Serverless compute cannot resolve external domain names for article URLs.
  ```

### Why it fails

Building chunk embeddings needs the **article body**, which lives on the
publisher's site. The notebook does, per article:

```python
resp = requests.get(article_url, timeout=15)   # <-- outbound to a random domain
text = trafilatura.extract(resp.text)          #     strip nav/ads -> clean text
```

Databricks **Free-edition Serverless** does not allow outbound network access to
**arbitrary external domains**. (The Massive API call works because it goes to a
small set of known hosts; a random news publisher does not.) Every
`requests.get(article_url)` fails, `chunks_df` comes back empty, and there's
nothing to embed.

On **paid** Databricks you'd just run this on a **classic (non-serverless)
cluster**, which has normal egress. But **Free edition can't create one** — it's
serverless-only — so that escape hatch is closed here.

### The key diagnosis

> The model isn't the problem. **Network egress is.**

We know this because step (2), which uses the *same* `sentence-transformers`
model on the *same* Serverless compute, succeeds. The only thing step (3) adds is
**fetching bytes from external websites**. That's the single thing Serverless
won't let us do. So we don't need more/bigger compute — we need a machine that's
**allowed to reach the open web**.

---

## 4. The decision: decouple *where vectors are computed* from *where the app runs*

The insight that unlocks everything:

> An embedding is just a list of numbers in a Postgres row. **Nothing about it
> depends on the machine that produced it.** Same model + same input text ⇒ byte-
> identical vector, whether computed on Databricks, a laptop, Colab, or AWS.

And Lakebase Postgres is **reachable over the public internet** (that's exactly
how `python app.py` runs locally against it — see the README's "Run locally"
step). So we can compute the chunk vectors on **any** box that has:

1. **Outbound internet** — to fetch article bodies, and
2. **Network access to Lakebase** — `host:5432`, `sslmode=require`.

This splits vector search into two phases that run in different places:

```
                 INDEX TIME (offline, run anywhere)          QUERY TIME (in the deployed app)
                 ------------------------------------          --------------------------------
  article_url ->  fetch body -> chunk -> embed -+                user query -> embed -> pgvector
                                                 v                                        search
                            ticker_news_chunk_embeddings  <----  reads the vectors  -----'
                                    (Lakebase / pgvector)
```

- **Index time** needs open-web egress (the thing Serverless blocks). Run it on a
  machine that has it.
- **Query time** runs inside the Databricks App and needs **no** web fetch — it
  only embeds the short query string and runs SQL. That's why the app itself is
  perfectly happy on Databricks; it never does the thing that broke.

So the app stays deployed on Databricks for the demo. Only the **offline index-
building** step moves off-platform. The app can't even tell the difference.

---

## 5. The options we considered

| Option | Cost | Setup effort | Notes |
|---|---|---|---|
| **A. Local (your laptop)** ✅ chosen | Free | `pip install` + run one script | Laptop already has internet + Python + can reach Lakebase. Zero new accounts. |
| **B. Google Colab** | Free | Paste cells, mount secret, run | Cloud, notebook-shaped, doesn't tie up your machine. Good if your local network is restricted. |
| **C. AWS free trial** | Trial credits | New account, EC2/IAM/security groups, deploy | Works, but the most moving parts for the same result. Credits expire. |

**Why we chose Local (A):** it's the lowest-friction path that fully removes the
blocker. The blocker was "no open-web egress"; a laptop has open-web egress. It's
free, needs no new account, and the code is a near-copy of notebook cells that
already work. AWS solves the same problem but asks you to stand up a whole cloud
environment (account, IAM, EC2, security groups) to get a machine that has
internet and Python — which is what your laptop already is. Reserve AWS/Colab for
when your **local** network itself blocks outbound web traffic (some corporate
VPNs/proxies do).

---

## 6. Chosen approach — run it locally

### Prerequisites

1. **Python 3.10+** on your machine.
2. **A way to reach Lakebase.** Either:
   - Set `LAKEBASE_URL` in a local `.env` (a standard Postgres URL:
     `postgresql://role:password@host.database.cloud.databricks.com:5432/databricks_postgres?sslmode=require`),
     **or**
   - Have the **Databricks CLI configured** (the script then reads the URL from
     the `database/lakebase-url` secret, same as the app).
3. **The news documents table is already populated.** The chunk step reads
   `ticker_news_documents`. If it's empty, first run `POST /news/sync` on the app
   (or the notebook's news-fetch cells).

### Install & run

```bash
# from databricks-lakebase-app-day-2/
pip install -r scripts/requirements-ingest.txt

# first run: also create the pgvector extension + chunk table if missing
python scripts/ingest_chunk_embeddings.py --create-table
```

The first run downloads the MiniLM model (~90 MB) once. Expect the fetch stage to
log some per-article skips — paywalls, timeouts, and dead links are normal and
are skipped individually rather than failing the whole run.

### Handy flags

```bash
# smoke test: 20 articles, embed but DON'T write to the DB
python scripts/ingest_chunk_embeddings.py --limit-articles 20 --dry-run

# just one ticker
python scripts/ingest_chunk_embeddings.py --ticker AAPL

# be gentler / rougher on publishers (seconds between fetches)
python scripts/ingest_chunk_embeddings.py --fetch-delay 1.0
```

### Verify it worked

```sql
SELECT COUNT(*) FROM ticker_news_chunk_embeddings;         -- should be > 0
SELECT ticker, COUNT(*) FROM ticker_news_chunk_embeddings  -- spread by ticker
GROUP BY ticker ORDER BY 2 DESC;
```

Then open the deployed app's **`/search`** page and run a natural-language query
(e.g. *"latest AI chip demand"*). You should get ranked passages back.

### What this script improves over the notebook

- **Sends a browser-like `User-Agent`.** Many publishers `403` the default
  `python-requests` UA. Setting a real UA meaningfully raises the fetch success
  rate — a concrete reason you'll get *more* chunks locally than the notebook got.
- **Writes vectors directly as pgvector literals** (`'[...]'::vector`) instead of
  inserting a `double precision[]` array and casting in a second `UPDATE` pass.
  Fewer steps, and it matches how `app.py` reads/writes vectors.
- **Resolves dimension from the model** (`get_sentence_embedding_dimension()`), so
  the table's `VECTOR(N)` can never drift out of sync with the model.
- **Idempotent**: `ON CONFLICT (id) DO NOTHING`, so re-runs only add new chunks.

---

## 7. Alternative approach B — Google Colab

Use this when you'd rather run in the cloud (e.g. your local network blocks
outbound web traffic, or you don't want to tie up your laptop).

1. Open <https://colab.research.google.com> → **New notebook**.
2. Install deps in the first cell:
   ```python
   !pip install -q sentence-transformers trafilatura psycopg2-binary requests
   ```
3. Provide the Lakebase URL **without hard-coding it** — use Colab Secrets
   (🔑 icon in the left sidebar), add a secret named `LAKEBASE_URL`, then:
   ```python
   import os
   from google.colab import userdata
   os.environ["LAKEBASE_URL"] = userdata.get("LAKEBASE_URL")
   ```
4. Upload `scripts/ingest_chunk_embeddings.py` (Files pane) and run it, or paste
   its functions into cells:
   ```python
   !python ingest_chunk_embeddings.py --create-table
   ```

**Pros:** free, cloud, notebook-friendly, Colab has open internet.
**Cons:** session is ephemeral (re-install each time); you must get the Lakebase
URL into the runtime securely (Colab Secrets, not a pasted plaintext cell).

---

## 8. Alternative approach C — AWS free trial

Use this if you want a persistent, schedulable cloud box and are happy to manage
an AWS account. It works — it's just the heaviest option.

**Shape of it (EC2):**

1. Launch a small **EC2** instance (e.g. `t3.small`, Amazon Linux). The free tier
   covers a limited number of `t2.micro`/`t3.micro` hours — check current limits.
2. In the instance's **security group**, you only need *outbound* internet (the
   default allows it). No inbound ports needed beyond SSH for you.
3. SSH in, then:
   ```bash
   sudo yum install -y python3-pip git
   git clone <your repo>
   cd databricks-lakebase-app-day-2
   pip3 install -r scripts/requirements-ingest.txt
   export LAKEBASE_URL='postgresql://role:password@host...:5432/databricks_postgres?sslmode=require'
   python3 scripts/ingest_chunk_embeddings.py --create-table
   ```
4. To schedule it, add a `cron` entry or an **EventBridge + Lambda/Batch** job.

**Do you need Bedrock / SageMaker?** No. Bedrock would give you *embeddings via an
API*, but (a) you'd still have to fetch the article bodies, and (b) it changes the
model and vector dimension, so it wouldn't match the app's query-time model
unless you also switch the app. Plain EC2 running the same script is simpler and
keeps model parity.

**Pros:** persistent, schedulable, real cloud egress.
**Cons:** new account; IAM/security-group/EC2 setup; credits expire; the most
overhead for a job a laptop does for free. Watch for surprise costs after the
trial ends (stop/terminate the instance when done).

> ⚠️ **Security note for any off-platform run (Colab/AWS/local):** the
> `LAKEBASE_URL` contains a live database password. Keep it in an environment
> variable / secret store — never commit it, never paste it into a shared
> notebook cell, and rotate the Lakebase role's password if it's ever exposed.

---

## 9. How this fits the deployed app (nothing else changes)

- The app stays deployed on Databricks exactly as-is.
- `POST /api/search` embeds the query with the **same** model
  (`sentence-transformers/all-MiniLM-L6-v2`, 384-dim) used here — **model parity
  is required** or similarity scores are meaningless.
- Query-time embedding needs no web fetch, so the app runtime is unaffected by the
  Serverless egress limit that blocked ingestion.
- First `/search` after a deploy may take ~30–60s while the app container lazily
  downloads + loads the model; subsequent calls are fast.

### Built-in fallback (demo safety net)

`POST /api/search` now degrades gracefully: it searches the **passage-level chunk
table first**, and if that table is empty/unavailable it **falls back to the
article-level title/description table** (`ticker_news_embeddings`, which populates
fine even on Serverless). The JSON response carries a `search_source` field
(`"chunks"` or `"articles"`), and the `/search` UI shows a small note when it's
serving the article-level fallback.

This means the demo shows *something* even before you've run the local chunk
ingestion — but passage-level retrieval (the precise, RAG-quality results) only
appears once `ticker_news_chunk_embeddings` is populated by this script. So the
fallback is a safety net, **not** a substitute for running the ingestion.

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `No articles with URLs found` | `ticker_news_documents` is empty | Run `POST /news/sync` (or the notebook's news cells) first |
| **All** fetches fail | You're on a network that blocks outbound web (e.g. the same class of restriction as Serverless, or a corporate proxy) | Run from open internet, Colab, or AWS |
| Many `403` skips | Publisher blocking bots | Expected; the browser UA already mitigates. Lower `--fetch-delay` won't help; some sites just block |
| `could not resolve a Lakebase connection URL` | No `LAKEBASE_URL` and no Databricks CLI | Set `LAKEBASE_URL` in `.env`, or configure the Databricks CLI |
| `type "vector" does not exist` | pgvector extension not enabled | Re-run with `--create-table` (runs `CREATE EXTENSION IF NOT EXISTS vector`) |
| Search returns nothing but table has rows | Model mismatch between ingestion and app | Ensure both use `all-MiniLM-L6-v2` (384-dim) |
| Dimension mismatch on insert | Table `VECTOR(N)` ≠ model dim | Recreate the table at the model's dimension (script derives it automatically) |

---

## 11. One-paragraph summary for reviewers

The vector-search feature depends on `ticker_news_chunk_embeddings`, which is
populated by fetching article bodies from publisher websites. Databricks
Free-edition Serverless blocks outbound web egress (and Free edition can't spin
up a classic cluster that wouldn't), so that step can't run on-platform — while
the embedding model itself runs fine there, proving the blocker is **networking,
not compute**. Because embeddings are model-deterministic data and Lakebase is
publicly reachable, we compute the chunk vectors **off-platform** (locally by
default; Colab or AWS as alternatives) and write them straight into Lakebase. The
app stays deployed on Databricks and is unaffected, since its query-time search
embeds only the query string and never fetches the open web.
