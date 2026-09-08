import os

# anchored to this file so scripts work from any directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CH_DIR = os.path.join(os.path.expanduser("~"), ".ch")
CHATS_SOURCE_DIR = os.path.join(CH_DIR, "tmp")
CACHE_DIR = os.path.join(CH_DIR, "index")

if not os.path.isdir(CH_DIR):
    raise SystemExit(
        "error: ~/.ch/ does not exist. Install Ch and configure local mode first: "
        "https://github.com/MehmetMHY/ch"
    )
if not os.path.isdir(CHATS_SOURCE_DIR):
    raise SystemExit(
        "error: ~/.ch/tmp/ does not exist. Install Ch and configure local mode first: "
        "https://github.com/MehmetMHY/ch"
    )

os.makedirs(CACHE_DIR, exist_ok=True)

# scratch space for retrieve's /view command; files are deleted right after use
TMP_DIR = os.path.join(CACHE_DIR, "tmp")
os.makedirs(TMP_DIR, exist_ok=True)

# paths
DB_PATH = os.path.join(CACHE_DIR, "chats.db")
EMBEDDINGS_CACHE_PATH = os.path.join(CACHE_DIR, "embeddings_cache.npz")
PRICING_CACHE_PATH = os.path.join(CACHE_DIR, "pricing_cache.json")
# persistent readline history for the main.py REPL prompt (up/down arrow recall)
QUERY_HISTORY_PATH = os.path.join(CACHE_DIR, "query_history")
QUERY_HISTORY_MAX = 1000

# models. embeddings define the stored vector space and cannot change without a
# full re-embed. retrieval LLM steps use OpenAI for rerank and query expansion.
SUMMARY_MODEL = "gpt-5.4-nano"  # process.py (OpenAI)
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI, defines the vector space
RERANK_MODEL = "gpt-5.6-luna"
RERANK_EFFORT = "high"
QUERY_EXPANSION_MODEL = "gpt-5.6-luna"
QUERY_EXPANSION_EFFORT = "medium"

# Which models.dev provider serves each model. Used by pricing.py to look up
# the right entry in the catalog (model ids are not unique across providers).
MODEL_PROVIDER = {
    SUMMARY_MODEL: "openai",
    EMBEDDING_MODEL: "openai",
    RERANK_MODEL: "openai",
    QUERY_EXPANSION_MODEL: "openai",
}

# Pricing is fetched from https://models.dev/api.json and cached locally at
# PRICING_CACHE_PATH (see pricing.py). TTL is a soft upper bound: a lookup
# also refreshes when a model is missing or its entry looks broken, regardless
# of age. Prices change rarely, so a few days is plenty.
PRICING_TTL = 6 * 86_400  # 6 days, in seconds


# processing
MAX_INPUT_CHARS = 900_000
COMMIT_EVERY = 25
PRINT_EVERY = 10
DEFAULT_WORKERS = 64

# retrieval
POOL = 30
RERANK_POOL = 10
TOP_K = 5
RRF_K = 60
# display width for a result's blurb; sized so ~90% of short_summary values
# print in full rather than being cut with an ellipsis
PREVIEW_CHARS = 400
# number of extra query variants the LLM generates before fusion (0 disables)
NUM_EXPANSIONS = 3
# /ls preview: how many of the most recent chats to precompute in parallel
# before fzf opens, and the max transcript chars to render per preview.
PREVIEW_BATCH = 500
PREVIEW_LIMIT = 5000
# selectable rolling time windows for the /time filter, in seconds from now.
# order is preserved in the fzf picker; months/years are approximate.
TIME_RANGES = {
    "1d": 86_400,
    "3d": 3 * 86_400,
    "1w": 7 * 86_400,
    "1m": 30 * 86_400,
    "1y": 365 * 86_400,
}
