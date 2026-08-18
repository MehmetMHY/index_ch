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

# scratch space for retrieve.py's /view command; files are deleted right after use
TMP_DIR = os.path.join(CACHE_DIR, "tmp")
os.makedirs(TMP_DIR, exist_ok=True)

# paths
DB_PATH = os.path.join(CACHE_DIR, "chats.db")
EMBEDDINGS_CACHE_PATH = os.path.join(CACHE_DIR, "embeddings_cache.npz")

# models. build.py/process.py stay on OpenAI (the stored embeddings define the
# vector space and cannot change provider). retrieve.py's two LLM steps (rerank,
# query expansion) run on Groq for speed and cost; it reaches Groq through the
# OpenAI-compatible endpoint below. embeddings always stay on OpenAI.
SUMMARY_MODEL = "gpt-5.4-nano"  # process.py (OpenAI)
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI, defines the vector space
GROQ_BASE_URL = "https://api.groq.com/openai/v1"  # needs GROQ_API_KEY
RERANK_MODEL = "openai/gpt-oss-120b"  # Groq, stronger ranking quality
RERANK_EFFORT = "low"
QUERY_EXPANSION_MODEL = "openai/gpt-oss-20b"  # Groq, a simple rewrite
QUERY_EXPANSION_EFFORT = "low"

# Pricing per model as (input, output) in USD per 1M tokens. Used only for the
# cost estimates the scripts print; update these if a model or its price changes.
# Embeddings have no output tokens, so their output price is 0.
# OpenAI: https://openai.com/business/pricing/  Groq: https://groq.com/pricing
PRICING = {
    SUMMARY_MODEL: (0.20, 1.25),
    EMBEDDING_MODEL: (0.02, 0.0),
    "openai/gpt-oss-120b": (0.15, 0.60),  # Groq rerank
    "openai/gpt-oss-20b": (0.075, 0.30),  # Groq query expansion
}


def estimate_cost(model, input_tokens=0, output_tokens=0):
    price_in, price_out = PRICING[model]
    return input_tokens / 1e6 * price_in + output_tokens / 1e6 * price_out


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
