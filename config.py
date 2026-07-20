import os

# anchored to this file so scripts work from any directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# all generated data lives here
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# scratch space for retrieve.py's /view command; files are deleted right after use
TMP_DIR = os.path.join(CACHE_DIR, "tmp")
os.makedirs(TMP_DIR, exist_ok=True)

# paths
CHATS_SOURCE_DIR = os.path.join(os.path.expanduser("~"), ".ch/tmp/")
DB_PATH = os.path.join(CACHE_DIR, "chats.db")
EMBEDDINGS_CACHE_PATH = os.path.join(CACHE_DIR, "embeddings_cache.npz")

# models
SUMMARY_MODEL = "gpt-5.4-nano"
EMBEDDING_MODEL = "text-embedding-3-small"
RERANK_MODEL = "gpt-5.6-luna"
RERANK_EFFORT = "low"
# query expansion reuses the cheap summary model (same pricing key); a simple
# rewrite needs no reasoning, so effort is off to keep it fast
QUERY_EXPANSION_MODEL = SUMMARY_MODEL
QUERY_EXPANSION_EFFORT = "none"

# Pricing per model as (input, output) in USD per 1M tokens. Used only for the
# cost estimates the scripts print; update these if a model or its price changes.
# Embeddings have no output tokens, so their output price is 0.
# Current prices: https://openai.com/business/pricing/
PRICING = {
    SUMMARY_MODEL: (0.20, 1.25),
    EMBEDDING_MODEL: (0.02, 0.0),
    RERANK_MODEL: (1.00, 6.00),
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
RERANK_POOL = 15
TOP_K = 5
RRF_K = 60
PREVIEW_CHARS = 200
# number of extra query variants the LLM generates before fusion (0 disables)
NUM_EXPANSIONS = 3
# selectable rolling time windows for the /time filter, in seconds from now.
# order is preserved in the fzf picker; months/years are approximate.
TIME_RANGES = {
    "1d": 86_400,
    "3d": 3 * 86_400,
    "1w": 7 * 86_400,
    "1m": 30 * 86_400,
    "1y": 365 * 86_400,
}
