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
