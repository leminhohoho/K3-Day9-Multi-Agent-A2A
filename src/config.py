import os
from dotenv import load_dotenv

load_dotenv()

# Model — declared in code per spec (not in .env)
MODEL_NAME = "openai/gpt-4o-mini"  # ≤10B params, fast output generation
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]  # in .env only
MAX_TOKENS = 2048
TEMPERATURE = 0.1

# Disable Qwen3 thinking mode for speed (non-reasoning model behavior)
REASONING_EFFORT = "none"

# Policy
POLICY_VERSION = "EC_POLICY_V1"

# Paths
DATA_DIR = "data"
INPUT_DIR = "input"
OUTPUT_DIR = "output"
TRACE_FILE = "logging/trace.jsonl"
METADATA_FILE = "logging/metadata.json"

# Static metadata recorded in logging/metadata.json after each run
PARAMETER_SIZE = "~8B"
FRAMEWORK = "pure Python + OpenRouter API"
RUNTIME = "OpenRouter (OpenAI-compatible API)"
AGENTS = [
    "Coordinator",
    "OrderAgent",
    "PaymentAgent",
    "DeliveryAgent",
    "PolicyAgent",
    "VerifierAgent",
]
PATTERN = "Coordinator + Parallel fan-out (data agents) → Sequential Policy → Sequential Verifier"
EXECUTION = "LLM reasoning (Order/Payment) + deterministic policy/verifier + corrector"
