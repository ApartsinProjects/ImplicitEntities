"""
OpenRouter async client with rate limiting, cost tracking, and cost optimization.

Cost reduction features:
  - `:floor` suffix routes to cheapest provider for any model
  - `:free` suffix uses free tier (20 req/min, 200 req/day limit)
  - `provider.sort: "price"` equivalent to :floor
  - `provider.max_price` caps per-token cost
  - `service_tier: "flex"` gives ~50% off OpenAI reasoning models
  - Prompt caching is automatic (50-75% savings on repeated system prompts)

Usage:
    from openrouter_client import batch_call, cheapest_model, FREE_MODELS

    # Use free model (rate limited)
    results = await batch_call(prompts, model="deepseek/deepseek-chat-v3-0324:free", concurrency=5)

    # Use floor pricing (cheapest provider)
    results = await batch_call(prompts, model="google/gemini-2.0-flash-001:floor")

    # Use flex tier for OpenAI (50% off)
    results = await batch_call(prompts, model="openai/gpt-4o-mini", provider_opts={"service_tier": "flex"})
"""
import asyncio
import aiohttp
import json
import time
from pathlib import Path
from dataclasses import dataclass, field

API_KEY_PATH = Path(__file__).parent.parent / "openRouter.key.txt"
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# ── Cost-optimized model presets ──────────────────────────────────────────
# Free models (rate limited: ~20 req/min, 200 req/day)
FREE_MODELS = [
    "deepseek/deepseek-chat-v3-0324:free",
    "deepseek/deepseek-r1:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-3-27b-it:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "qwen/qwen3-32b:free",
]

# Cheapest paid models (with :floor for cheapest provider routing)
CHEAP_MODELS = [
    "google/gemini-2.0-flash-001:floor",          # ~$0.10/M input
    "meta-llama/llama-3.1-8b-instruct:floor",     # ~$0.05/M input
    "mistralai/mistral-7b-instruct:floor",         # ~$0.05/M input
    "google/gemini-2.0-flash-001",                 # ~$0.10/M input (default provider)
]

# Moderate models for quality comparison
MODERATE_MODELS = [
    "anthropic/claude-3.5-haiku:floor",
    "openai/gpt-4o-mini:floor",
    "google/gemini-2.5-flash-preview-05-20:floor",
]

# Default: cheapest reasonable quality
DEFAULT_MODEL = "google/gemini-2.0-flash-001:floor"


def cheapest_model(quality: str = "low") -> str:
    """Return the cheapest model for a given quality tier."""
    if quality == "free":
        return FREE_MODELS[0]
    elif quality == "low":
        return CHEAP_MODELS[0]
    elif quality == "medium":
        return MODERATE_MODELS[0]
    else:
        return DEFAULT_MODEL


def load_api_key():
    return API_KEY_PATH.read_text().strip()


def estimate_cost(n_prompts: int, avg_prompt_tokens: int = 150,
                  avg_completion_tokens: int = 60, model: str = DEFAULT_MODEL) -> dict:
    """Estimate cost for a batch of prompts. Returns dict with token counts and cost."""
    # Approximate pricing per 1M tokens (input/output)
    pricing = {
        "google/gemini-2.0-flash-001": (0.10, 0.40),
        "google/gemini-2.0-flash-001:floor": (0.075, 0.30),
        "meta-llama/llama-3.1-8b-instruct:floor": (0.05, 0.05),
        "mistralai/mistral-7b-instruct:floor": (0.05, 0.05),
        "anthropic/claude-3.5-haiku:floor": (0.80, 4.00),
        "openai/gpt-4o-mini:floor": (0.15, 0.60),
    }
    # Free models
    if ":free" in model:
        return {
            "n_prompts": n_prompts,
            "total_prompt_tokens": n_prompts * avg_prompt_tokens,
            "total_completion_tokens": n_prompts * avg_completion_tokens,
            "estimated_cost_usd": 0.0,
            "note": "Free tier; rate limited to ~20 req/min, 200 req/day",
        }

    base_model = model.replace(":floor", "")
    prices = pricing.get(model, pricing.get(base_model, (0.10, 0.40)))
    input_cost = (n_prompts * avg_prompt_tokens / 1_000_000) * prices[0]
    output_cost = (n_prompts * avg_completion_tokens / 1_000_000) * prices[1]
    return {
        "n_prompts": n_prompts,
        "total_prompt_tokens": n_prompts * avg_prompt_tokens,
        "total_completion_tokens": n_prompts * avg_completion_tokens,
        "estimated_cost_usd": round(input_cost + output_cost, 4),
        "price_per_1m_input": prices[0],
        "price_per_1m_output": prices[1],
        "note": "Estimate; actual cost may vary. :floor routes to cheapest provider.",
    }


@dataclass
class UsageTracker:
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_requests: int = 0
    total_cost_usd: float = 0.0
    failed_requests: int = 0
    start_time: float = field(default_factory=time.time)

    def add(self, prompt_tokens: int, completion_tokens: int, cost: float = 0.0):
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_requests += 1
        self.total_cost_usd += cost

    def summary(self) -> str:
        elapsed = time.time() - self.start_time
        rps = self.total_requests / elapsed if elapsed > 0 else 0
        return (
            f"Requests: {self.total_requests} ({self.failed_requests} failed) | "
            f"Tokens: {self.total_prompt_tokens:,} prompt + {self.total_completion_tokens:,} completion | "
            f"Cost: ${self.total_cost_usd:.4f} | "
            f"Time: {elapsed:.1f}s ({rps:.1f} req/s)"
        )


async def call_openrouter(
    session: aiohttp.ClientSession,
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 300,
    api_key: str = None,
    tracker: UsageTracker = None,
    semaphore: asyncio.Semaphore = None,
    retries: int = 3,
    provider_opts: dict = None,
) -> str | None:
    """Make a single OpenRouter API call with retry logic and cost optimization."""
    if semaphore:
        await semaphore.acquire()
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ApartsinProjects/ImplicitEntities",
            "X-Title": "IRC-Recovery-Experiments",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Add provider options for cost optimization
        if provider_opts:
            if "service_tier" in provider_opts:
                # OpenAI flex tier (~50% discount)
                payload["service_tier"] = provider_opts["service_tier"]
            if "sort" in provider_opts:
                payload.setdefault("provider", {})["sort"] = provider_opts["sort"]
            if "max_price" in provider_opts:
                payload.setdefault("provider", {})["max_price"] = provider_opts["max_price"]

        for attempt in range(retries):
            try:
                async with session.post(BASE_URL, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    if resp.status == 429:
                        wait = float(resp.headers.get("Retry-After", 2 * (attempt + 1)))
                        await asyncio.sleep(wait)
                        continue
                    if resp.status != 200:
                        body = await resp.text()
                        if attempt < retries - 1:
                            await asyncio.sleep(1)
                            continue
                        if tracker:
                            tracker.failed_requests += 1
                        return None

                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"]

                    if tracker:
                        usage = data.get("usage", {})
                        tracker.add(
                            usage.get("prompt_tokens", 0),
                            usage.get("completion_tokens", 0),
                            0.0,
                        )
                    return content.strip()

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < retries - 1:
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
                if tracker:
                    tracker.failed_requests += 1
                return None
    finally:
        if semaphore:
            semaphore.release()


async def batch_call(
    prompts: list[list[dict]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 300,
    concurrency: int = 10,
    progress_every: int = 50,
    provider_opts: dict = None,
) -> list[str | None]:
    """
    Process a batch of prompts concurrently with cost optimization.
    Each prompt is a list of message dicts [{role, content}, ...].
    Returns list of response strings (None for failures).

    Cost tips:
      - Use model="...:floor" for cheapest provider routing
      - Use model="...:free" for free tier (rate limited)
      - Use provider_opts={"service_tier": "flex"} for OpenAI 50% discount
      - For free models, set concurrency=5 to respect rate limits
    """
    # Auto-adjust concurrency for free models
    if ":free" in model and concurrency > 5:
        print(f"  [COST] Free model detected; reducing concurrency to 5 (rate limit)")
        concurrency = 5

    api_key = load_api_key()
    tracker = UsageTracker()
    semaphore = asyncio.Semaphore(concurrency)
    results = [None] * len(prompts)
    completed = 0

    async with aiohttp.ClientSession() as session:
        async def process_one(idx, msgs):
            nonlocal completed
            result = await call_openrouter(
                session, msgs, model=model, temperature=temperature,
                max_tokens=max_tokens, api_key=api_key, tracker=tracker,
                semaphore=semaphore, provider_opts=provider_opts,
            )
            results[idx] = result
            completed += 1
            if completed % progress_every == 0:
                print(f"  [{completed}/{len(prompts)}] {tracker.summary()}")

        tasks = [process_one(i, msgs) for i, msgs in enumerate(prompts)]
        await asyncio.gather(*tasks)

    print(f"  [DONE] {tracker.summary()}")
    return results
