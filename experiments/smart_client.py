"""
Smart API client that auto-selects the cheapest backend for each model.

Routing logic:
  - OpenAI models (gpt-4o-mini, gpt-4o, etc.) -> OpenAI Batch API (50% off)
  - Other models -> OpenRouter with :floor suffix (cheapest provider)
  - Free models -> OpenRouter :free suffix ($0, rate limited)

Usage:
    from smart_client import smart_batch_call, estimate_smart_cost

    # Auto-routes to cheapest option
    results = await smart_batch_call(prompts, model="gpt-4o-mini")  # -> OpenAI Batch (50% off)
    results = await smart_batch_call(prompts, model="google/gemini-2.0-flash-001")  # -> OpenRouter :floor
    results = await smart_batch_call(prompts, model="deepseek/deepseek-chat-v3-0324:free")  # -> OpenRouter free
"""
import asyncio
import json
from pathlib import Path

# Import both clients
from openrouter_client import batch_call as openrouter_batch_call, estimate_cost as openrouter_estimate

OPENAI_BATCH_AVAILABLE = False
try:
    from openai_batch_client import (
        batch_call_openai, estimate_batch_cost as openai_estimate,
        OpenAIBatchClient, OPENAI_AVAILABLE,
    )
    if OPENAI_AVAILABLE:
        OPENAI_BATCH_AVAILABLE = True
except ImportError:
    pass

# Models that should use OpenAI Batch API (50% discount)
OPENAI_MODELS = {
    "gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo",
    "openai/gpt-4o-mini", "openai/gpt-4o", "openai/gpt-4-turbo",
}


def _is_openai_model(model: str) -> bool:
    """Check if model should be routed through OpenAI Batch API."""
    base = model.replace(":floor", "").replace(":free", "")
    base_short = base.split("/")[-1] if "/" in base else base
    return base in OPENAI_MODELS or base_short in OPENAI_MODELS


def _strip_openrouter_prefix(model: str) -> str:
    """Convert 'openai/gpt-4o-mini' -> 'gpt-4o-mini' for OpenAI API."""
    if "/" in model:
        return model.split("/", 1)[1]
    return model


def estimate_smart_cost(n_prompts: int, model: str, avg_prompt_tokens: int = 150,
                        avg_completion_tokens: int = 60) -> dict:
    """Estimate cost using the cheapest available backend."""
    if _is_openai_model(model) and OPENAI_BATCH_AVAILABLE:
        est = openai_estimate(n_prompts, avg_prompt_tokens, avg_completion_tokens,
                              _strip_openrouter_prefix(model))
        est["backend"] = "openai_batch"
        est["discount"] = "50% batch discount"
        return est
    else:
        est = openrouter_estimate(n_prompts, avg_prompt_tokens, avg_completion_tokens, model)
        est["backend"] = "openrouter"
        if ":free" in model:
            est["discount"] = "100% free (rate limited)"
        elif ":floor" in model:
            est["discount"] = "~25-50% floor routing"
        else:
            est["discount"] = "none"
        return est


async def smart_batch_call(
    prompts: list[list[dict]],
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 300,
    concurrency: int = 10,
    job_name: str = "experiment",
    wait: bool = True,
) -> list[str | None]:
    """
    Route to the cheapest backend automatically.

    - OpenAI models -> Batch API (50% off, async 24h)
    - Others -> OpenRouter (sync, concurrent)
    """
    est = estimate_smart_cost(len(prompts), model)
    print(f"  [Smart] Model: {model}")
    print(f"  [Smart] Backend: {est['backend']} ({est.get('discount', 'none')})")
    print(f"  [Smart] Estimated cost: ${est.get('cost_with_batch_50pct_off', est.get('estimated_cost_usd', '?'))}")

    if _is_openai_model(model) and OPENAI_BATCH_AVAILABLE:
        # Use OpenAI Batch API (50% off, but async)
        clean_model = _strip_openrouter_prefix(model.replace(":floor", "").replace(":free", ""))
        print(f"  [Smart] Using OpenAI Batch API for {clean_model} (50% discount)")

        # batch_call_openai is synchronous (it polls internally)
        results = batch_call_openai(
            prompts, model=clean_model, temperature=temperature,
            max_tokens=max_tokens, job_name=job_name, wait=wait,
        )
        if isinstance(results, str):
            # wait=False returned batch_id
            print(f"  [Smart] Batch submitted: {results}")
            return results
        return results

    else:
        # Use OpenRouter (sync concurrent)
        # Auto-add :floor if not already specified and not :free
        if ":floor" not in model and ":free" not in model:
            model_with_floor = model + ":floor"
            print(f"  [Smart] Auto-adding :floor routing: {model_with_floor}")
        else:
            model_with_floor = model

        return await openrouter_batch_call(
            prompts, model=model_with_floor, temperature=temperature,
            max_tokens=max_tokens, concurrency=concurrency,
        )


def print_cost_comparison(n_prompts: int, models: list[str]):
    """Print a cost comparison table for all models and backends."""
    print(f"\n  Cost Comparison ({n_prompts} prompts)")
    print(f"  {'Model':<45s} {'Backend':<15s} {'Cost':>10s} {'Discount':>20s}")
    print(f"  {'-' * 90}")

    for model in models:
        est = estimate_smart_cost(n_prompts, model)
        cost = est.get("cost_with_batch_50pct_off", est.get("estimated_cost_usd", 0))
        backend = est.get("backend", "?")
        discount = est.get("discount", "none")
        print(f"  {model:<45s} {backend:<15s} ${cost:>8.4f} {discount:>20s}")


if __name__ == "__main__":
    # Demo: print cost comparison for all models
    models = [
        "gpt-4o-mini",
        "openai/gpt-4o-mini",
        "google/gemini-2.0-flash-001",
        "google/gemini-2.0-flash-001:floor",
        "meta-llama/llama-3.1-8b-instruct:floor",
        "deepseek/deepseek-chat-v3-0324:free",
        "anthropic/claude-3.5-haiku:floor",
    ]
    print_cost_comparison(1000, models)
