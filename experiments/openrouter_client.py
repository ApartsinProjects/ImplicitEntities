"""
OpenRouter async client with rate limiting and cost tracking.
Uses concurrent requests with semaphore to maximize throughput while staying under limits.
"""
import asyncio
import aiohttp
import json
import time
from pathlib import Path
from dataclasses import dataclass, field

API_KEY_PATH = Path(__file__).parent.parent / "openRouter.key.txt"
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

def load_api_key():
    return API_KEY_PATH.read_text().strip()

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
    model: str = "google/gemini-2.0-flash-001",
    temperature: float = 0.7,
    max_tokens: int = 300,
    api_key: str = None,
    tracker: UsageTracker = None,
    semaphore: asyncio.Semaphore = None,
    retries: int = 3,
) -> str | None:
    """Make a single OpenRouter API call with retry logic."""
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
                            0.0,  # cost tracked separately
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
    model: str = "google/gemini-2.0-flash-001",
    temperature: float = 0.7,
    max_tokens: int = 300,
    concurrency: int = 10,
    progress_every: int = 50,
) -> list[str | None]:
    """
    Process a batch of prompts concurrently.
    Each prompt is a list of message dicts [{role, content}, ...].
    Returns list of response strings (None for failures).
    """
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
                semaphore=semaphore,
            )
            results[idx] = result
            completed += 1
            if completed % progress_every == 0:
                print(f"  [{completed}/{len(prompts)}] {tracker.summary()}")

        tasks = [process_one(i, msgs) for i, msgs in enumerate(prompts)]
        await asyncio.gather(*tasks)

    print(f"  [DONE] {tracker.summary()}")
    return results
