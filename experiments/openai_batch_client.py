"""
OpenAI Batch API client for 50% cost reduction.

The OpenAI Batch API processes requests asynchronously with:
  - 50% discount on all token costs
  - 24-hour completion window
  - Up to 50,000 requests per batch
  - Up to 200 MB input file
  - Separate, higher rate limits

Supported models: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo

Usage:
    from openai_batch_client import OpenAIBatchClient

    client = OpenAIBatchClient()

    # Submit a batch
    batch_id = client.submit_batch(
        prompts=[
            [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
            ...
        ],
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=300,
        job_name="e2t_veterans",
    )

    # Check status
    status = client.check_status(batch_id)

    # Retrieve results (blocks until complete, polls every 30s)
    results = client.wait_and_retrieve(batch_id)
    # results = ["response text 1", "response text 2", None, ...]
"""
import json
import time
import os
from pathlib import Path
from datetime import datetime

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

API_KEY_PATH = Path(__file__).parent.parent / "OPenAI.key.txt"
BATCH_DIR = Path(__file__).parent / "batches"
BATCH_DIR.mkdir(exist_ok=True)

# ── Pricing (per 1M tokens, BEFORE 50% batch discount) ──────────────────
# Batch API applies 50% discount automatically
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},       # batch: $0.075/$0.30
    "gpt-4o": {"input": 2.50, "output": 10.00},            # batch: $1.25/$5.00
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},      # batch: $5.00/$15.00
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},      # batch: $0.25/$0.75
}

# Default: cheapest with good quality
DEFAULT_MODEL = "gpt-4o-mini"


def load_openai_key() -> str:
    return API_KEY_PATH.read_text().strip()


def estimate_batch_cost(n_prompts: int, avg_prompt_tokens: int = 150,
                        avg_completion_tokens: int = 60,
                        model: str = DEFAULT_MODEL) -> dict:
    """Estimate cost with 50% batch discount applied."""
    prices = PRICING.get(model, PRICING[DEFAULT_MODEL])
    input_cost = (n_prompts * avg_prompt_tokens / 1_000_000) * prices["input"] * 0.5
    output_cost = (n_prompts * avg_completion_tokens / 1_000_000) * prices["output"] * 0.5
    return {
        "model": model,
        "n_prompts": n_prompts,
        "total_prompt_tokens": n_prompts * avg_prompt_tokens,
        "total_completion_tokens": n_prompts * avg_completion_tokens,
        "cost_without_batch": round(
            (n_prompts * avg_prompt_tokens / 1_000_000) * prices["input"] +
            (n_prompts * avg_completion_tokens / 1_000_000) * prices["output"], 4
        ),
        "cost_with_batch_50pct_off": round(input_cost + output_cost, 4),
        "savings": "50%",
        "completion_window": "24h",
    }


class OpenAIBatchClient:
    """Client for OpenAI Batch API with 50% cost discount."""

    def __init__(self, api_key: str = None):
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            )
        self.api_key = api_key or load_openai_key()
        self.client = OpenAI(api_key=self.api_key)

    def create_batch_jsonl(
        self,
        prompts: list[list[dict]],
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 300,
        job_name: str = "batch",
    ) -> Path:
        """
        Create a JSONL file for the Batch API.

        Args:
            prompts: list of message lists, each [{role, content}, ...]
            model: OpenAI model ID
            temperature: sampling temperature
            max_tokens: max completion tokens
            job_name: prefix for the JSONL filename

        Returns:
            Path to the created JSONL file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        jsonl_path = BATCH_DIR / f"{job_name}_{timestamp}.jsonl"

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for idx, messages in enumerate(prompts):
                task = {
                    "custom_id": f"{job_name}-{idx:06d}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "messages": messages,
                    },
                }
                f.write(json.dumps(task, ensure_ascii=False) + "\n")

        size_mb = jsonl_path.stat().st_size / (1024 * 1024)
        print(f"  [Batch] Created JSONL: {jsonl_path}")
        print(f"  [Batch] {len(prompts)} requests, {size_mb:.2f} MB")

        if len(prompts) > 50000:
            raise ValueError(f"Batch API limit is 50,000 requests; got {len(prompts)}")
        if size_mb > 200:
            raise ValueError(f"Batch API limit is 200 MB; file is {size_mb:.1f} MB")

        return jsonl_path

    def upload_file(self, jsonl_path: Path) -> str:
        """Upload JSONL file to OpenAI. Returns file ID."""
        print(f"  [Batch] Uploading {jsonl_path.name}...")
        with open(jsonl_path, "rb") as f:
            file_obj = self.client.files.create(file=f, purpose="batch")
        print(f"  [Batch] Uploaded. File ID: {file_obj.id}")
        return file_obj.id

    def create_batch(self, file_id: str) -> str:
        """Create a batch job. Returns batch ID."""
        batch = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        print(f"  [Batch] Created batch. ID: {batch.id}")
        print(f"  [Batch] Status: {batch.status}")
        return batch.id

    def check_status(self, batch_id: str) -> dict:
        """Check batch status. Returns status dict."""
        batch = self.client.batches.retrieve(batch_id)
        info = {
            "id": batch.id,
            "status": batch.status,
            "created_at": batch.created_at,
            "completed_at": batch.completed_at,
            "failed_at": batch.failed_at,
            "request_counts": {
                "total": batch.request_counts.total if batch.request_counts else 0,
                "completed": batch.request_counts.completed if batch.request_counts else 0,
                "failed": batch.request_counts.failed if batch.request_counts else 0,
            },
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id,
        }
        return info

    def wait_and_retrieve(
        self, batch_id: str, poll_interval: int = 30, timeout: int = 86400
    ) -> list[str | None]:
        """
        Poll until batch completes, then download and parse results.
        Returns list of response strings ordered by custom_id.
        """
        start = time.time()
        print(f"  [Batch] Waiting for batch {batch_id}...")

        while True:
            status = self.check_status(batch_id)
            elapsed = time.time() - start
            counts = status["request_counts"]
            print(
                f"  [Batch] [{elapsed:.0f}s] Status: {status['status']} "
                f"({counts['completed']}/{counts['total']} done, "
                f"{counts['failed']} failed)"
            )

            if status["status"] == "completed":
                break
            elif status["status"] in ("failed", "expired", "cancelled"):
                print(f"  [Batch] FAILED with status: {status['status']}")
                # Try to download error file
                if status["error_file_id"]:
                    self._download_file(
                        status["error_file_id"],
                        BATCH_DIR / f"errors_{batch_id}.jsonl",
                    )
                return []

            if elapsed > timeout:
                print(f"  [Batch] TIMEOUT after {timeout}s")
                return []

            time.sleep(poll_interval)

        # Download results
        output_path = BATCH_DIR / f"output_{batch_id}.jsonl"
        self._download_file(status["output_file_id"], output_path)

        # Parse results
        return self._parse_output(output_path)

    def _download_file(self, file_id: str, output_path: Path):
        """Download a file from OpenAI."""
        print(f"  [Batch] Downloading {file_id} to {output_path}...")
        content = self.client.files.content(file_id).content
        with open(output_path, "wb") as f:
            f.write(content)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  [Batch] Downloaded {size_mb:.2f} MB")

    def _parse_output(self, output_path: Path) -> list[str | None]:
        """Parse batch output JSONL into ordered response list."""
        results_map = {}
        max_idx = -1

        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                custom_id = obj["custom_id"]
                # Extract index from custom_id (format: "jobname-000042")
                idx = int(custom_id.rsplit("-", 1)[1])
                max_idx = max(max_idx, idx)

                if obj.get("error"):
                    results_map[idx] = None
                else:
                    try:
                        content = obj["response"]["body"]["choices"][0]["message"]["content"]
                        results_map[idx] = content.strip()
                    except (KeyError, IndexError):
                        results_map[idx] = None

        # Build ordered list
        results = [results_map.get(i) for i in range(max_idx + 1)]
        succeeded = sum(1 for r in results if r is not None)
        print(f"  [Batch] Parsed {len(results)} results ({succeeded} succeeded)")
        return results

    def submit_batch(
        self,
        prompts: list[list[dict]],
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 300,
        job_name: str = "batch",
    ) -> str:
        """
        Full pipeline: create JSONL, upload, submit batch.
        Returns batch_id for status checking and retrieval.
        """
        # Cost estimate
        est = estimate_batch_cost(len(prompts), model=model)
        print(f"  [Batch] Model: {model}")
        print(f"  [Batch] Requests: {len(prompts)}")
        print(f"  [Batch] Estimated cost (with 50% batch discount): ${est['cost_with_batch_50pct_off']}")
        print(f"  [Batch] (Without batch: ${est['cost_without_batch']})")
        print(f"  [Batch] Completion window: 24h")

        # Create JSONL
        jsonl_path = self.create_batch_jsonl(
            prompts, model, temperature, max_tokens, job_name
        )

        # Upload
        file_id = self.upload_file(jsonl_path)

        # Submit
        batch_id = self.create_batch(file_id)

        # Save batch metadata
        meta_path = BATCH_DIR / f"meta_{batch_id}.json"
        with open(meta_path, "w") as f:
            json.dump({
                "batch_id": batch_id,
                "file_id": file_id,
                "jsonl_path": str(jsonl_path),
                "model": model,
                "n_prompts": len(prompts),
                "cost_estimate": est,
                "submitted_at": datetime.now().isoformat(),
            }, f, indent=2)

        return batch_id

    def list_batches(self, limit: int = 20) -> list[dict]:
        """List recent batches."""
        batches = self.client.batches.list(limit=limit)
        results = []
        for b in batches.data:
            results.append({
                "id": b.id,
                "status": b.status,
                "created_at": b.created_at,
                "request_counts": {
                    "total": b.request_counts.total if b.request_counts else 0,
                    "completed": b.request_counts.completed if b.request_counts else 0,
                    "failed": b.request_counts.failed if b.request_counts else 0,
                },
            })
        return results

    def cancel_batch(self, batch_id: str):
        """Cancel a pending/in-progress batch."""
        self.client.batches.cancel(batch_id)
        print(f"  [Batch] Cancelled batch {batch_id}")


# ── Convenience functions for integration with experiment pipeline ─────────

def batch_call_openai(
    prompts: list[list[dict]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 300,
    job_name: str = "experiment",
    wait: bool = True,
    poll_interval: int = 30,
) -> list[str | None] | str:
    """
    Drop-in replacement for openrouter_client.batch_call using OpenAI Batch API.

    If wait=True: blocks until batch completes, returns list of response strings.
    If wait=False: submits batch and returns batch_id for later retrieval.

    50% cheaper than standard OpenAI API calls.
    """
    client = OpenAIBatchClient()
    batch_id = client.submit_batch(prompts, model, temperature, max_tokens, job_name)

    if wait:
        return client.wait_and_retrieve(batch_id, poll_interval)
    else:
        print(f"  [Batch] Submitted. Batch ID: {batch_id}")
        print(f"  [Batch] Use `client.wait_and_retrieve('{batch_id}')` to get results")
        return batch_id


# ── CLI for batch management ──────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="OpenAI Batch API manager")
    sub = parser.add_subparsers(dest="command")

    # List batches
    sub.add_parser("list", help="List recent batches")

    # Check status
    p = sub.add_parser("status", help="Check batch status")
    p.add_argument("batch_id")

    # Retrieve results
    p = sub.add_parser("retrieve", help="Download batch results")
    p.add_argument("batch_id")

    # Cancel
    p = sub.add_parser("cancel", help="Cancel a batch")
    p.add_argument("batch_id")

    # Estimate cost
    p = sub.add_parser("estimate", help="Estimate batch cost")
    p.add_argument("--n-prompts", type=int, required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--avg-prompt-tokens", type=int, default=150)
    p.add_argument("--avg-completion-tokens", type=int, default=60)

    args = parser.parse_args()

    if args.command == "list":
        client = OpenAIBatchClient()
        batches = client.list_batches()
        for b in batches:
            counts = b["request_counts"]
            print(f"  {b['id']}  {b['status']:<12}  {counts['completed']}/{counts['total']} done")

    elif args.command == "status":
        client = OpenAIBatchClient()
        status = client.check_status(args.batch_id)
        print(json.dumps(status, indent=2))

    elif args.command == "retrieve":
        client = OpenAIBatchClient()
        results = client.wait_and_retrieve(args.batch_id, poll_interval=10)
        print(f"  Retrieved {len(results)} results")

    elif args.command == "cancel":
        client = OpenAIBatchClient()
        client.cancel_batch(args.batch_id)

    elif args.command == "estimate":
        est = estimate_batch_cost(
            args.n_prompts, args.avg_prompt_tokens,
            args.avg_completion_tokens, args.model,
        )
        print(json.dumps(est, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
