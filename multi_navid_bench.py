#!/usr/bin/env python3
"""
LLM Multi-User TPS Benchmark Suite
==================================
Same tests / temperatures / runs as llm_tps_benchmark.py, but every request is sent by
N users AT THE SAME MOMENT, for each user level in --levels (default 1,2,4).

For each level it prints the same tables as the single-user suite:
  - Per-user TPS : completion tokens / wall time for each user (same metric as the
                   single-user script -> the 1-user level is directly comparable)
  - Total TPS    : all users' tokens / batch wall time = whole-server throughput
Then a LEVEL COMPARISON table across all levels.

A warm-up pass runs first at the highest user level (not counted).

The server needs at least max(levels) slots:
  llama.cpp : --parallel N   (-c is SPLIT across slots -> use -c = N x per-user context)
  vLLM      : --max-num-seqs N
  SGLang    : --max-running-requests N

Usage:
    python llm_tps_benchmark_multiuser.py --url http://127.0.0.1:8080
    python llm_tps_benchmark_multiuser.py --url http://127.0.0.1:8080 --levels 1,2,4,8
    python llm_tps_benchmark_multiuser.py --url http://127.0.0.1:8080 --runs 1 --output mu.json
    python llm_tps_benchmark_multiuser.py --url http://127.0.0.1:8080 --warmup-passes 0
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional
import statistics

try:
    import requests
except ImportError:
    print("Missing dependency: pip install requests")
    sys.exit(1)

# ─────────────────────────────────────────────
# CONFIG — edit these defaults or use CLI flags
# ─────────────────────────────────────────────
DEFAULT_URL     = "http://IP:PORT"  # change to http://5090:8081, etc.
DEFAULT_MODEL   = ""                    # None = auto-detect from /v1/models
DEFAULT_API_KEY = "no-key"                 # set to "no-key" or your actual key
DEFAULT_RUNS    = 3                        # repetitions per test (for averaging)
DEFAULT_MAX_TOKENS = 512
TEMPERATURES    = [0.0, 0.5]
DEFAULT_LEVELS  = "1,2,4"                  # users at once
DEFAULT_WARMUP_PASSES = 1                  # full passes over all tests at max level, not counted
REQUEST_TIMEOUT = 300  # seconds (higher than single-user: several users share the GPU)

# ─────────────────────────────────────────────
# TEST SUITE DEFINITION (identical to llm_tps_benchmark.py)
# ─────────────────────────────────────────────
TEST_CASES = [
    {
        "id": "short_factual",
        "category": "Short Prompt",
        "description": "One-word / very short answer",
        "prompt": "What is the capital of France?",
        "max_tokens": 32,
    },
    {
        "id": "short_chat",
        "category": "Short Prompt",
        "description": "Brief conversational reply",
        "prompt": "Hi! How are you today?",
        "max_tokens": 64,
    },
    {
        "id": "medium_chat",
        "category": "General Chat",
        "description": "Medium-length open-ended question",
        "prompt": "Explain the difference between machine learning and deep learning in simple terms.",
        "max_tokens": 256,
    },
    {
        "id": "general_advice",
        "category": "General Chat",
        "description": "Practical advice generation",
        "prompt": "Give me 5 practical tips for improving my sleep quality.",
        "max_tokens": 300,
    },
    {
        "id": "math_simple",
        "category": "Math / Reasoning",
        "description": "Basic arithmetic word problem",
        "prompt": "A train travels 120 km in 1.5 hours. What is its average speed in km/h?",
        "max_tokens": 128,
    },
    {
        "id": "math_medium",
        "category": "Math / Reasoning",
        "description": "Multi-step reasoning problem",
        "prompt": (
            "Alice has twice as many apples as Bob. Bob has 5 more apples than Carol. "
            "Carol has 8 apples. How many apples do Alice and Bob have together? "
            "Show your reasoning step by step."
        ),
        "max_tokens": 256,
    },
    {
        "id": "math_hard",
        "category": "Math / Reasoning",
        "description": "Complex logic / combinatorics",
        "prompt": (
            "There are 5 teams in a round-robin tournament where every team plays every other team exactly once. "
            "How many total games are played? Now generalise: write the formula for n teams and explain why it works."
        ),
        "max_tokens": 350,
    },
    {
        "id": "coding_short",
        "category": "Coding",
        "description": "Short code generation",
        "prompt": "Write a Python function that returns the nth Fibonacci number using recursion with memoization.",
        "max_tokens": 256,
    },
    {
        "id": "coding_medium",
        "category": "Coding",
        "description": "Medium code generation",
        "prompt": (
            "Write a Python class for a thread-safe LRU cache with get and put methods. "
            "Include type hints and docstrings."
        ),
        "max_tokens": 512,
    },
    {
        "id": "long_prompt_summary",
        "category": "Long Prompt",
        "description": "Long-context summarisation",
        "prompt": (
            "Please summarise the following passage and identify the three main arguments:\n\n"
            "Artificial intelligence is rapidly transforming every sector of the global economy. "
            "In healthcare, AI systems can now detect cancers from imaging scans with accuracy matching or exceeding "
            "that of experienced radiologists, enabling earlier intervention and better patient outcomes. "
            "In finance, algorithmic trading systems process millions of data points per second to execute trades "
            "faster than any human could react, while fraud detection models protect billions of transactions daily. "
            "The transportation industry is undergoing a revolution with autonomous vehicles, which promise to "
            "dramatically reduce the 1.35 million road fatalities that occur worldwide each year. "
            "Manufacturing plants deploy robotic systems guided by computer vision to assemble complex products "
            "with precision and consistency impossible for human hands at scale. "
            "Education is being personalised through adaptive learning platforms that adjust content difficulty "
            "and teaching style to each individual student. "
            "However, these advances bring serious concerns. Labour markets face disruption as automation "
            "threatens white-collar knowledge work for the first time, not just repetitive manual tasks. "
            "Bias encoded in training data can perpetuate or amplify societal inequalities in hiring, "
            "lending, and criminal justice. The concentration of AI capabilities in a small number of large "
            "technology companies raises questions about power, accountability, and democratic governance. "
            "Privacy erosion accelerates as surveillance systems and data brokers exploit personal information. "
            "Finally, the long-term trajectory of increasingly capable AI systems raises profound philosophical "
            "questions about consciousness, agency, and what it means to be human. "
            "Addressing these challenges requires coordinated action by governments, industry, civil society, "
            "and the research community to ensure that the benefits of AI are broadly shared and the risks "
            "are carefully managed."
        ),
        "max_tokens": 400,
    },
    {
        "id": "long_generation",
        "category": "Long Generation",
        "description": "Extended creative/explanatory output",
        "prompt": (
            "Write a detailed technical blog post introduction (at least 400 words) explaining "
            "how transformer attention mechanisms work, aimed at software engineers who know Python "
            "but are new to deep learning."
        ),
        "max_tokens": 600,
    },
    {
        "id": "instruction_follow",
        "category": "Instruction Following",
        "description": "Structured output / format compliance",
        "prompt": (
            "List the planets of the solar system. For each planet provide: "
            "name, distance from sun in AU, and one unique fact. "
            "Format as a numbered list with each field on its own line, labelled."
        ),
        "max_tokens": 400,
    },
    {
        "id": "translation",
        "category": "Language",
        "description": "Translation task",
        "prompt": (
            "Translate the following paragraph into formal French, then into casual Spanish:\n\n"
            "The quick brown fox jumps over the lazy dog. "
            "This sentence is famous because it contains every letter of the English alphabet."
        ),
        "max_tokens": 256,
    },
    {
        "id": "role_play",
        "category": "General Chat",
        "description": "Persona / roleplay generation",
        "prompt": (
            "You are a pirate captain from the 1700s who just discovered a treasure map. "
            "Describe what you see on the map and your plan to find the treasure."
        ),
        "max_tokens": 300,
    },
]

# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────
@dataclass
class SingleResult:
    test_id: str
    category: str
    description: str
    temperature: float
    run_index: int
    users: int          # users in this batch (level)
    user_index: int     # which user (1..users)
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    elapsed_sec: float
    tps: float          # completion tokens / second for THIS user (primary metric)
    ttfb_sec: float     # time-to-first-byte (streaming not used here, so ≈ latency)
    error: Optional[str] = None
    start_t: float = 0.0
    end_t: float = 0.0

@dataclass
class BatchResult:
    test_id: str
    temperature: float
    run_index: int
    users: int
    ok_users: int
    total_completion_tokens: int
    span_sec: float     # first request start -> last request end
    total_tps: float    # all users' tokens / span  (server throughput)

@dataclass
class AggregatedResult:
    test_id: str
    category: str
    description: str
    temperature: float
    users: int
    runs: int
    avg_tps: float          # per-user
    min_tps: float
    max_tps: float
    stdev_tps: float
    avg_total_tps: float    # whole server
    avg_completion_tokens: float
    avg_elapsed_sec: float
    avg_prompt_tokens: float
    errors: int

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def detect_model(base_url: str, api_key: str = "no-key") -> str:
    try:
        r = requests.get(
            f"{base_url}/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        r.raise_for_status()
        models = r.json().get("data", [])
        if models:
            return models[0]["id"]
    except Exception as e:
        print(f"  [warn] Could not auto-detect model: {e}")
    return "unknown"


def get_slots(base_url: str) -> Optional[int]:
    """llama.cpp only: number of parallel slots (None for vLLM / SGLang)."""
    try:
        r = requests.get(f"{base_url}/props", timeout=10)
        if r.ok:
            return r.json().get("total_slots")
    except Exception:
        pass
    return None


def run_single(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    run_index: int,
    test_id: str,
    category: str,
    description: str,
    users: int,
    user_index: int,
    barrier: threading.Barrier,
    api_key: str = "no-key",
) -> SingleResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    try:
        barrier.wait(timeout=60)  # all users fire together
    except threading.BrokenBarrierError:
        pass
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        t1 = time.perf_counter()
        data = resp.json()
        usage = data.get("usage", {})
        prompt_tokens     = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens      = usage.get("total_tokens", 0)
        elapsed = t1 - t0
        tps = completion_tokens / elapsed if elapsed > 0 else 0.0
        return SingleResult(
            test_id=test_id,
            category=category,
            description=description,
            temperature=temperature,
            run_index=run_index,
            users=users,
            user_index=user_index,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            elapsed_sec=round(elapsed, 3),
            tps=round(tps, 2),
            ttfb_sec=round(elapsed, 3),  # non-streaming: total latency
            start_t=t0,
            end_t=t1,
        )
    except Exception as e:
        t1 = time.perf_counter()
        return SingleResult(
            test_id=test_id,
            category=category,
            description=description,
            temperature=temperature,
            run_index=run_index,
            users=users,
            user_index=user_index,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            elapsed_sec=round(t1 - t0, 3),
            tps=0.0,
            ttfb_sec=round(t1 - t0, 3),
            error=str(e),
            start_t=t0,
            end_t=t1,
        )


def run_batch(
    base_url: str,
    model: str,
    tc: dict,
    max_tokens: int,
    temperature: float,
    run_index: int,
    users: int,
    api_key: str = "no-key",
) -> tuple[list[SingleResult], BatchResult]:
    """Send the same test from `users` users at the same moment."""
    barrier = threading.Barrier(users)
    with ThreadPoolExecutor(max_workers=users) as ex:
        futs = [
            ex.submit(
                run_single,
                base_url=base_url,
                model=model,
                prompt=tc["prompt"],
                max_tokens=max_tokens,
                temperature=temperature,
                run_index=run_index,
                test_id=tc["id"],
                category=tc["category"],
                description=tc["description"],
                users=users,
                user_index=u,
                barrier=barrier,
                api_key=api_key,
            )
            for u in range(1, users + 1)
        ]
        results = [f.result() for f in futs]

    ok = [r for r in results if r.error is None]
    if ok:
        span = max(r.end_t for r in ok) - min(r.start_t for r in ok)
        tok = sum(r.completion_tokens for r in ok)
        total_tps = tok / span if span > 0 else 0.0
    else:
        span, tok, total_tps = 0.0, 0, 0.0
    batch = BatchResult(
        test_id=tc["id"],
        temperature=temperature,
        run_index=run_index,
        users=users,
        ok_users=len(ok),
        total_completion_tokens=tok,
        span_sec=round(span, 3),
        total_tps=round(total_tps, 2),
    )
    return results, batch


def aggregate(results: list[SingleResult], batches: list[BatchResult]) -> AggregatedResult:
    good = [r for r in results if r.error is None and r.tps > 0]
    errors = sum(1 for r in results if r.error is not None)
    r0 = good[0] if good else results[0]
    if not good:
        return AggregatedResult(
            test_id=r0.test_id, category=r0.category, description=r0.description,
            temperature=r0.temperature, users=r0.users, runs=len(batches),
            avg_tps=0, min_tps=0, max_tps=0, stdev_tps=0, avg_total_tps=0,
            avg_completion_tokens=0, avg_elapsed_sec=0, avg_prompt_tokens=0,
            errors=errors,
        )
    tps_list = [r.tps for r in good]
    total_list = [b.total_tps for b in batches if b.total_tps > 0]
    return AggregatedResult(
        test_id=r0.test_id,
        category=r0.category,
        description=r0.description,
        temperature=r0.temperature,
        users=r0.users,
        runs=len(batches),
        avg_tps=round(statistics.mean(tps_list), 2),
        min_tps=round(min(tps_list), 2),
        max_tps=round(max(tps_list), 2),
        stdev_tps=round(statistics.stdev(tps_list) if len(tps_list) > 1 else 0.0, 2),
        avg_total_tps=round(statistics.mean(total_list), 2) if total_list else 0.0,
        avg_completion_tokens=round(statistics.mean([r.completion_tokens for r in good]), 1),
        avg_elapsed_sec=round(statistics.mean([r.elapsed_sec for r in good]), 3),
        avg_prompt_tokens=round(statistics.mean([r.prompt_tokens for r in good]), 1),
        errors=errors,
    )

# ─────────────────────────────────────────────
# PRETTY PRINTING
# ─────────────────────────────────────────────
COL = {
    "reset":  "\033[0m",
    "bold":   "\033[1m",
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "red":    "\033[91m",
    "cyan":   "\033[96m",
    "grey":   "\033[90m",
}

def c(color: str, text: str) -> str:
    return f"{COL[color]}{text}{COL['reset']}"

def tps_color(tps: float, width: int = 6) -> str:
    if tps >= 60:   return c("green",  f"{tps:{width}.1f}")
    if tps >= 25:   return c("yellow", f"{tps:{width}.1f}")
    return              c("red",    f"{tps:{width}.1f}")

def print_banner(text: str):
    print(f"\n{c('bold', '═' * 78)}")
    print(c("bold", f"  {text}"))
    print(c("bold", '═' * 78))

def users_label(users: int) -> str:
    return f"{users} user" + ("" if users == 1 else "s")

def print_summary_table(aggregated: list[AggregatedResult], users: int):
    print_banner(f"SUMMARY TABLE  —  {users_label(users)}  —  per-user TPS & total TPS")
    header = (f"{'Test':<22} {'Category':<20} {'Temp':>5} {'AvgTPS':>8} {'MinTPS':>8} {'MaxTPS':>8} "
              f"{'σTPS':>6} {'TotTPS':>8} {'AvgTok':>7} {'AvgSec':>8}")
    print(c("cyan", header))
    print(c("grey", "─" * 106))
    for a in aggregated:
        print(
            f"{a.test_id:<22} {a.category:<20} {a.temperature:>5.1f} "
            f"{tps_color(a.avg_tps, 8)} {a.min_tps:>8.1f} {a.max_tps:>8.1f} "
            f"{a.stdev_tps:>6.1f} {tps_color(a.avg_total_tps, 8)} {a.avg_completion_tokens:>7.0f} {a.avg_elapsed_sec:>8.2f}s"
            + (c("red", f"  ({a.errors} errors)") if a.errors else "")
        )

def print_category_summary(aggregated: list[AggregatedResult], users: int):
    print_banner(f"CATEGORY AVERAGES  —  {users_label(users)}  (all temperatures combined)")
    cats: dict[str, list[AggregatedResult]] = {}
    for a in aggregated:
        cats.setdefault(a.category, []).append(a)
    print(c("cyan", f"{'Category':<25} {'Per-user':>9} {'Total':>9} {'Tests':>6}"))
    print(c("grey", "─" * 52))
    for cat, items in sorted(cats.items()):
        pu = statistics.mean([a.avg_tps for a in items])
        tot = statistics.mean([a.avg_total_tps for a in items])
        print(f"{cat:<25} {tps_color(pu, 9)} {tps_color(tot, 9)} {len(items):>6}")

def print_temp_comparison(aggregated: list[AggregatedResult], users: int):
    print_banner(f"TEMPERATURE COMPARISON  —  {users_label(users)}  (avg TPS across all tests)")
    temp_groups: dict[float, list[AggregatedResult]] = {}
    for a in aggregated:
        temp_groups.setdefault(a.temperature, []).append(a)
    print(c("cyan", f"{'Temperature':>15} {'AvgTPS':>8} {'MedianTPS':>10} {'MinTPS':>8} {'MaxTPS':>8} {'TotTPS':>8}"))
    print(c("grey", "─" * 60))
    for temp in sorted(temp_groups):
        vals = [a.avg_tps for a in temp_groups[temp]]
        tots = [a.avg_total_tps for a in temp_groups[temp]]
        print(
            f"{temp:>15.1f} {tps_color(statistics.mean(vals), 8)} "
            f"{statistics.median(vals):>10.1f} {min(vals):>8.1f} {max(vals):>8.1f} "
            f"{tps_color(statistics.mean(tots), 8)}"
        )

def print_top_bottom(aggregated: list[AggregatedResult], users: int, n: int = 3):
    print_banner(f"TOP {n} & BOTTOM {n} TESTS by avg per-user TPS  —  {users_label(users)}")
    sorted_agg = sorted(aggregated, key=lambda x: x.avg_tps, reverse=True)
    print(c("green", f"  {'Fastest':}"))
    for a in sorted_agg[:n]:
        print(f"    {a.test_id:<22} temp={a.temperature}  {tps_color(a.avg_tps)} TPS per user  "
              f"| total {tps_color(a.avg_total_tps)} TPS")
    print(c("red", f"\n  {'Slowest':}"))
    for a in sorted_agg[-n:]:
        print(f"    {a.test_id:<22} temp={a.temperature}  {tps_color(a.avg_tps)} TPS per user  "
              f"| total {tps_color(a.avg_total_tps)} TPS")

def level_stats(aggregated: list[AggregatedResult]) -> dict:
    pu = [a.avg_tps for a in aggregated if a.avg_tps > 0]
    tot = [a.avg_total_tps for a in aggregated if a.avg_total_tps > 0]
    return {
        "per_user_avg": round(statistics.mean(pu), 2) if pu else 0.0,
        "per_user_max": round(max(pu), 2) if pu else 0.0,
        "per_user_min": round(min(pu), 2) if pu else 0.0,
        "total_avg": round(statistics.mean(tot), 2) if tot else 0.0,
        "errors": sum(a.errors for a in aggregated),
    }

def print_overall(aggregated: list[AggregatedResult], model: str, url: str, elapsed_total: float, users: int):
    s = level_stats(aggregated)
    print_banner(f"OVERALL RESULTS  —  {users_label(users)}")
    print(f"  Server URL  : {c('cyan', url)}")
    print(f"  Model       : {c('cyan', model)}")
    print(f"  Users       : {c('bold', str(users))} at once")
    print(f"  Tests run   : {c('bold', str(len(aggregated)))}")
    print(f"  Level time  : {c('bold', f'{elapsed_total:.1f}s')}")
    if s["per_user_avg"]:
        print(f"  Per-user avg TPS : {tps_color(s['per_user_avg'])}")
        print(f"  Per-user max TPS : {tps_color(s['per_user_max'])}")
        print(f"  Per-user min TPS : {tps_color(s['per_user_min'])}")
        print(f"  Total avg TPS    : {tps_color(s['total_avg'])}   (all {users} users combined)")
    if s["errors"]:
        print(c("red", f"  Errors      : {s['errors']}"))

def print_level_comparison(summaries: dict[int, dict], level_times: dict[int, float]):
    print_banner("LEVEL COMPARISON  —  per-user speed vs whole-server throughput")
    levels = sorted(summaries)
    base = summaries[levels[0]]["total_avg"] or None
    print(c("cyan", f"{'Users':>6} {'Per-user avg':>13} {'min':>7} {'max':>7} {'Total TPS':>10} "
                    f"{'vs ' + users_label(levels[0]):>11} {'Time':>8} {'Errors':>7}"))
    print(c("grey", "─" * 76))
    for u in levels:
        s = summaries[u]
        scale = f"{s['total_avg'] / base:.2f}x" if base else "-"
        print(f"{u:>6} {tps_color(s['per_user_avg'], 13)} {s['per_user_min']:>7.1f} {s['per_user_max']:>7.1f} "
              f"{tps_color(s['total_avg'], 10)} {scale:>11} {level_times[u]:>7.0f}s {s['errors']:>7}")
    print(c("grey", "  Per-user = speed each person sees.  Total = all users combined (server throughput)."))

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LLM Multi-User TPS Benchmark Suite")
    parser.add_argument("--url",        default=DEFAULT_URL,     help="Base URL of the OpenAI-compatible server")
    parser.add_argument("--model",      default=DEFAULT_MODEL,   help="Model name (auto-detected if omitted)")
    parser.add_argument("--api-key",    default=DEFAULT_API_KEY, help="Bearer token (use 'no-key' if none required)")
    parser.add_argument("--runs",       default=DEFAULT_RUNS,  type=int, help="Repetitions per test per level")
    parser.add_argument("--temps",      default=",".join(str(t) for t in TEMPERATURES),
                        help="Comma-separated temperatures, e.g. 0.0,0.5")
    parser.add_argument("--levels",     default=DEFAULT_LEVELS,
                        help="Comma-separated user counts to test, e.g. 1,2,4,8")
    parser.add_argument("--warmup-passes", default=DEFAULT_WARMUP_PASSES, type=int,
                        help="Warm-up passes over all tests at the highest level, not counted (0 = skip)")
    parser.add_argument("--max-tokens", default=DEFAULT_MAX_TOKENS, type=int,
                        help="Global max_tokens override (per-test values take precedence)")
    parser.add_argument("--skip-long",  action="store_true", help="Skip long prompt / generation tests")
    parser.add_argument("--test-ids",   default=None,
                        help="Run only specific test IDs (comma-separated), e.g. math_simple,coding_short")
    parser.add_argument("--output",     default=None, help="Save full results to JSON file")
    parser.add_argument("--no-color",   action="store_true", help="Disable ANSI color output")
    args = parser.parse_args()

    if args.no_color:
        for k in COL:
            COL[k] = ""

    temperatures = [float(t) for t in args.temps.split(",")]
    levels = sorted({int(x) for x in args.levels.split(",") if x.strip()})
    base_url = args.url.rstrip("/")

    # Model detection
    model = args.model or detect_model(base_url, args.api_key)
    slots = get_slots(base_url)
    print(f"\n{c('bold', 'LLM Multi-User TPS Benchmark Suite')}")
    print(f"  URL    : {base_url}")
    print(f"  Model  : {model}")
    print(f"  Slots  : {slots if slots is not None else 'n/a (not llama.cpp)'}")
    print(f"  Levels : {levels} users at once")
    print(f"  Runs   : {args.runs} per test per level")
    print(f"  Temps  : {temperatures}")
    if slots is not None and max(levels) > slots:
        print(c("red", f"  WARNING: server has {slots} slot(s) but you test up to {max(levels)} users — "
                       f"extra users will QUEUE, not run in parallel. Restart with --parallel {max(levels)}."))

    # Filter tests
    tests = TEST_CASES
    if args.skip_long:
        tests = [t for t in tests if t["category"] not in ("Long Prompt", "Long Generation")]
    if args.test_ids:
        ids = set(args.test_ids.split(","))
        tests = [t for t in tests if t["id"] in ids]

    total_calls = len(tests) * len(temperatures) * args.runs * sum(levels)
    print(f"  Total API calls : {total_calls} (+ warm-up)\n")

    # ── Warm-up (not counted) ──
    if args.warmup_passes > 0:
        wu = max(levels)
        print_banner(f"WARM-UP  —  {args.warmup_passes} pass(es) at {users_label(wu)}  (not counted)")
        for p in range(1, args.warmup_passes + 1):
            for tc in tests:
                max_tok = tc.get("max_tokens", args.max_tokens)
                _, b = run_batch(base_url, model, tc, max_tok, temperatures[0], 0, wu, args.api_key)
                status = (c("red", f"{wu - b.ok_users} ERRORS") if b.ok_users < wu
                          else f"total {tps_color(b.total_tps)} TPS")
                print(f"    pass {p}  {tc['id']:<22} → {status}")
        print(c("green", "  Warm-up done."))

    all_raw: list[SingleResult] = []
    all_batches: list[BatchResult] = []
    agg_by_level: dict[int, list[AggregatedResult]] = {}
    summaries: dict[int, dict] = {}
    level_times: dict[int, float] = {}

    t_start = time.perf_counter()

    for users in levels:
        print_banner(f"LEVEL: {users_label(users)} at once")
        t_level = time.perf_counter()
        level_agg: list[AggregatedResult] = []

        for tc in tests:
            for temp in temperatures:
                label = f"[{tc['id']}]  temp={temp}  users={users}"
                print(f"\n  {c('bold', label)}")
                print(f"  {c('grey', tc['description'])}")
                run_results: list[SingleResult] = []
                run_batches: list[BatchResult] = []
                max_tok = tc.get("max_tokens", args.max_tokens)
                for run_i in range(1, args.runs + 1):
                    results, b = run_batch(base_url, model, tc, max_tok, temp, run_i, users, args.api_key)
                    all_raw.extend(results)
                    all_batches.append(b)
                    run_results.extend(results)
                    run_batches.append(b)
                    ok = [r for r in results if r.error is None]
                    errs = [r for r in results if r.error is not None]
                    if ok:
                        pu = [r.tps for r in ok]
                        status = (
                            f"per-user {tps_color(statistics.mean(pu))} TPS "
                            f"(min {min(pu):.1f} max {max(pu):.1f})  |  total {tps_color(b.total_tps)} TPS  "
                            f"({b.total_completion_tokens} tok / {b.span_sec:.2f}s)"
                        )
                    else:
                        status = ""
                    if errs:
                        status += c("red", f"  {len(errs)} ERROR(S): {errs[0].error}")
                    print(f"    run {run_i}/{args.runs}  →  {status}")

                agg = aggregate(run_results, run_batches)
                level_agg.append(agg)
                if not agg.errors:
                    print(
                        f"  {c('bold', 'avg')} → per-user {tps_color(agg.avg_tps)} TPS  "
                        f"(σ={agg.stdev_tps:.1f}  min={agg.min_tps:.1f}  max={agg.max_tps:.1f})  |  "
                        f"total {tps_color(agg.avg_total_tps)} TPS"
                    )

        level_times[users] = time.perf_counter() - t_level
        agg_by_level[users] = level_agg
        summaries[users] = level_stats(level_agg)

        # ── Per-level summaries (same structure as the single-user suite) ──
        print_summary_table(level_agg, users)
        print_category_summary(level_agg, users)
        print_temp_comparison(level_agg, users)
        print_top_bottom(level_agg, users)
        print_overall(level_agg, model, base_url, level_times[users], users)

    t_total = time.perf_counter() - t_start

    # ── Across all levels ──
    print_level_comparison(summaries, level_times)
    print(f"\n  Total time (excl. warm-up): {c('bold', f'{t_total:.1f}s')}")

    # ── JSON output ──
    if args.output:
        payload = {
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "url": base_url,
                "model": model,
                "slots": slots,
                "levels": levels,
                "runs_per_test": args.runs,
                "temperatures": temperatures,
                "warmup_passes": args.warmup_passes,
                "total_elapsed_sec": round(t_total, 2),
            },
            "level_summary": {str(u): summaries[u] for u in levels},
            "aggregated": {str(u): [asdict(a) for a in agg_by_level[u]] for u in levels},
            "batches": [asdict(b) for b in all_batches],
            "raw": [asdict(r) for r in all_raw],
        }
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n  {c('green', f'Results saved → {args.output}')}")

    print()


if __name__ == "__main__":
    main()
