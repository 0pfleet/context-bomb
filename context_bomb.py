#!/usr/bin/env python3
"""
context-bomb: Generate documents with an exact number of tokens to test
LLM context window limits.

Uses the OpenAI tiktoken tokenizer for accurate token counting.
Supports shorthand like 128K, 200K, 1M, etc.
"""

from __future__ import annotations

import argparse
import sys
import os
import random
import re

try:
    import tiktoken
except ImportError:
    print("Error: tiktoken is required. Install it with:\n  pip install tiktoken", file=sys.stderr)
    sys.exit(1)


# Diverse vocabulary for realistic-looking filler text across domains
WORDS = [
    # Common English
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    # Technical
    "function", "return", "value", "system", "process", "data", "output",
    "input", "module", "class", "method", "parameter", "variable", "array",
    "object", "string", "integer", "boolean", "interface", "protocol",
    "network", "server", "client", "request", "response", "endpoint",
    "database", "query", "table", "index", "schema", "migration",
    "algorithm", "complexity", "optimization", "recursive", "iteration",
    "memory", "allocation", "pointer", "reference", "garbage", "collection",
    "thread", "mutex", "semaphore", "deadlock", "concurrent", "parallel",
    # Scientific
    "hypothesis", "experiment", "observation", "analysis", "conclusion",
    "molecule", "atom", "electron", "quantum", "wavelength", "frequency",
    "temperature", "pressure", "velocity", "acceleration", "gravity",
    "evolution", "species", "genetic", "mutation", "chromosome", "protein",
    # Business / general
    "strategy", "implementation", "framework", "architecture", "deployment",
    "infrastructure", "scalability", "performance", "reliability", "monitoring",
    "customer", "revenue", "growth", "market", "product", "service",
    "management", "leadership", "organization", "communication", "decision",
    # Misc longer words (multi-token)
    "unprecedented", "comprehensive", "differentiation", "accountability",
    "infrastructure", "authentication", "internationalization", "microservices",
    "containerization", "observability", "interoperability", "sustainability",
    "representation", "transformation", "recommendation", "characteristics",
]

PUNCTUATION = [".", ".", ".", ",", ",", ";", ":", "!", "?"]
PARAGRAPH_BREAK_CHANCE = 0.04


def parse_token_count(s: str) -> int:
    """Parse token count from string like '128K', '1M', '275000', '1.5M'."""
    s = s.strip().upper().replace(",", "").replace("_", "")
    m = re.match(r'^([0-9]*\.?[0-9]+)\s*(K|M|B)?$', s)
    if not m:
        raise argparse.ArgumentTypeError(
            f"Invalid token count: '{s}'. Use a number like 1000, 128K, 1M, 1.5M"
        )
    num = float(m.group(1))
    suffix = m.group(2)
    multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    if suffix:
        num *= multipliers[suffix]
    return int(num)


def generate_text_block(rng: random.Random, approx_words: int) -> str:
    """Generate a block of pseudo-natural text."""
    parts = []
    sentence_len = rng.randint(5, 20)
    word_count = 0
    capitalize_next = True

    while word_count < approx_words:
        word = rng.choice(WORDS)
        if capitalize_next:
            word = word.capitalize()
            capitalize_next = False

        parts.append(word)
        word_count += 1
        sentence_len -= 1

        if sentence_len <= 0:
            punct = rng.choice(PUNCTUATION)
            parts[-1] += punct
            capitalize_next = punct in ".!?"
            sentence_len = rng.randint(5, 20)

            if rng.random() < PARAGRAPH_BREAK_CHANCE:
                parts.append("\n\n")

    return " ".join(parts)


def generate_document(target_tokens: int, encoding_name: str = "cl100k_base",
                      model: str | None = None, seed: int = 42) -> str:
    """Generate a document with exactly `target_tokens` tokens."""
    if model:
        enc = tiktoken.encoding_for_model(model)
    else:
        enc = tiktoken.get_encoding(encoding_name)

    rng = random.Random(seed)

    # Estimate: ~1.3 tokens per word on average for English text
    approx_words = int(target_tokens / 1.3) + 500
    text = generate_text_block(rng, approx_words)

    tokens = enc.encode(text)
    current = len(tokens)

    if current >= target_tokens:
        # Trim to exact count
        tokens = tokens[:target_tokens]
        return enc.decode(tokens)

    # Need more tokens — keep appending
    while current < target_tokens:
        needed = target_tokens - current
        extra_words = int(needed / 1.3) + 200
        extra_text = " " + generate_text_block(rng, extra_words)
        text += extra_text
        tokens = enc.encode(text)
        current = len(tokens)

    tokens = tokens[:target_tokens]
    return enc.decode(tokens)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a document with an exact token count (OpenAI tokenizer).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s 1000                  # 1,000 tokens, print to stdout
  %(prog)s 128K -o bomb_128k.txt # 128K tokens, save to file
  %(prog)s 275K -o bomb_275k.txt # 275K tokens
  %(prog)s 1M -o bomb_1m.txt    # 1 million tokens
  %(prog)s 200K --model gpt-4o  # use gpt-4o's tokenizer
  %(prog)s 200K --encoding o200k_base  # use a specific encoding
"""
    )
    parser.add_argument(
        "tokens", type=parse_token_count,
        help="Target token count. Supports K/M/B suffixes (e.g. 128K, 1.5M)"
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output file path (default: stdout)"
    )
    parser.add_argument(
        "--encoding", type=str, default="cl100k_base",
        help="Tiktoken encoding name (default: cl100k_base)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Infer encoding from model name (e.g. gpt-4, gpt-4o, gpt-3.5-turbo)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible output (default: 42)"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Re-count tokens after generation and print verification"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Suppress progress messages (only output the document)"
    )

    args = parser.parse_args()

    if not args.quiet:
        print(f"Generating {args.tokens:,} tokens...", file=sys.stderr)

    doc = generate_document(
        target_tokens=args.tokens,
        encoding_name=args.encoding,
        model=args.model,
        seed=args.seed,
    )

    if args.output:
        with open(args.output, "w") as f:
            f.write(doc)
        if not args.quiet:
            size_mb = os.path.getsize(args.output) / (1024 * 1024)
            print(f"Wrote {args.output} ({size_mb:.1f} MB)", file=sys.stderr)
    else:
        sys.stdout.write(doc)

    if args.verify:
        if args.model:
            enc = tiktoken.encoding_for_model(args.model)
        else:
            enc = tiktoken.get_encoding(args.encoding)
        actual = len(enc.encode(doc))
        print(f"Verified: {actual:,} tokens (target: {args.tokens:,})", file=sys.stderr)


if __name__ == "__main__":
    main()
