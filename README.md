# context-bomb

Generate documents with an **exact** number of tokens to test LLM context window limits.

Uses OpenAI's [tiktoken](https://github.com/openai/tiktoken) for accurate token counting.

## Install

```bash
pip install tiktoken
```

## Usage

```bash
# Print 1,000 tokens to stdout
python context_bomb.py 1000

# Generate a 128K token file
python context_bomb.py 128K -o bomb_128k.txt

# Generate 275K tokens (for testing large context windows)
python context_bomb.py 275K -o bomb_275k.txt

# 1 million tokens
python context_bomb.py 1M -o bomb_1m.txt

# Use a specific model's tokenizer
python context_bomb.py 200K --model gpt-4o -o bomb_200k.txt

# Verify the token count after generation
python context_bomb.py 128K -o bomb.txt --verify
```

## Supported suffixes

| Input | Tokens |
|-------|--------|
| `1000` | 1,000 |
| `128K` | 128,000 |
| `275K` | 275,000 |
| `1M` | 1,000,000 |
| `1.5M` | 1,500,000 |

## Options

| Flag | Description |
|------|-------------|
| `-o, --output FILE` | Write to file instead of stdout |
| `--encoding NAME` | Tiktoken encoding (default: `cl100k_base`) |
| `--model NAME` | Infer encoding from model (e.g. `gpt-4o`) |
| `--seed N` | Random seed for reproducible output |
| `--verify` | Re-count tokens and print verification |
| `-q, --quiet` | Suppress progress messages |
