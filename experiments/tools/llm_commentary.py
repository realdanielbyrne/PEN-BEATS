"""LLM-enhanced commentary generation for N-BEATS analysis scripts.

Provides a single `generate_commentary(section_type, context, instructions)` function
that returns LLM-generated markdown prose, or None if no API key is configured.

Responses are cached to disk using SHA-256(section_type + sorted JSON of context)
so repeated analysis runs do not incur extra API costs.

Authentication (in priority order):
  1. ANTHROPIC_API_KEY env var → Anthropic API key auth (per-call billing)
  2. ANTHROPIC_OAUTH_TOKEN env var → OAuth token auth (subscription billing)
  3. Claude Code credentials file (~/.claude/.credentials.json) → OAuth token
     auth (subscription billing), auto-detected when NBEATS_LLM_PROVIDER is
     "anthropic" or "claude_code"
  4. OPENAI_API_KEY env var with NBEATS_LLM_PROVIDER=openai → OpenAI

OAuth tokens can be generated via `claude setup-token` in Claude Code.

Environment variables:
    NBEATS_LLM_PROVIDER      anthropic (default), openai, or claude_code
    ANTHROPIC_API_KEY        Anthropic API key (highest priority)
    ANTHROPIC_OAUTH_TOKEN    OAuth token from `claude setup-token` (subscription billing)
    OPENAI_API_KEY           required for OpenAI provider
    NBEATS_LLM_MODEL         override model (defaults per provider below)
    NBEATS_LLM_CACHE_DIR     override cache directory

Default models:
    Anthropic/Claude Code: claude-haiku-4-5-20251001
    OpenAI:                gpt-4o-mini
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_PROVIDER = os.environ.get("NBEATS_LLM_PROVIDER", "anthropic").lower()
_ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
_OAUTH_TOKEN = os.environ.get("ANTHROPIC_OAUTH_TOKEN", "")
_OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")

_DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "claude_code": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
}
_MODEL = os.environ.get("NBEATS_LLM_MODEL", _DEFAULT_MODELS.get(_PROVIDER, "claude-haiku-4-5-20251001"))

_SCRIPT_DIR = Path(__file__).parent
_DEFAULT_CACHE_DIR = _SCRIPT_DIR / "analysis_reports" / ".llm_cache"
_CACHE_DIR = Path(os.environ.get("NBEATS_LLM_CACHE_DIR", str(_DEFAULT_CACHE_DIR)))

# Path to Claude Code credentials file
_CLAUDE_CODE_CREDENTIALS = Path.home() / ".claude" / ".credentials.json"

# ---------------------------------------------------------------------------
# System prompt — domain knowledge baked into every call
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an expert ML researcher writing concise, insightful analysis commentary "
    "for N-BEATS time series forecasting experiments. N-BEATS uses stacks of "
    "basis-expansion blocks; OWA (Overall Weighted Average of sMAPE and MASE) is the "
    "primary metric — lower is better. Baselines: NBEATS-G (0.8198 OWA), NBEATS-I "
    "(0.8132), NBEATS-I+G (0.8057). The M4-Yearly dataset is the primary benchmark. "
    "Successive halving progressively eliminates weak configs while increasing training "
    "budgets. AE variants use an encoder-decoder bottleneck inside each block. When "
    "writing commentary, reference the specific numbers given, explain architectural "
    "WHY, and give actionable guidance. Use markdown formatting. Be concise — 2-4 "
    "paragraphs maximum per section."
)

# ---------------------------------------------------------------------------
# Claude Code OAuth token helpers
# ---------------------------------------------------------------------------

def _load_claude_code_token() -> str | None:
    """Read the Claude Code OAuth access token from ~/.claude/.credentials.json.

    Returns the token string if valid and not expired, or None.
    """
    if not _CLAUDE_CODE_CREDENTIALS.exists():
        return None
    try:
        with open(_CLAUDE_CODE_CREDENTIALS, "r", encoding="utf-8") as f:
            creds = json.load(f)
        oauth = creds.get("claudeAiOauth", {})
        token = oauth.get("accessToken")
        expires_at_ms = oauth.get("expiresAt")  # Unix timestamp in milliseconds
        if not token:
            return None
        if expires_at_ms is not None:
            expires_at_s = expires_at_ms / 1000
            now = time.time()
            if now >= expires_at_s:
                print(
                    "[llm_commentary] Claude Code OAuth token has expired. "
                    "Set ANTHROPIC_API_KEY to use the Anthropic API.",
                    file=sys.stderr,
                )
                return None
            # Warn if token expires within 5 minutes
            if expires_at_s - now < 300:
                print(
                    f"[llm_commentary] Warning: Claude Code OAuth token expires in "
                    f"{int(expires_at_s - now)}s.",
                    file=sys.stderr,
                )
        return token
    except (json.JSONDecodeError, OSError, KeyError):
        return None


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(section_type: str, context: dict) -> Path:
    """Return the cache file path for a given section_type + context."""
    payload = section_type + json.dumps(context, sort_keys=True, default=str)
    hex_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return _CACHE_DIR / f"{hex_hash}.json"


def _load_cache(path: Path) -> str | None:
    """Return cached response string or None."""
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("response")
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _save_cache(path: Path, section_type: str, response: str) -> None:
    """Write response to cache file."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "section_type": section_type,
        "cache_key": path.stem,
        "response": response,
        "model": _MODEL,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Client helpers
# ---------------------------------------------------------------------------

def _call_anthropic(user_prompt: str, api_key: str = "", auth_token: str = "") -> str:
    """Call the Anthropic API using either an API key or OAuth token."""
    import anthropic
    if auth_token:
        client = anthropic.Anthropic(auth_token=auth_token)
    else:
        client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=_MODEL,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return getattr(message.content[0], "text", str(message.content[0]))


def _call_openai(user_prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=_OPENAI_KEY)
    response = client.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1024,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Section-type to user prompt templates
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATES = {
    "hyperparameter_marginal": (
        "Write commentary on the marginal effect of the hyperparameter `{parameter_name}` "
        "on median OWA. Best value: {best_value} (OWA={best_owa:.4f}), worst: {worst_value} "
        "(OWA={worst_owa:.4f}), delta={delta:.4f}. Full distribution: {all_values}. "
        "Explain why the best value wins architecturally and give guidance on how to set this parameter."
    ),
    "hyperparameter_discussion": (
        "Write an in-depth discussion on selecting `{parameter_name}` for the `{architecture_name}` "
        "architecture on M4-Yearly (backcast_length={backcast_length}, forecast_length={forecast_length}). "
        "Stats: best={best_value} (OWA={best_owa:.4f}), worst={worst_value} (OWA={worst_owa:.4f}), "
        "delta={delta:.4f}. Full distribution: {all_values}. "
        "Explain the regularisation trade-off and give a practical recommendation."
    ),
    "variant_comparison": (
        "Write a conclusion for the head-to-head comparison of these variants: {variants}. "
        "Round results (best OWA per variant per round): {round_results}. "
        "Identify which variant is most robust across rounds and explain why architecturally."
    ),
    "stability_analysis": (
        "Write a stability analysis conclusion. Mean spread: {mean_spread:.4f}, "
        "max spread: {max_spread:.4f}, most stable config(s): {most_stable}, "
        "most volatile config(s): {most_volatile}. "
        "What does high/low spread imply about seed sensitivity and production reliability?"
    ),
    "param_efficiency": (
        "Write a parameter efficiency commentary. Baseline params: {baseline_params:,}. "
        "Best config: {best_config}. Config details: {configs}. "
        "Discuss the efficiency frontier and whether the parameter reduction justifies the OWA trade-off."
    ),
    "round_progression": (
        "Write a round-over-round progression commentary. {n_improved} of {n_total} surviving "
        "configs improved from first to last round. Progression data: {progression_data}. "
        "What does this reveal about the benefit of increased training budget in successive halving?"
    ),
    "convergence_analysis": (
        "Write a convergence curve discussion for this experiment. Epoch-by-epoch stats: {epoch_stats}. "
        "Additional context: {context_extra}. "
        "Interpret what the convergence pattern reveals about training stability and optimal stopping."
    ),
}


def _build_user_prompt(section_type: str, context: dict, instructions: str) -> str:
    """Build the user message from template + context + optional extra instructions."""
    template = _PROMPT_TEMPLATES.get(section_type)
    if template:
        try:
            prompt = template.format(**context)
        except (KeyError, ValueError):
            # Fall back to raw JSON if template fields are missing
            prompt = f"Section: {section_type}\nContext: {json.dumps(context, indent=2, default=str)}"
    else:
        prompt = f"Section: {section_type}\nContext: {json.dumps(context, indent=2, default=str)}"

    if instructions:
        prompt += f"\n\nAdditional instructions: {instructions}"
    return prompt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_commentary(
    section_type: str,
    context: dict[str, Any],
    instructions: str = "",
) -> str | None:
    """Generate LLM commentary for the given section type and context.

    Authentication is resolved in this order:
      1. ANTHROPIC_API_KEY env var → API key auth (per-call billing)
      2. ANTHROPIC_OAUTH_TOKEN env var → OAuth token auth (subscription billing)
      3. Claude Code credentials file (~/.claude/.credentials.json) → OAuth token
      4. OPENAI_API_KEY with NBEATS_LLM_PROVIDER=openai → OpenAI

    Returns:
        str:  Markdown-formatted commentary from the LLM.
        None: If no credentials are available (caller falls back to hardcoded text).
    """
    # Resolve credentials
    api_key = ""
    auth_token = ""
    effective_provider = _PROVIDER

    if effective_provider in ("anthropic", "claude_code"):
        if _ANTHROPIC_KEY:
            api_key = _ANTHROPIC_KEY
        elif _OAUTH_TOKEN:
            auth_token = _OAUTH_TOKEN
        else:
            auth_token = _load_claude_code_token() or ""
        if not api_key and not auth_token:
            return None
    elif effective_provider == "openai":
        if not _OPENAI_KEY:
            return None
    else:
        return None

    # Check cache first
    cache_file = _cache_path(section_type, context)
    cached = _load_cache(cache_file)
    if cached is not None:
        return cached

    # Build prompt and call API
    user_prompt = _build_user_prompt(section_type, context, instructions)
    try:
        if effective_provider in ("anthropic", "claude_code"):
            response = _call_anthropic(user_prompt, api_key=api_key, auth_token=auth_token)
        else:
            response = _call_openai(user_prompt)
    except Exception as e:
        print(f"[llm_commentary] API error ({type(e).__name__}): {e}", file=sys.stderr)
        return None

    # Cache and return
    _save_cache(cache_file, section_type, response)
    return response
