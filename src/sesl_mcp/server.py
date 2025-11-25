import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastmcp import FastMCP
from mcp.types import TextContent

# SESL Engine + Linter
from sesl.engine.rule_engine import load_model_from_yaml, forward_chain, Monitor
from sesl.tools.linter_core import lint_model_from_yaml


# ============================================================
# LOGGING SETUP
# ============================================================

LOGGER_NAME = "sesl-mcp-server"
logger = logging.getLogger(LOGGER_NAME)

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


# ============================================================
# PROMPT LOADING (generate_sesl)
# ============================================================

PROMPT_FILE_NAME = "sesl_prompt.txt"
PROMPT_PATH = Path(__file__).with_name(PROMPT_FILE_NAME)


def _load_prompt_template() -> str:
    try:
        text = PROMPT_PATH.read_text(encoding="utf-8")
        if "{prompt}" not in text:
            logger.warning(
                "SESL prompt file does not contain '{prompt}' placeholder; "
                "generate_sesl will not inject user prompt correctly."
            )
        return text
    except FileNotFoundError:
        logger.error(
            "SESL prompt file '%s' not found. Falling back to a minimal template.",
            PROMPT_FILE_NAME,
        )
        return "Generate SESL code for the prompt: {prompt}"


PROMPT_TEMPLATE = _load_prompt_template()


# ============================================================
# HELPERS
# ============================================================

def format_error(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


def _json_fallback(obj: Any):
    if isinstance(obj, set):
        return sorted(list(obj))
    return str(obj)


def make_text_content(payload: Any) -> TextContent:
    if isinstance(payload, str):
        text = payload
    else:
        text = json.dumps(payload, indent=2, default=_json_fallback)

    return TextContent(type="text", text=text)


def make_error_content(message: str) -> TextContent:
    return make_text_content({"error": message})


def make_issues_content(issues: List[Dict[str, Any]]) -> TextContent:
    return make_text_content({"issues": issues})


# ============================================================
# NORMALIZE CONTENTS
# ============================================================

def _normalize_contents(contents: List[Any]) -> str:
    """
    Accept both:
      - TextContent objects
      - dicts shaped like {"type": "text", "text": "..."}
      - raw strings
    Returns concatenated YAML text.
    """
    parts = []
    for c in contents:
        if isinstance(c, TextContent):
            parts.append(c.text)
        elif isinstance(c, dict) and c.get("type") == "text":
            parts.append(c.get("text", ""))
        else:
            parts.append(str(c))
    return "".join(parts)


# ============================================================
# FASTMCP SERVER SETUP
# ============================================================

mcp = FastMCP("sesl-mcp-server")
mcp.debug = True


# ============================================================
# TOOL 1: generate_sesl
# ============================================================

@mcp.tool()
async def generate_sesl(prompt: str) -> str:
    logger.info("generate_sesl called with prompt: %r", prompt)

    try:
        return PROMPT_TEMPLATE.format(prompt=prompt)
    except Exception:
        logger.exception("Failed to format SESL prompt template")
        return f"Generate SESL code for the prompt: {prompt}"


# ============================================================
# TOOL 2: lint_sesl  (patched!)
# ============================================================

@mcp.tool()
async def lint_sesl(contents: List[Any]) -> List[TextContent]:
    raw = _normalize_contents(contents)
    logger.info("lint_sesl called")

    try:
        if not raw.strip():
            issues = [{
                "level": "ERROR",
                "message": "Empty SESL",
                "rule": "parser"
            }]
            return [make_issues_content(issues)]

        issues_obj = lint_model_from_yaml(raw)

        issues = [{
            "level": i.level,
            "message": i.message,
            "rule": i.rule,
        } for i in issues_obj]

        return [make_issues_content(issues)]

    except Exception as e:
        logger.exception("Exception during lint_sesl")
        issues = [{
            "level": "ERROR",
            "message": f"Exception during lint: {format_error(e)}",
            "rule": "internal"
        }]
        return [make_issues_content(issues)]


# ============================================================
# TOOL 3: run_sesl (patched!)
# ============================================================

def _normalize_facts_input(facts: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not facts:
        return {}

    if isinstance(facts, dict) and "facts" in facts:
        candidate = facts["facts"]
        if isinstance(candidate, list) and candidate and isinstance(candidate[0], dict):
            return candidate[0]

    if isinstance(facts, dict) and "scenarios" in facts:
        candidate = facts["scenarios"]
        if isinstance(candidate, list) and candidate and isinstance(candidate[0], dict):
            return candidate[0]

    if isinstance(facts, list) and facts and isinstance(facts[0], dict):
        return facts[0]

    if isinstance(facts, dict) and facts and all(isinstance(v, dict) for v in facts.values()):
        first_key = next(iter(facts))
        return facts[first_key]

    if isinstance(facts, dict):
        return facts

    return {}


@mcp.tool()
async def run_sesl(
    contents: List[Any],
    facts: Dict[str, Any],
) -> List[TextContent]:

    raw = _normalize_contents(contents)
    base_facts = _normalize_facts_input(facts)

    logger.info("run_sesl called")

    if not raw.strip():
        return [make_error_content("Empty SESL")]

    # LINT FIRST
    try:
        lint_issues_obj = lint_model_from_yaml(raw)
    except Exception as e:
        logger.exception("Exception during lint inside run_sesl")
        return [make_error_content(f"Lint failure: {format_error(e)}")]

    lint_errors = [{
        "level": i.level,
        "message": i.message,
        "rule": i.rule,
    } for i in lint_issues_obj if str(i.level).upper() == "ERROR"]

    if lint_errors:
        logger.warning("run_sesl found lint errors")
        return [make_issues_content(lint_errors)]

    # LOAD RULES
    try:
        rules, scenarios = load_model_from_yaml(raw)
    except Exception as e:
        logger.warning("Failed to load SESL model: %s", format_error(e))
        return [make_error_content(format_error(e))]

    # SELECT SCENARIO
    try:
        if scenarios:
            scenario_name, scenario_facts = scenarios[0]
            if not isinstance(scenario_facts, dict):
                scenario_facts = {}
        else:
            scenario_name, scenario_facts = ("runtime", {})

        runtime_facts = {
            **scenario_facts,
            **base_facts,
        }

        runtime_facts.setdefault("scenario", scenario_name)
        runtime_facts.setdefault("result", {})
    except Exception as e:
        logger.exception("Exception building runtime facts")
        return [make_error_content(format_error(e))]

    # EXECUTE RULES
    try:
        monitor = Monitor()
        forward_chain(rules, runtime_facts, monitor=monitor)
        result = runtime_facts.get("result", {})
        return [make_text_content(result)]

    except Exception as e:
        logger.exception("Exception during SESL forward chaining")
        return [make_error_content(format_error(e))]


# ============================================================
# SERVER ENTRYPOINT
# ============================================================

def main() -> None:
    logger.info("🌟 SESL MCP Server Running...\n")

    os.environ["UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN"] = "0"

    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=3000,
    )


if __name__ == "__main__":
    main()
