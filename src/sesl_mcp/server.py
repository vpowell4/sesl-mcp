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
# CONFIGURATION
# ============================================================

SERVER_VERSION = "0.1.0"
SERVER_HOST = os.getenv("SESL_MCP_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SESL_MCP_PORT", "3000"))
DEBUG_MODE = os.getenv("SESL_MCP_DEBUG", "true").lower() == "true"

# HTTP request limits (in bytes)
MAX_REQUEST_SIZE = int(os.getenv("SESL_MCP_MAX_REQUEST_SIZE", str(100 * 1024 * 1024)))  # 100MB default
REQUEST_TIMEOUT = int(os.getenv("SESL_MCP_TIMEOUT", "300"))  # 5 minutes default


# ============================================================
# LOGGING SETUP
# ============================================================

LOGGER_NAME = "sesl-mcp-server"
logger = logging.getLogger(LOGGER_NAME)

if not logger.handlers:
    log_level = logging.DEBUG if DEBUG_MODE else logging.INFO
    logging.basicConfig(
        level=log_level,
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


def make_detailed_error(error_type: str, message: str, details: Dict[str, Any] = None) -> TextContent:
    """Create detailed error response with debugging information."""
    error_data = {
        "error": message,
        "error_type": error_type,
        "timestamp": str(Path(__file__).parent),  # Using Path as placeholder for datetime
    }
    if details:
        error_data["details"] = details
    return make_text_content(error_data)


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

mcp = FastMCP(
    name="sesl-mcp-server",
    version=SERVER_VERSION,
)
mcp.debug = DEBUG_MODE


# ============================================================
# TOOL 1: generate_sesl
# ============================================================

@mcp.tool()
async def generate_sesl(prompt: str) -> str:
    """
    Generate a SESL prompt template for converting natural language to SESL YAML.
    
    Args:
        prompt: Natural language description of business rules or policies
    
    Returns:
        Complete prompt template with SESL specification and examples
    """
    prompt_length = len(prompt) if prompt else 0
    logger.info("generate_sesl called, prompt length: %d chars", prompt_length)

    if not prompt or not prompt.strip():
        logger.warning("Empty prompt provided to generate_sesl")
        return "Error: Prompt cannot be empty. Please provide a natural language description of the rules you want to generate."
    
    # Warn if prompt is very large (may indicate an issue)
    if prompt_length > 50000:  # ~50KB
        logger.warning("Very large prompt received (%d chars). This may cause performance issues.", prompt_length)

    try:
        result = PROMPT_TEMPLATE.format(prompt=prompt.strip())
        result_length = len(result)
        logger.info("generate_sesl successful, returning %d chars (input: %d chars)", result_length, prompt_length)
        return result
    except KeyError as e:
        error_msg = f"Template formatting error: missing placeholder {e}"
        logger.exception(error_msg)
        return f"Error: {error_msg}\n\nFalling back to basic template.\n\nGenerate SESL code for: {prompt[:500]}..."
    except Exception as e:
        error_msg = f"Unexpected error in generate_sesl: {format_error(e)}"
        logger.exception(error_msg)
        return f"Error: {error_msg}\n\nPrompt length: {prompt_length} chars\nPlease check server logs for details."


# ============================================================
# TOOL 2: lint_sesl  (patched!)
# ============================================================

@mcp.tool()
async def lint_sesl(contents: List[Any]) -> List[TextContent]:
    """
    Validate SESL YAML and return structured error/warning messages.
    
    Args:
        contents: SESL YAML content as list of TextContent, dicts, or strings
    
    Returns:
        List containing TextContent with validation issues (errors/warnings)
    """
    raw = _normalize_contents(contents)
    content_length = len(raw)
    logger.info("lint_sesl called, content length: %d bytes", content_length)

    try:
        if not raw.strip():
            issues = [{
                "level": "ERROR",
                "message": "Empty SESL content provided",
                "rule": "parser",
                "details": "No content to lint. Please provide SESL YAML."
            }]
            return [make_issues_content(issues)]

        # Log first 200 chars for debugging
        logger.debug("lint_sesl content preview: %s", raw[:200])

        issues_obj = lint_model_from_yaml(raw)

        issues = [{
            "level": i.level,
            "message": i.message,
            "rule": i.rule,
        } for i in issues_obj]

        logger.info("lint_sesl completed: %d issues found", len(issues))
        return [make_issues_content(issues)]

    except yaml.YAMLError as e:
        logger.exception("YAML parsing error during lint")
        error_details = {
            "error_type": "YAMLError",
            "message": str(e),
            "content_length": content_length,
            "content_preview": raw[:500] if len(raw) > 500 else raw,
        }
        issues = [{
            "level": "ERROR",
            "message": f"Invalid YAML syntax: {str(e)}",
            "rule": "yaml_parser",
            "details": error_details
        }]
        return [make_issues_content(issues)]
    except Exception as e:
        logger.exception("Exception during lint_sesl")
        error_details = {
            "error_type": type(e).__name__,
            "message": str(e),
            "content_length": content_length,
            "stack_trace": format_error(e),
        }
        issues = [{
            "level": "ERROR",
            "message": f"Linting failed: {format_error(e)}",
            "rule": "internal",
            "details": error_details
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
    """
    Execute SESL rules with provided facts and return computed results.
    
    Args:
        contents: SESL YAML content as list of TextContent, dicts, or strings
        facts: Input facts/data for rule evaluation (dict or nested structure)
    
    Returns:
        List containing TextContent with execution results or errors
    """
    raw = _normalize_contents(contents)
    base_facts = _normalize_facts_input(facts)

    content_length = len(raw)
    logger.info("run_sesl called, content: %d bytes, facts keys: %s",
                content_length, list(base_facts.keys()) if base_facts else [])

    if not raw.strip():
        return [make_detailed_error(
            "EmptyContent",
            "Empty SESL content provided",
            {"suggestion": "Please provide valid SESL YAML content"}
        )]

    # LINT FIRST
    try:
        logger.debug("run_sesl: Starting lint phase")
        lint_issues_obj = lint_model_from_yaml(raw)
    except yaml.YAMLError as e:
        logger.exception("YAML parsing error in run_sesl")
        return [make_detailed_error(
            "YAMLError",
            f"Invalid YAML syntax: {str(e)}",
            {
                "content_length": content_length,
                "content_preview": raw[:500] if len(raw) > 500 else raw,
                "suggestion": "Check YAML syntax: indentation, colons, dashes, quotes"
            }
        )]
    except Exception as e:
        logger.exception("Exception during lint inside run_sesl")
        return [make_detailed_error(
            "LintError",
            f"Lint failure: {format_error(e)}",
            {
                "content_length": content_length,
                "error_type": type(e).__name__,
            }
        )]

    lint_errors = [{
        "level": i.level,
        "message": i.message,
        "rule": i.rule,
    } for i in lint_issues_obj if str(i.level).upper() == "ERROR"]

    if lint_errors:
        logger.warning("run_sesl found %d lint errors", len(lint_errors))
        return [make_issues_content(lint_errors)]

    # LOAD RULES
    try:
        logger.debug("run_sesl: Loading rules from YAML")
        rules, scenarios = load_model_from_yaml(raw)
        logger.info("run_sesl: Loaded %d rules, %d scenarios", len(rules), len(scenarios))
    except Exception as e:
        logger.exception("Failed to load SESL model")
        return [make_detailed_error(
            "ModelLoadError",
            f"Failed to load SESL model: {format_error(e)}",
            {
                "error_type": type(e).__name__,
                "content_length": content_length,
                "suggestion": "Check rule structure: each rule needs rule, priority, if, then, because"
            }
        )]

    # SELECT SCENARIO
    try:
        if scenarios:
            scenario_name, scenario_facts = scenarios[0]
            if not isinstance(scenario_facts, dict):
                scenario_facts = {}
            logger.debug("run_sesl: Using scenario '%s'", scenario_name)
        else:
            scenario_name, scenario_facts = ("runtime", {})
            logger.debug("run_sesl: No scenarios defined, using runtime facts")

        runtime_facts = {
            **scenario_facts,
            **base_facts,
        }

        runtime_facts.setdefault("scenario", scenario_name)
        runtime_facts.setdefault("result", {})
        
        logger.debug("run_sesl: Runtime facts keys: %s", list(runtime_facts.keys()))
    except Exception as e:
        logger.exception("Exception building runtime facts")
        return [make_detailed_error(
            "FactsBuildError",
            f"Failed to build runtime facts: {format_error(e)}",
            {
                "error_type": type(e).__name__,
                "provided_facts": list(base_facts.keys()) if base_facts else [],
                "scenario_count": len(scenarios) if scenarios else 0,
            }
        )]

    # EXECUTE RULES
    try:
        logger.debug("run_sesl: Starting rule execution")
        monitor = Monitor()
        forward_chain(rules, runtime_facts, monitor=monitor)
        result = runtime_facts.get("result", {})
        
        # Add execution metadata
        execution_summary = {
            "result": result,
            "metadata": {
                "rules_executed": len(rules),
                "scenario": scenario_name,
                "result_fields": list(result.keys()) if result else [],
            }
        }
        
        logger.info("SESL execution successful: %d rules, %d result fields",
                   len(rules), len(result))
        return [make_text_content(execution_summary)]

    except Exception as e:
        logger.exception("Exception during SESL forward chaining")
        return [make_detailed_error(
            "ExecutionError",
            f"Rule execution failed: {format_error(e)}",
            {
                "error_type": type(e).__name__,
                "rules_count": len(rules),
                "scenario": scenario_name,
                "facts_provided": list(base_facts.keys()) if base_facts else [],
                "suggestion": "Check rule conditions and LET variable references"
            }
        )]


# ============================================================
# HEALTH CHECK / INFO
# ============================================================

@mcp.tool()
async def server_info() -> Dict[str, Any]:
    """
    Get server version and status information.
    
    Returns:
        Dictionary with server version, status, and available tools
    """
    return {
        "name": "sesl-mcp-server",
        "version": SERVER_VERSION,
        "status": "running",
        "tools": ["generate_sesl", "lint_sesl", "run_sesl", "server_info"],
        "sesl_engine": "https://github.com/vpowell4/sesl",
    }


# ============================================================
# SERVER ENTRYPOINT
# ============================================================

def main() -> None:
    logger.info("🌟 SESL MCP Server v%s Starting...", SERVER_VERSION)
    logger.info("   Host: %s", SERVER_HOST)
    logger.info("   Port: %s", SERVER_PORT)
    logger.info("   Debug: %s", DEBUG_MODE)
    logger.info("   Endpoint: http://%s:%s/mcp\n", 
                SERVER_HOST if SERVER_HOST != "0.0.0.0" else "localhost", 
                SERVER_PORT)

    # Configure Uvicorn for large payloads
    os.environ["UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN"] = "0"
    os.environ["UVICORN_TIMEOUT_KEEP_ALIVE"] = str(REQUEST_TIMEOUT)

    try:
        mcp.run(
            transport="http",
            host=SERVER_HOST,
            port=SERVER_PORT,
            # Uvicorn server config
            timeout_keep_alive=REQUEST_TIMEOUT,
            limit_max_requests=None,  # No request limit
            limit_concurrency=None,   # No concurrency limit
        )
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.exception("Server failed to start: %s", format_error(e))
        raise


if __name__ == "__main__":
    main()
