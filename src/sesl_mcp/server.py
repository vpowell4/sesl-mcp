import json
import logging
import os
import traceback
from typing import Any, Dict, List, Optional, Tuple

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
# HELPER: format_error
# ============================================================

def format_error(e: BaseException) -> str:
    """
    Return a concise, readable error string for exceptions.
    """
    return f"{type(e).__name__}: {e}"


# ============================================================
# HELPER: TextContent factories
# ============================================================

def make_text_content(payload: Any) -> TextContent:
    """
    Wrap a Python object (dict, list, str, etc.) into a JSON-encoded TextContent.
    Always returns a valid TextContent for MCP (type='text').
    """
    if isinstance(payload, str):
        text = payload
    else:
        text = json.dumps(payload, indent=2)
    return TextContent(type="text", text=text)


def make_error_content(message: str) -> TextContent:
    """
    Standard JSON error wrapper.
    """
    return make_text_content({"error": message})


def make_issues_content(issues: List[Dict[str, Any]]) -> TextContent:
    """
    Wrap SESL lint issues in a standard shape.
    """
    return make_text_content({"issues": issues})


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
    """
    Generate SESL authoring instructions for a given natural-language prompt.
    NOTE: This returns a string (not TextContent) on purpose so the client
    can feed it directly into an LLM as a system or tool prompt.
    """

    logger.info("generate_sesl called with prompt: %r", prompt)

    # 🔴 IMPORTANT: Do not change the instructional text content.
    return f"""Generate SESL code for the prompt: {prompt} using the following rulles.
Produce VALID SESL YAML ONLY. ONLY USE THE commands and structures below. Do not add anything. You are an expert SESL rule-writer.
Your task is to convert natural-language policies or business logic into SESL YAML compatible with the SESL rule engine. Follow these rules exactly:
SESL File Structure
A SESL file is valid YAML and contains:
const: (optional dictionary of global constants)
rules: (list of rules)
facts: (the list of facts which drives the rules)
A rule must have this structure:
rule: <rule name>
priority: <integer>
let: <optional dictionary of precomputed variables>
if: <condition: string, list, or nested logic dict>
then: <result assignments>
because: <short explanation string>
stop: true|false (optional)
Indentation must be 2 spaces.
LET Variables
LET variables are evaluated before conditions. They may reference facts, constants, and other LET variables except themselves.
Allowed expressions: literals, arithmetic, boolean expressions, dotted paths like user.age. Missing dotted paths return None. None behaves as 0 in arithmetic.
Division by zero returns 0. Self-referential LETs are forbidden.
Conditions
Valid condition forms:
Simple: user.age > 18, status == "ACTIVE", country in ["US","CA"], role not in ["admin","owner"].
Nested logic:
if:
  all:
    user.age > 18
  any:
    score > 700
    status == "VIP"
Logic keys: all or and, any or or, not.
Operand Resolution Rules
String literals are quoted. true and false are booleans. Numbers without quotes are numeric.
Dotted names (user.age) are field lookups.
Bare identifiers (no dot, no quotes) resolve in this order: 1. LET variables, 2. constants (_const), 3. fact fields, otherwise error. result.* lookups are soft and missing values return None.
Actions (THEN block)
Actions assign fields under result. If the expression cannot be resolved as a literal or identifier, treat it as a string.
Fields without dots are implicitly placed under result. Lower-priority rules cannot override fields set by higher-priority rules.
Rule Priorities
Higher priority executes first and owns the fields it sets. Lower-priority rules cannot overwrite those fields.
Truth Maintenance System
Do not reference TMS explicitly. A field remains set only while the rules supporting it remain true. When supporting rules become false, the field is retracted.
Write rules so this behavior is correct.
Error-Safe SESL Generation
Avoid unreachable or contradictory conditions. Avoid self-referential LETs. All lists must be valid Python literals.
Quote all strings. Use explicit values in actions such as "APPROVED".
Output Requirements
When encoding a source document, extract rules and convert them into SESL YAML. Use only valid YAML with 2-space indentation.
Produce minimal, concise rules. Include a because field summarizing each rule. Output only SESL YAML with no commentary.
Example Rule
rule: AdultCheck
priority: 10
if: user.age >= 18
then:
  is_adult: true
because: "User is 18 or older"
You must follow all rules above exactly when generating SESL.
Facts
The facts: block is an optional top-level YAML list used only for testing SESL rules.
Each item in the facts: list represents one input scenario. A fact scenario defines the fields available to rule conditions and LET expressions.
Facts simulate real input data such as user.age, country, or status. Facts do not affect rule logic or priority; they exist only to verify rule behavior.
Each fact is a YAML dictionary representing one evaluation context.
Missing dotted-path fields automatically resolve to None. Facts may contain nested structures, lists, numbers, booleans, and strings.
Example structure:
facts:
  - user:
      age: 25
      country: "US"
      status: "ACTIVE"
  - user:
      age: 16
      country: "CA"
      status: "INACTIVE"
Facts should include only fields referenced by rules or LET variables. The model must output a facts block only when explicitly requested.
In SESL, a scenario is one item inside the facts: block.
A scenario represents one complete input dataset used to test rule evaluation.
A scenario defines the values of all fields that may be referenced by:
  rule conditions
  LET variables
  operand lookups
  dotted-path expressions (user.age, account.balance, etc.)

"""


# ============================================================
# TOOL 2: lint_sesl
# ============================================================

@mcp.tool()
async def lint_sesl(contents: List[TextContent]) -> List[TextContent]:
    """
    Lint SESL YAML content.
    `contents` is a list of TextContent segments that will be concatenated.
    Returns a single TextContent with JSON:
    {
      "issues": [
        { "level": "...", "message": "...", "rule": "..." },
        ...
      ]
    }
    """

    raw = "".join(c.text for c in contents)
    logger.info("lint_sesl called")
    logger.debug("lint_sesl input YAML: %r", raw)

    try:
        if not raw.strip():
            issues = [{
                "level": "error",
                "message": "Empty SESL",
                "rule": "parser"
            }]
            return [make_issues_content(issues)]

        issues_obj = lint_model_from_yaml(raw)

        issues = [
            {
                "level": i.level,
                "message": i.message,
                "rule": i.rule,
            }
            for i in issues_obj
        ]

        return [make_issues_content(issues)]

    except Exception as e:
        logger.exception("Exception during lint_sesl")
        issues = [{
            "level": "error",
            "message": f"Exception during lint: {format_error(e)}",
            "rule": "internal"
        }]
        return [make_issues_content(issues)]


# ============================================================
# TOOL 3: run_sesl
# ============================================================

def _normalize_facts_input(facts: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Normalize the 'facts' parameter into a flat dict representing a single scenario.

    Supports a few flexible shapes:
    - {} or None -> {}
    - {"facts": [ {...}, ... ]} -> first element
    - {"scenarios": [ {...}, ... ]} -> first element
    - {"x": 1, "y": 2} -> used as-is

    The MCP client can therefore send:
      facts={ "temperature": 31 }
    or
      facts={ "scenarios": [ { "temperature": 31 }, { "temperature": 15 } ] }
    """
    if not facts:
        return {}

    # If we got an envelope with 'facts' list
    if "facts" in facts and isinstance(facts["facts"], list) and facts["facts"]:
        first = facts["facts"][0]
        if isinstance(first, dict):
            return first

    # Or with 'scenarios' list
    if "scenarios" in facts and isinstance(facts["scenarios"], list) and facts["scenarios"]:
        first = facts["scenarios"][0]
        if isinstance(first, dict):
            return first

    # Otherwise assume it's already a dict of fields
    return facts


@mcp.tool()
async def run_sesl(
    contents: List[TextContent],
    facts: Dict[str, Any],
) -> List[TextContent]:
    """
    Execute SESL rules against a provided facts object.

    Input:
      contents: list of TextContent, concatenated into a SESL YAML string
      facts: dict providing the fields used by rules

    Output (on success):
      [
        {
          "type": "text",
          "text": "{\n  \"result\": { ... }\n}"
        }
      ]

    Output (on error):
      [
        {
          "type": "text",
          "text": "{\n  \"error\": \"...\"\n}"
        }
      ]
    """

    raw = "".join(c.text for c in contents)
    base_facts = _normalize_facts_input(facts)

    logger.info("run_sesl called")
    logger.debug("run_sesl input YAML: %r", raw)
    logger.debug("run_sesl input facts: %r", base_facts)

    try:
        if not raw.strip():
            return [make_error_content("Empty SESL")]

        # --------------------------------------------------------
        # STEP 1: Load original SESL YAML to ensure it's valid
        # --------------------------------------------------------
        try:
            rules, _scenarios = load_model_from_yaml(raw)
        except Exception as e:
            logger.warning("Failed to load SESL model from raw YAML: %s", format_error(e))
            return [make_error_content(format_error(e))]

        # --------------------------------------------------------
        # STEP 2: Construct a runtime model that SESL can execute
        # --------------------------------------------------------
        try:
            runtime_facts: Dict[str, Any] = {
                "scenario": "runtime",
                **(base_facts or {}),
                "result": {},
            }

            core_model: Dict[str, Any] = {
                "rules": [],
                "facts": [runtime_facts],
            }

            for r in rules:
                entry: Dict[str, Any] = {"rule": r.name}

                if getattr(r, "conditions", None):
                    entry["if"] = r.conditions
                if getattr(r, "actions", None):
                    entry["then"] = r.actions
                if getattr(r, "because", None):
                    entry["because"] = r.because
                if getattr(r, "priority", None) is not None:
                    entry["priority"] = r.priority
                if getattr(r, "lets", None):
                    # We don't want to reconstruct compiled code here; just mark presence.
                    entry["let"] = {k: "(compiled)" for k in r.lets}
                if getattr(r, "stop_on_fire", None):
                    entry["stop"] = True

                core_model["rules"].append(entry)

            rebuilt_yaml = yaml.safe_dump(core_model, sort_keys=False)
            logger.debug("run_sesl rebuilt YAML: %r", rebuilt_yaml)

        except Exception as e:
            logger.exception("Exception while constructing runtime SESL model")
            return [make_error_content(format_error(e))]

        # --------------------------------------------------------
        # STEP 3: Reload runtime SESL model
        # --------------------------------------------------------
        try:
            rules, scenarios = load_model_from_yaml(rebuilt_yaml)
            if not scenarios:
                raise RuntimeError("No scenarios found after rebuilding SESL model")
            _, facts_obj = scenarios[0]
        except Exception as e:
            logger.exception("Exception while reloading runtime SESL model")
            return [make_error_content(format_error(e))]

        # --------------------------------------------------------
        # STEP 4: Execute forward chaining
        # --------------------------------------------------------
        try:
            monitor = Monitor()
            forward_chain(rules, facts_obj, monitor=monitor)
            result = facts_obj.get("result", {})
            logger.debug("run_sesl result: %r", result)
            return [make_text_content(result)]
        except Exception as e:
            logger.exception("Exception during SESL forward chaining")
            return [make_error_content(format_error(e))]

    except Exception as e:
        logger.exception("Unexpected exception in run_sesl")
        return [make_error_content(format_error(e))]


# ============================================================
# Server Entrypoint
# ============================================================

def main() -> None:
    logger.info(
        "🌟 SESL MCP Server Running...\n"
        "   Endpoint: http://localhost:3000/mcp\n"
        "   Use ngrok/cloudflared for remote access.\n"
    )

    # Avoid uvicorn shutdown hangs in some environments
    os.environ["UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN"] = "0"

    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=3000,
    )


if __name__ == "__main__":
    main()
