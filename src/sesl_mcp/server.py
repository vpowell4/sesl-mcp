
import json
import yaml
import traceback
from typing import Any, Optional, List, Dict

from fastmcp import FastMCP
from mcp.types import TextContent


# SESL Engine + Linter
from sesl.engine.rule_engine import (
    load_model_from_yaml,
    forward_chain,
    Monitor,
)
from sesl.tools.linter_core import lint_model_from_yaml


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

    print("\n=== generate_sesl CALLED ===")
    print("PROMPT:", prompt)

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
user:
age: 25
country: "US"
status: "ACTIVE"
user:
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
    contents[] = Text segments containing SESL YAML.
    """
    raw = "".join(c.text for c in contents)
    print("\n=== lint_sesl CALLED ===")
    print("INPUT YAML:", repr(raw))

    try:
        if not raw.strip():
            payload = {
                "issues": [
                    {"level": "error", "message": "Empty SESL", "rule": "parser"}
                ]
            }
            return [TextContent(text=json.dumps(payload, indent=2))]

        issues = lint_model_from_yaml(raw)

        payload = {
            "issues": [
                {"level": i.level, "message": i.message, "rule": i.rule}
                for i in issues
            ]
        }

        return [TextContent(text=json.dumps(payload, indent=2))]

    except Exception as e:
        return [TextContent(text=json.dumps({
            "issues": [{
                "level": "error",
                "message": f"Exception during lint: {format_error(e)}",
                "rule": "internal"
            }]
        }, indent=2))]


# ============================================================
# TOOL 3: run_sesl
# ============================================================
@mcp.tool()
async def run_sesl(
    contents: List[TextContent],
    facts: Dict[str, Any],
) -> List[TextContent]:

    raw = "".join(c.text for c in contents)

    print("\n=== run_sesl CALLED ===")
    print("YAML:", repr(raw))
    print("FACTS:", facts)

    try:
        if not raw.strip():
            return [TextContent(text=json.dumps({"error": "Empty SESL"}, indent=2))]

        # Load original YAML
        try:
            rules, scenarios = load_model_from_yaml(raw)
        except Exception as e:
            return [TextContent(text=json.dumps({
                "error": format_error(e)
            }, indent=2))]

        # Build runtime facts
        try:
            runtime = {"scenario": "runtime", **(facts or {}), "result": {}}
            core = {"rules": [], "facts": [runtime]}

            for r in rules:
                entry = {"rule": r.name}
                if getattr(r, "conditions", None):
                    entry["if"] = r.conditions
                if getattr(r, "actions", None):
                    entry["then"] = r.actions
                if getattr(r, "because", None):
                    entry["because"] = r.because
                if getattr(r, "priority", None):
                    entry["priority"] = r.priority
                if getattr(r, "lets", None):
                    entry["let"] = {k: "(compiled)" for k in r.lets}
                if getattr(r, "stop_on_fire", None):
                    entry["stop"] = True
                core["rules"].append(entry)

            rebuilt_yaml = yaml.safe_dump(core, sort_keys=False)

        except Exception as e:
            return [TextContent(text=json.dumps({
                "error": format_error(e)
            }, indent=2))]

        # Reload runtime model
        try:
            rules, scenarios = load_model_from_yaml(rebuilt_yaml)
            _, facts_obj = scenarios[0]
        except Exception as e:
            return [TextContent(text=json.dumps({
                "error": format_error(e)
            }, indent=2))]

        # Execute forward chaining
        try:
            monitor = Monitor()
            forward_chain(rules, facts_obj, monitor=monitor)
            result = facts_obj.get("result", {})
            return [TextContent(text=json.dumps(result, indent=2))]
        except Exception as e:
            return [TextContent(text=json.dumps({
                "error": format_error(e)
            }, indent=2))]

    except Exception as e:
        return [TextContent(text=json.dumps({
            "error": format_error(e)
        }, indent=2))]


# ============================================================
# Server Entrypoint
# ============================================================

def main():
    print(
        "🌟 SESL MCP Server Running...\n"
        "   Endpoint: http://localhost:3000/mcp\n"
        "   Use ngrok/cloudflared for remote access.\n"
    )

    import os
    os.environ["UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN"] = "0"

    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=3000,
    )



if __name__ == "__main__":
    main()