📘 SESL-MCP — SESL Model Computation Protocol Server

sesl-mcp is an MCP (Model Context Protocol) server built on FastMCP that exposes SESL capabilities over a standard MCP interface. It enables:

✔ SESL rule generation

✔ SESL linting

✔ SESL execution (forward-chaining rule engine)

✔ JSON-based tool responses

✔ HTTP-based MCP transport

This server is designed to work with the SESL library, located at:

👉 https://github.com/vpowell4/sesl

🚀 Features
Tool	Description
generate_sesl(prompt)	Converts natural-language descriptions into SESL YAML
lint_sesl(contents)	Validates SESL YAML and returns structured error messages
run_sesl(contents, facts)	Executes SESL rules using the SESL engine and returns computed results
📦 Installation
Option 1 — Clone & run locally (recommended)
git clone https://github.com/vpowell4/sesl-mcp.git
cd sesl-mcp
uv sync
uv run sesl-mcp-server


This starts the MCP server on:

http://0.0.0.0:3000/mcp

Option 2 — Install directly from GitHub

Requires uv or pip:

uv pip install "sesl-mcp @ git+https://github.com/vpowell4/sesl-mcp.git@main"


Run it globally:

sesl-mcp-server

Option 3 — Add as a dependency to another project

In another uv project:

uv add "sesl-mcp @ git+https://github.com/vpowell4/sesl-mcp.git@main"


Then run:

uv run sesl-mcp-server

🛠️ Running the MCP Server

Once installed:

sesl-mcp-server


You should see:

🌟 SESL MCP Server Running...
Endpoint: http://localhost:3000/mcp
Use ngrok/cloudflared for remote access.


To expose the MCP server publicly:

cloudflared tunnel --url http://localhost:3000


or:

ngrok http 3000

🧩 MCP Tools (API)

The server exposes 3 tools.

1. generate_sesl(prompt: str)

Generates valid SESL YAML from natural language instructions.

Example call:

{
  "tool": "generate_sesl",
  "arguments": { "prompt": "Approve loan if credit score > 700" }
}

2. lint_sesl(contents: List[TextContent])

Validates SELS YAML.

Example request:

rule: IsAdult
priority: 10
if: user.age >= 18
then:
  is_adult: true
because: "User is an adult"


Returns structured JSON:

{
  "issues": [
    { "level": "error", "message": "...", "rule": "parser" }
  ]
}

3. run_sesl(contents, facts)

Executes SESL YAML with provided runtime facts.

Example:

{
  "contents": [{ "text": "rule: TestRule ..." }],
  "facts": { "user": { "age": 25 } }
}


Returns:

{ "is_adult": true }

📁 Project Structure
sesl-mcp/
│ pyproject.toml
│ README.md
│ .gitignore
└── src/
    └── sesl_mcp/
        ├── __init__.py
        └── server.py

🔧 Development

Clone the repo:

git clone https://github.com/vpowell4/sesl-mcp.git
cd sesl-mcp
uv sync


Run:

uv run sesl-mcp-server

🧪 Updating SESL dependency

To update to a new SESL version:

uv lock --upgrade-package sesl


Push changes:

git add pyproject.toml uv.lock
git commit -m "Upgrade SESL dependency"
git push

🏷️ Versioning

Tag releases:

git tag v0.1.0
git push --tags


Consumers can then install:

sesl-mcp @ git+https://github.com/vpowell4/sesl-mcp.git@v0.1.0

📣 Contributing

Fork the repository

Create a feature branch

Commit your changes

Open a pull request

❗ Troubleshooting
❌ uv hardlink errors

If using OneDrive:

setx UV_LINK_MODE copy
uv sync

❌ “program not found: sesl-mcp-server”

Project wasn’t installed.

Run:

uv sync --force

❌ SESL import errors

Ensure SESL installed:

uv pip install "sesl @ git+https://github.com/vpowell4/sesl.git@main"

🙌 Thanks

This project is part of the SESL rule engine ecosystem.
Feedback & contributions welcome!