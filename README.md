Morphik Code Exec Demo

This demo ingests Excel files into Morphik and runs an OpenAI Responses API agent
that can retrieve chunks, list documents, pull page/chunk ranges, and load files
into the code interpreter.

Prerequisites
- Python 3.11+
- uv installed
- Morphik URI
- OpenAI API key

Setup
1) Create a `.env` file with your credentials:

```bash
MORPHIK_URI="morphik://<owner_id>:<token>@<host>"
OPENAI_API_KEY="sk-..."
# Optional: model override
OPENAI_MODEL="gpt-4.1"
```

2) Install dependencies:

```bash
uv sync
```

Run the demo (ingest -> status -> agent)
1) Ingest files from `files/`:

```bash
uv run ingest.py
```

2) Check ingestion status:

```bash
uv run status.py
```

3) Run the agent:

```bash
uv run agent.py
```

The agent will prompt for a query and write the final response to `response.md`.
