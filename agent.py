import json
import os

from dotenv import load_dotenv
from morphik import Morphik
from openai import OpenAI

from tools import build_tools, run_tool_call

load_dotenv()

morphik = Morphik(uri=os.getenv("MORPHIK_URI"))
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
SYSTEM_INSTRUCTIONS = (
    "You are a helpful assistant with access to Morphik retrieval tools and the "
    "built-in code interpreter. Use retrieve_chunks for semantic search, "
    "get_page_range for page/chunk ranges, list_documents to browse documents, "
    "and load_file_for_execution to load files for Python analysis. After loading "
    "a file, use its returned filename in the code interpreter."
)


def _json_dumps(value: object) -> str:
    return json.dumps(value, default=str, ensure_ascii=True)


def _collect_function_calls(response) -> list:
    return [item for item in response.output if item.type == "function_call"]


def run_agent(query: str) -> str:
    state = {"file_ids": set(), "loaded_files": {}}

    response = openai.responses.create(
        model=MODEL,
        instructions=SYSTEM_INSTRUCTIONS,
        input=[{"role": "user", "content": query}],
        tools=build_tools(state["file_ids"]),
    )

    while True:
        tool_calls = _collect_function_calls(response)
        if not tool_calls:
            return response.output_text

        tool_outputs = []
        for call in tool_calls:
            try:
                args = json.loads(call.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            try:
                result = run_tool_call(
                    call.name,
                    args,
                    morphik=morphik,
                    openai_client=openai,
                    state=state,
                )
                output = _json_dumps(result)
            except Exception as exc:
                output = _json_dumps({"error": str(exc)})

            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": output,
                }
            )

        response = openai.responses.create(
            model=MODEL,
            instructions=SYSTEM_INSTRUCTIONS,
            input=tool_outputs,
            previous_response_id=response.id,
            tools=build_tools(state["file_ids"]),
        )


def main() -> None:
    query = input("Query: ").strip()
    if not query:
        print("No query provided.")
        return

    response_text = run_agent(query)
    with open("response.md", "w", encoding="utf-8") as handle:
        handle.write(response_text)
    print("Response saved to response.md")


if __name__ == "__main__":
    main()
