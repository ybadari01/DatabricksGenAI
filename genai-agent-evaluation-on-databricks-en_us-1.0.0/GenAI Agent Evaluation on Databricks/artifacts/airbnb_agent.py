import json
import warnings
from typing import Any, Callable, Generator
from uuid import uuid4

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_openai import UCFunctionToolkit, VectorSearchRetrieverTool
from mlflow.entities import SpanType, Document
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)
from openai import OpenAI
from pydantic import BaseModel
from unitycatalog.ai.core.base import get_uc_function_client

try: 
    model_config=mlflow.models.ModelConfig()
except:
    config_file="configs/agent_config.yaml"
    model_config=mlflow.models.ModelConfig(development_config=config_file)

SYSTEM_PROMPT = model_config.get("SYSTEM_PROMPT")
LLM_ENDPOINT_NAME=model_config.get("LLM_ENDPOINT_NAME")
CATALOG_NAME = model_config.get("CATALOG_NAME")
SCHEMA_NAME = model_config.get("SCHEMA_NAME")
TOOL1 = model_config.get("TOOL1")
TOOL2 = model_config.get("TOOL2")
# INDEX_NAME = model_config.get("INDEX_NAME")
# INDEX_DESCRIPTION = model_config.get("INDEX_DESCRIPTION")


###############################################################################
# Tool definitions and helpers
###############################################################################
class ToolInfo(BaseModel):
    name: str
    spec: dict
    exec_fn: Callable
    is_retriever: bool = False

def create_tool_info(
    tool_spec: dict,
    exec_fn_param: Callable | None = None,
    *,
    is_retriever: bool = False,
) -> ToolInfo:
    tool_spec["function"].pop("strict", None)
    tool_name = tool_spec["function"]["name"]
    udf_name = tool_name.replace("__", ".")
    uc_function_client = get_uc_function_client()

    def exec_fn(**kwargs):
        function_result = uc_function_client.execute_function(udf_name, kwargs)
        if function_result.error is not None:
            return function_result.error
        else:
            return function_result.value

    return ToolInfo(
        name=tool_name,
        spec=tool_spec,
        exec_fn=exec_fn_param or exec_fn,
        is_retriever=is_retriever,
    )

###############################################################################
# Build tools at module level
###############################################################################
TOOL_INFOS: list[ToolInfo] = []

# Unity Catalog tools
uc_tool_names = [f"{CATALOG_NAME}.{SCHEMA_NAME}.{TOOL1}", f"{CATALOG_NAME}.{SCHEMA_NAME}.{TOOL2}"]

try:
    uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)
    for tool_spec in uc_toolkit.tools:
        TOOL_INFOS.append(create_tool_info(tool_spec, is_retriever=False))
except Exception:
    # During MLflow validation, tools might not be accessible
    warnings.warn(f"UC tool init failed: {e}")

# Vector Search tools
# try:
#     vs_tool = VectorSearchRetrieverTool(
#         index_name=INDEX_NAME,
#         tool_description=INDEX_DESCRIPTION,
#     )
#     TOOL_INFOS.append(create_tool_info(vs_tool.tool, vs_tool.execute, is_retriever=True))
# except Exception:
#     # During MLflow validation, vector search might not be accessible
#     pass

###############################################################################
# Agent implementation
###############################################################################
class ToolCallingAgent(ResponsesAgent):
    """Agent that calls tools based on LLM responses"""
    
    def __init__(self, llm_endpoint: str, tools: list[ToolInfo]):
        self.llm_endpoint = llm_endpoint
        self.workspace_client = WorkspaceClient()
        self.model_serving_client: OpenAI = self.workspace_client.serving_endpoints.get_open_ai_client()
        self._tools_dict = {tool.name: tool for tool in tools}

    def get_tool_specs(self) -> list[dict]:
        return [tool_info.spec for tool_info in self._tools_dict.values()]

    def _to_documents(self, result: Any) -> list[Document]:
        docs: list[Document] = []
        if isinstance(result, list):
            for d in result:
                if isinstance(d, dict):
                    text = d.get("page_content") or d.get("content") or ""
                    uri = d.get("doc_uri") or (d.get("metadata") or {}).get("doc_uri")
                    docs.append(Document(page_content=text, metadata={"doc_uri": uri} if uri else {}))
        return docs

    def call_llm(self, messages: list[dict[str, Any]]) -> Generator[dict[str, Any], None, None]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"PydanticSerializationUnexpectedValue", category=UserWarning)

        for chunk in self.model_serving_client.chat.completions.create(
            model=self.llm_endpoint,
            messages=to_chat_completions_input(messages),
            tools=self.get_tool_specs(),
            stream=True,
            response_format={"type": "text"},
        ):
            cd = chunk.to_dict()
            for choice in cd.get("choices", []):
                msg = choice.get("delta") or choice.get("message") or {}
                content = msg.get("content")
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") in ("text", "output_text"):
                            parts.append(part.get("text") or part.get("output_text") or "")
                    msg["content"] = "".join(parts).strip()
            yield cd

    def execute_tool(self, tool_name: str, args: dict) -> Any:
        tool = self._tools_dict[tool_name]
        if tool.is_retriever:
            with mlflow.start_span(name=tool_name, span_type=SpanType.RETRIEVER) as span:
                result = tool.exec_fn(**args)
                try:
                    span.set_outputs(self._to_documents(result))
                except Exception:
                    pass
                return result
        else:
            with mlflow.start_span(name=tool_name, span_type=SpanType.TOOL):
                return tool.exec_fn(**args)

    def handle_tool_call(
        self,
        tool_call: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> ResponsesAgentStreamEvent:
        args = json.loads(tool_call["arguments"])
        result_obj = self.execute_tool(tool_name=tool_call["name"], args=args)
        result_text = result_obj if isinstance(result_obj, str) else json.dumps(result_obj, ensure_ascii=False)
        tool_call_output = self.create_function_call_output_item(tool_call["call_id"], result_text)
        messages.append(tool_call_output)
        return ResponsesAgentStreamEvent(type="response.output_item.done", item=tool_call_output)

    def call_and_run_tools(
        self,
        messages: list[dict[str, Any]],
        max_iter: int = 10,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        for _ in range(max_iter):
            last_msg = messages[-1]
            if last_msg.get("role", None) == "assistant":
                return
            elif last_msg.get("type", None) == "function_call":
                yield self.handle_tool_call(last_msg, messages)
            else:
                yield from output_to_responses_items_stream(
                    chunks=self.call_llm(messages),
                    aggregator=messages,
                )

        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item("Max iterations reached. Stopping.", str(uuid4())),
        )

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    def predict_stream(self, request: ResponsesAgentRequest) -> Generator[ResponsesAgentStreamEvent, None, None]:
        messages = to_chat_completions_input([i.model_dump() for i in request.input])
        if SYSTEM_PROMPT:
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        yield from self.call_and_run_tools(messages=messages)

###############################################################################
# Initialize and register the agent with MLflow
###############################################################################

#mlflow.openai.autolog()
AGENT = ToolCallingAgent(llm_endpoint=LLM_ENDPOINT_NAME, tools=TOOL_INFOS)
mlflow.models.set_model(AGENT)