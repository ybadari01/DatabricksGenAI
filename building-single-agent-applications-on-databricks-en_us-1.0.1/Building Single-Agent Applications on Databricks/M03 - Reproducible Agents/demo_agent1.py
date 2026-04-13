from databricks_langchain import ChatDatabricks, UCFunctionToolkit
from langchain.agents import create_agent  # high-level API
import json

class DatabricksAgent:
    def __init__(self, catalog_name: str, schema_name: str, config_file_path: str = "./demo_agent1_config.json"):
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        self.config_file_path = config_file_path
        self._setup_agent()

    def _setup_agent(self):
        with open(self.config_file_path, "r") as f:
            config = json.load(f)

        tool_list_raw = config["tool_list"]
        llm_endpoint = config["llm_endpoint"]
        llm_temperature = config["llm_temperature"]
        system_prompt = config["system_prompt"]

        # Build fully qualified UC function names
        function_names = [f"{tool}" for tool in tool_list_raw]

        # UC tools
        toolkit = UCFunctionToolkit(function_names=function_names)
        tools = toolkit.tools

        # LLM
        llm = ChatDatabricks(endpoint=llm_endpoint, temperature=llm_temperature)

        # Create the agent (no AgentExecutor needed)
        self.agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
        )

    def query(self, prompt: str, chat_history: list | None = None):
        # Expect messages=[...] input format for create_agent
        messages = []
        if chat_history:
            # Accept tuples like ("human","...") / ("ai","...") and map to OpenAI roles
            role_map = {"human": "user", "ai": "assistant", "system": "system"}
            for role, content in chat_history:
                messages.append({"role": role_map.get(role, role), "content": content})
        messages.append({"role": "user", "content": prompt})

        return self.agent.invoke({"messages": messages})

    def ask(self, prompt: str, chat_history: list | None = None):
        return self.query(prompt, chat_history)