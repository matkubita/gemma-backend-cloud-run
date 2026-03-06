import os
from pathlib import Path

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.agents import SequentialAgent
from google.adk.models.lite_llm import LiteLlm
import google.auth
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StreamableHTTPConnectionParams

# Load environment variables
root_dir = Path(__file__).parent.parent
dotenv_path = root_dir / ".env"
load_dotenv(dotenv_path=dotenv_path)

# Configure Google Cloud
try:
    _, project_id = google.auth.default()
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
except Exception:
    pass

os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "europe-west1")

# Configure model connection
gemma_model_name = os.getenv("GEMMA_MODEL_NAME", "gemma3:270m")
model_name = os.getenv("MODEL")
api_base = os.getenv("OLLAMA_API_BASE", "localhost:10010")  # Location of Ollama server

mcp_server_url = os.getenv("MCP_SERVER_URL")
mcp_tools = MCPToolset(connection_params=StreamableHTTPConnectionParams(url=mcp_server_url))

# Production Gemma Agent - GPU-accelerated conversational assistant
# production_agent = Agent(
#    model=LiteLlm(model=f"ollama_chat/{gemma_model_name}", api_base=api_base),
#    name="production_agent",
#    description="A production-ready conversational assistant powered by GPU-accelerated Gemma.",
#    instruction="""You are an employee in a software startup, you take care of the marketing. Help the user by answering marketing questions. You have an MCP tool for checking the latest google trends, use it when the user asks about google trends""",
#    tools=[mcp_tools],  # Gemma focuses on conversational capabilities
# )

def add_prompt_to_state(
    tool_context: ToolContext, prompt: str
) -> dict[str, str]:
    """Saves the user's initial prompt to the state."""
    tool_context.state["PROMPT"] = prompt
    return {"status": "success"}

comprehensive_researcher = Agent(
    name="comprehensive_researcher",
    model=model_name,
    description="The primary researcher that can access data of Google Trends",
    instruction="""
    You are a helpful research assistant. Your goal is to fully answer the user's PROMPT.
    You have access to one tool:
    1. A tool for getting data about latest google trends

    First, analyze the user's PROMPT.
    - Use that tool.
    - Synthesize the results from the tool you use into preliminary data outputs.

    PROMPT:
    {{ PROMPT }}
    """,
    tools=[
        mcp_tools,
    ],
    output_key="google_trends_data" # A key to store the combined findings
)

response_formatter = Agent(
    name="response_formatter",
    model=model_name,
    description="Synthesizes all information into a friendly, readable response.",
    instruction="""
    You are the friendly voice of the Marketing department. Your task is to take the
    RESEARCH_DATA and present it to the user in a complete and helpful answer.

    - Be conversational and engaging.

    RESEARCH_DATA:
    {{ google_trends_data }}
    """
)

marketing_guide_workflow = SequentialAgent(
    name="marketing_guide_workflow",
    description="The main workflow for handling a user's request about marketing.",
    sub_agents=[
        comprehensive_researcher, # Step 1: Gather all data
        response_formatter,       # Step 2: Format the final response
    ]
)

# Set as root agent
root_agent = Agent(
    name="greeter",
    model=model_name,
    description="The main entry point for the Marketing department.",
    instruction="""
    - Let the user know you will help them learn about the Marketing.
    - When the user responds, use the 'add_prompt_to_state' tool to save their response.
    After using the tool, transfer control to the 'marketing_guide_workflow' agent.
    """,
    tools=[add_prompt_to_state],
    sub_agents=[marketing_guide_workflow]
)