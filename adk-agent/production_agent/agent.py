import os
from pathlib import Path

from dotenv import load_dotenv
from google.adk.agents import Agent
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
api_base = os.getenv("OLLAMA_API_BASE", "localhost:10010")  # Location of Ollama server

mcp_server_url = os.getenv("MCP_SERVER_URL")
mcp_tools = MCPToolset(connection_params=StreamableHTTPConnectionParams(url=mcp_server_url))

# Production Gemma Agent - GPU-accelerated conversational assistant
production_agent = Agent(
   model=LiteLlm(model=f"ollama_chat/{gemma_model_name}", api_base=api_base),
   name="production_agent",
   description="A production-ready conversational assistant powered by GPU-accelerated Gemma.",
   instruction="""You are Gem, a Growth Marketing Lead at a startup who uses Google Trends to spot 
   the next big thing. You provide data-driven insights on search patterns, audience intent, and 
   content strategy to help the team win. Since you lack access to live internal dashboards or 
   private CRM data, you rely on your deep general knowledge of digital consumer behavior. 
   Always keep your tone caffeinated, collaborative, and results-oriented—like a peer with their finger on the internet's pulse. 🦁✨""",
   tools=[],  # Gemma focuses on conversational capabilities
)

# Set as root agent
root_agent = production_agent