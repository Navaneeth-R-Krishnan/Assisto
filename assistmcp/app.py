from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from mcp_use import MCPClient
import os
import logging
from typing import Optional, Literal
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AssistMCPError(Exception):
    """Base exception class for AssistMCP errors."""
    pass

class LLMConfigError(AssistMCPError):
    """Raised when there are issues with LLM configuration."""
    pass

def load_environment_variables(env_path: str = ".env") -> None:
    """Load environment variables from .env file."""
    env_path = Path(env_path)
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    else:
        logger.warning(f".env file not found at {env_path}")

def get_llm(provider: Literal["groq", "openai"] = "groq") -> ChatGroq | ChatOpenAI:
    """
    Get LLM instance based on provider.
    
    Args:
        provider: The LLM provider to use ("groq" or "openai")
        
    Returns:
        Configured LLM instance
        
    Raises:
        LLMConfigError: If there are configuration issues
    """
    try:
        if provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise LLMConfigError("GROQ_API_KEY environment variable not set")
            return ChatGroq(
                api_key=api_key,
                model_name="mixtral-8x7b-32768",
                temperature=0.7,
                max_tokens=32768
            )
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise LLMConfigError("OPENAI_API_KEY environment variable not set")
            return ChatOpenAI(
                temperature=0.7,
                model_name="gpt-4-turbo-preview"
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    except Exception as e:
        logger.error(f"Error initializing LLM with provider {provider}: {str(e)}")
        raise LLMConfigError(f"Failed to initialize LLM: {str(e)}")

class AssistMCP:
    """Main application class for AssistMCP."""
    
    def __init__(
        self,
        mcp_config_path: str = "browser_mcp.json",
        llm_provider: Literal["groq", "openai"] = "groq"
    ):
        self.mcp_config_path = Path(mcp_config_path)
        self.llm_provider = llm_provider
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize MCP client and LLM."""
        try:
            if not self.mcp_config_path.exists():
                raise AssistMCPError(f"MCP config file not found: {self.mcp_config_path}")
            
            self.mcp = MCPClient(config_path=str(self.mcp_config_path))
            self.llm = get_llm(self.llm_provider)
            logger.info(f"Initialized AssistMCP with {self.llm_provider} LLM")
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            raise AssistMCPError(f"Failed to initialize AssistMCP: {str(e)}")
    
    def run(self) -> None:
        """Main application loop."""
        try:
            # Your main application logic here
            logger.info("AssistMCP running...")
            pass
        except Exception as e:
            logger.error(f"Runtime error: {str(e)}")
            raise AssistMCPError(f"Runtime error: {str(e)}")

def main():
    """Entry point for the application."""
    try:
        # Load environment variables
        load_environment_variables()
        
        # Initialize and run application
        app = AssistMCP()
        app.run()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()

