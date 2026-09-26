import asyncio
import logging
import os

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from src.domain.ports.jules_mcp_port import IJulesMcpAdapter

logger = logging.getLogger(__name__)


class JulesMcpAdapter(IJulesMcpAdapter):
    def ask_jules(self, prompt: str) -> str:
        try:
            return asyncio.run(self._ask_jules_async(prompt))
        except Exception as e:  # noqa: BLE001
            logger.error("Error communicating with Jules MCP Server: %s", e)
            return f"❌ Error communicating with Jules MCP Server: {e}"

    async def _ask_jules_async(self, prompt: str) -> str:
        env = os.environ.copy()

        server_params = StdioServerParameters(
            command="node",
            args=["/opt/homebrew/lib/node_modules/jules-mcp-server/dist/server.js"],
            env=env,
        )

        async with (
            stdio_client(server_params) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()

            # List tools to find the correct tool name to call
            tools_response = await session.list_tools()
            if not tools_response or not tools_response.tools:
                return "No tools available on the Jules MCP Server."

            # Assume we call the first available tool for delegation, or 'ask_jules' if present
            tool_name = tools_response.tools[0].name
            for t in tools_response.tools:
                if (
                    "ask" in t.name.lower()
                    or "delegate" in t.name.lower()
                    or "jules" in t.name.lower()
                ):
                    tool_name = t.name
                    break

            try:
                # Some tools might require 'task' instead of 'prompt', try 'prompt' first
                result = await session.call_tool(tool_name, {"prompt": prompt})
                return str(result.content) if hasattr(result, "content") else str(result)
            except Exception:  # noqa: BLE001
                try:
                    # Fallback to 'task' if 'prompt' fails
                    result = await session.call_tool(tool_name, {"task": prompt})
                    return str(result.content) if hasattr(result, "content") else str(result)
                except Exception as inner_e:  # noqa: BLE001
                    logger.error("Error calling tool on MCP: %s", inner_e)
                    return f"Error executing task on Jules MCP: {inner_e}"
