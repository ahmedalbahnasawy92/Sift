"""
Router + specialist agents + tools, using Agno via OpenRouter.

Flow:
  user question -> Router team (cheap model, mode=route: picks ONE member)
                -> General assistant (tools: get_weather, get_current_time)
                   or Expert assistant (tool: calculate)
                -> member's answer returned directly

Setup:
  pip install agno openai httpx python-dotenv
  .env file containing: OPENROUTER_API_KEY=...
"""

import ast
import asyncio
import operator
from datetime import datetime
from zoneinfo import ZoneInfo

import dotenv
import httpx
from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.team import Team
from agno.team.mode import TeamMode

dotenv.load_dotenv()  # OpenRouter() reads OPENROUTER_API_KEY from the env

# ---------------------------------------------------------------------------
# Models (Agno has a native OpenRouter class; no client wiring needed)
# Note: Agno's default max_tokens is 1024, too low for code answers.
# ---------------------------------------------------------------------------
deepseek_flash = OpenRouter(id="deepseek/deepseek-v4-flash", max_tokens=4096)
glm = OpenRouter(id="z-ai/glm-5.2", max_tokens=8192)


# ---------------------------------------------------------------------------
# Tools: plain functions. Name -> tool name, type hints -> schema,
# docstring -> description. No decorator needed (use @tool for extra options).
# ---------------------------------------------------------------------------
async def get_weather(city: str) -> str:
    """Get the CURRENT weather for a city. Use for any question about
    weather or temperature right now. Do NOT use for forecasts or climate.

    Args:
        city: City name, e.g. "Tokyo".
    """
    try:
        async with httpx.AsyncClient(timeout=10) as http:
            geo = await http.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1},
            )
            results = geo.json().get("results")
            if not results:
                return f"Could not find a city named '{city}'."
            place = results[0]

            wx = await http.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "current_weather": True,
                },
            )
            current = wx.json()["current_weather"]
        return (
            f"{place['name']}, {place.get('country', '')}: "
            f"{current['temperature']}°C, wind {current['windspeed']} km/h"
        )
    except Exception as e:  # return errors as text instead of raising
        return f"Weather lookup failed: {e}"


def get_current_time(timezone: str) -> str:
    """Get the current date and time in an IANA timezone.
    Use for 'what time is it' questions.

    Args:
        timezone: IANA timezone such as "Asia/Dubai" or "Asia/Tokyo".
    """
    try:
        now = datetime.now(ZoneInfo(timezone))
        return now.strftime(f"%Y-%m-%d %H:%M:%S ({timezone})")
    except Exception:
        return f"Unknown timezone '{timezone}'. Use an IANA name like 'Europe/London'."


# Safe arithmetic evaluator (never use eval() on model-provided text)
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression. Supports + - * / ** %.
    Use for exact calculations instead of doing arithmetic yourself.
    Do NOT use for anything that isn't plain arithmetic.

    Args:
        expression: e.g. "(12 + 8) * 3 / 4".
    """
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"Could not evaluate '{expression}': {e}"


# ---------------------------------------------------------------------------
# Specialist agents  (handoff_description -> role)
# ---------------------------------------------------------------------------
general_agent = Agent(
    name="General assistant",
    role=(
        "Everyday questions, chit-chat, quick facts, weather, current time, "
        "short answers."
    ),
    instructions=(
        "Answer clearly and concisely. For weather or time questions you MUST "
        "use the tools; never guess. If you lack a tool for something live "
        "(news, prices, recent releases), say you can't verify it instead of "
        "making it up."
    ),
    model=deepseek_flash,
    tools=[get_weather, get_current_time],
    tool_call_limit=5,  # like max_turns, per agent
)

expert_agent = Agent(
    name="Expert assistant",
    role=(
        "Coding, math, multi-step reasoning, analysis, comparisons, "
        "long or complex tasks."
    ),
    instructions=(
        "Think step by step and give a thorough, accurate answer. Use the "
        "calculate tool for exact arithmetic. If a question depends on recent "
        "information you can't verify, say so."
    ),
    model=glm,
    tools=[calculate],
    tool_call_limit=5,
)

# ---------------------------------------------------------------------------
# Router: a Team in route mode. The leader must delegate to ONE member and
# the member's answer is returned as-is (the leader never answers itself).
# ---------------------------------------------------------------------------
router_team = Team(
    name="Router",
    mode=TeamMode.route,
    model=deepseek_flash,
    members=[general_agent, expert_agent],
    instructions=(
        "You are a router. Do NOT answer the user yourself. "
        "Route simple/general requests, weather and time questions to "
        "'General assistant'. Route coding, math, reasoning, comparisons "
        "or complex requests to 'Expert assistant'."
    ),
    determine_input_for_members=False,  # pass the user's question unchanged
    tool_choice="required",             # must call the delegate tool
)


# ---------------------------------------------------------------------------
# Debug helper: show routing + tool calls, with arguments
# ---------------------------------------------------------------------------
def print_trace(result) -> None:
    for t in result.tools or []:  # leader's calls (the delegation)
        print(f"  [Route] {t.tool_name}({t.tool_args})")
    for member in result.member_responses or []:
        for t in member.tools or []:
            print(f"  [{member.agent_name}] {t.tool_name}({t.tool_args})")
            print(f"  [ToolOutput] {str(t.result)[:200]}")


def handled_by(result) -> str:
    members = result.member_responses or []
    return members[-1].agent_name if members else "Router"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
QUESTIONS = [
    "What's the capital of Japan?",
    "Write a Python function that finds the longest palindromic substring.",
    "What is the weather in Tokyo?",
    "What is the weather in Dubai?",
    "What time is it in Dubai right now?",
    "What is (1234 * 56) / 7 plus 2 to the power of 10?",
    "What is the difference between litellm and the Vercel AI SDK?",
]


async def main() -> None:
    for q in QUESTIONS:
        result = await router_team.arun(q)
        print(f"\nQ: {q}")
        print(f"Handled by: {handled_by(result)}")
        print_trace(result)
        print(f"A: {result.content}")


if __name__ == "__main__":
    asyncio.run(main())
