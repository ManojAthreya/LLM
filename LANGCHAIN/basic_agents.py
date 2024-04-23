import getpass
import os
from langchain_groq import ChatGroq
import warnings
from dotenv import load_dotenv
import nest_asyncio
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from operator import itemgetter
from typing import Dict, List, Union
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent

warnings.simplefilter("ignore")
load_dotenv()
nest_asyncio.apply()

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

prompt = hub.pull("hwchase17/openai-tools-agent")

@tool
def multiply(int1:int, int2:int) -> int:
    """Multiply two integers together."""
    return int1*int2

@tool
def add(int1:int, int2:int) -> int:
    "Add two integers."
    return int1+int2

@tool
def exponentiate(base:int, exponent:int) -> int:
    "Exponentiate the base to the exponent power."
    return base**exponent

llm = ChatGroq(model_name='llama3-70b-8192', temperature=0)
#llm = ChatOpenAI(model="gpt-3.5-turbo")


tools = [multiply, exponentiate, add]
# llm_with_tools = llm.bind_tools(tools)
# tool_map = {tool.name: tool for tool in tools}


# def call_tools(msg: AIMessage) -> Runnable:
#     """Simple sequential tool calling helper."""
#     tool_map = {tool.name: tool for tool in tools}
#     tool_calls = msg.tool_calls.copy()
#     for tool_call in tool_calls:
#         tool_call["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"])
#     return tool_calls


# chain = llm_with_tools | call_tools

# print(chain.invoke("cube thirty-seven"))

agent = create_openai_tools_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


agent_executor.invoke(
    {
        "input": "Take 3 to the fifth power and multiply that by the sum of twelve and three, then square the whole result"
    }
)

