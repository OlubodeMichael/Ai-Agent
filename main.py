from dotenv import load_dotenv
load_dotenv()

from langsmith import Client
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from schemas import AgentResponse
from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4", temperature=0)
hub_client = Client()
react_prompt = hub_client.pull_prompt("hwchase17/react")

output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tool_names"],
).partial(format_instructions=output_parser.get_format_instructions())

# Convert the prompt template to a string for the system prompt
system_prompt = react_prompt_with_format_instructions.format(
    tools="\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
    tool_names=", ".join([tool.name for tool in tools]),
    input="{input}",
    agent_scratchpad="{agent_scratchpad}"
)

# Create agent using the new API
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
)

extract_output = RunnableLambda(lambda x: x.get("output"))
parse_output = RunnableLambda(lambda x: output_parser.parse(x))

chain = agent | extract_output | parse_output


def main():
    result = agent.invoke({
        "messages": [{"role": "user", "content": "search 3 jobs for software engineer in San Francisco"}]
    })
    print(result)


if __name__ == "__main__":
    main()
