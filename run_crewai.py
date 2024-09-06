import os
from typing import Literal

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from dotenv import find_dotenv
from dotenv import load_dotenv
from langchain.agents import load_tools
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatAnthropic
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from pydantic import validate_call

load_dotenv(find_dotenv(raise_error_if_not_found=True))


def build_crew_with_tools(verbose: bool = False):
    """Build a crew with all the tools we want to use."""

    llm = _get_llm("openai")
    agent_params = {
        "memory": True,
        "verbose": verbose,
        "llm": llm,
    }

    semantic_scholar_agent = Agent(
        role="Research Analyst",
        goal="Find papers that support this query: {query}",
        backstory="A resaerch analyst skilled at doing in-depth research into complex "
        "topics and discovering the truth",
        **agent_params,
    )
    semantic_scholar_task = Task(
        description="",
        expected_output="",
        agent=semantic_scholar_agent,
        tools=[SemanticScholarQueryRun()],
    )

    crew = Crew(
        agents=[semantic_scholar_agent],
        tasks=[semantic_scholar_task],
        verbose=True,
        process=Process.sequential,
    )
    return crew


def get_all_tools():
    """Return all tools we want to use.

    This is more for documentation about how to load them. We will probably
    want to give each agent a single tool e.g. ArxivAgent, TavilyAgent etc.
    """
    tools = load_tools(
        [
            "arxiv",
            "wikipedia",
        ],
    )
    tools += [
        SemanticScholarQueryRun(),
        TavilySearchResults(),
        PubmedQueryRun(),
    ]
    return tools


def main():

    os.environ["OPENAI_MODEL_NAME"] = "gpt-4o-mini"

    search_tool = SerperDevTool()

    # Define your agents with roles and goals
    pro_researcher = Agent(
        role="Senior Research Analyst",
        goal="Find many (10+) arguments, papers, websites, and blog posts that support a given statement.",
        backstory="""You work at a leading research institution and are known for your thoroughness and ability
      to search deep into the web to find the most relevant information supporting statements.""",
        verbose=True,
        allow_delegation=False,
        # You can pass an optional llm attribute specifying what model you wanna use.
        # llm=ChatOpenAI(model_name="gpt-3.5", temperature=0.7),
        tools=[search_tool],
    )
    con_researcher = Agent(
        role="Senior Research Analyst",
        goal="Find many (10+) arguments, papers, websites, and blog posts that argue against a given statement.",
        backstory="""You work at a leading research institution and are known for your thoroughness and ability
        to search deep into the web to find the most relevant information that disagrees with particular statements.""",
        verbose=True,
        allow_delegation=False,
        # You can pass an optional llm attribute specifying what model you wanna use.
        # llm=ChatOpenAI(model_name="gpt-3.5", temperature=0.7),
        tools=[search_tool],
    )
    writer = Agent(
        role="Summariser",
        goal="Summarize the arguments found by the pro and con researchers into a coherent and concise report",
        backstory="""You are a skilled writer and summarizer, known for your ability to distill complex information
      into clear and concise summaries. You clearly present both sides of an argument.""",
        verbose=True,
        allow_delegation=True,
    )

    # Create tasks for your agents
    pro_arguments_task = Task(
        description="""Find 10+ resources from a wide range of sources that support the following statement: {statement}""",
        expected_output="Links to 10+ resources supporting the statement, including a title for each and a short summary of"
        "the position taken in each resource.",
        agent=pro_researcher,
    )

    con_arguments_task = Task(
        description="""Find 10+ resources from a wide range of sources that argue against the following statement: {statement}""",
        expected_output="Links to 10+ resources arguing against the statement, including a title for each and a short summary of"
        "the position taken in each resource.",
        agent=con_researcher,
    )

    summarise_arguments_task = Task(
        description="""Using the inputs provided by the pro and con researchers, summarize the arguments found into a coherent
      and concise report. Include a brief overview of the main points made by each side, highlighting the key arguments and
      counterarguments.""",
        expected_output="Full blog post of at least 4 paragraphs",
        agent=writer,
    )

    # Instantiate your crew with a sequential process
    crew = Crew(
        agents=[pro_researcher, con_researcher, writer],
        tasks=[pro_arguments_task, con_arguments_task, summarise_arguments_task],
        verbose=True,  # You can set it to 1 or 2 to different logging levels
        process=Process.sequential,
    )

    statement = """While the CDC team found that 1 in 1,000,000 patients was injured by vaccines, the Lazarus team found that 1 in 37 kids had potential claims for vaccine injuries"""
    result = crew.kickoff({"statement": statement})

    print("######################")
    print(result)


if __name__ == "__main__":
    main()
