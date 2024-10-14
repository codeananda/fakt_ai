import os
import warnings
from typing import Literal

from semanticscholar import SemanticScholar

warnings.filterwarnings(
    "ignore", message="Valid config keys have changed in V2:*", category=UserWarning
)

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, BaseTool
from dotenv import find_dotenv, load_dotenv
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatAnthropic
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from pydantic import validate_call

load_dotenv(find_dotenv(raise_error_if_not_found=True))

warnings.filterwarnings(
    "ignore", message="Overriding of current TracerProvider is not allowed*"
)


@validate_call
def semantic_scholar_crew(
    model_name: Literal["openai", "anthropic", "groq"] = "groq",
    verbose: bool = False,
    **kwargs,
):
    """Build a crew to search Semantic Scholar

    Parameters
    ----------
    model_name : str, optional
        The model to use as the backbone of the Crew
    verbose : bool, optional
        Whether to print log messages to the screen or not
    **kwargs
        Other arguments passed to Crew constructor
    """

    llm = _get_llm(model_name)
    agent_params = {
        "memory": True,
        "verbose": verbose,
        "llm": llm,
    }

    semantic_scholar_agent = Agent(
        role="Research Analyst",
        goal="Find papers that supporting the given query: {query}",
        backstory="A research analyst skilled at doing in-depth research into complex "
        "topics and discovering the truth",
        max_rpm=1,  # S2 has a 1 request/second rate limit
        max_iter=1,  # we just need to search once and return results
        max_execution_time=60,
        **agent_params,
    )
    semantic_scholar_task = Task(
        description="Search on semantic scholar for papers that support this query: {query}. "
        "\nNote that there may not be papers supporting a given query. In which case, "
        "it is ok and informative to return no papers. ",
        expected_output="A list of papers supporting the query + reasons for each. Please include "
        "1. Paper title, 2. Authors, 3. Publication date, 4. Abstract, 5. Reason for "
        "inclusion.",
        agent=semantic_scholar_agent,
        tools=[SemanticScholarTool(result_as_answer=True)],
    )

    crew = Crew(
        agents=[semantic_scholar_agent],
        tasks=[semantic_scholar_task],
        verbose=False,
        process=Process.sequential,
        planning=False,
        **kwargs,
    )
    return crew


@validate_call
def paper_analysis_crew(
    model_name: Literal["openai", "anthropic", "groq"] = "groq",
    verbose: bool = False,
    **kwargs,
):
    llm = _get_llm(model_name)
    agent_params = {
        "memory": True,
        "verbose": verbose,
        "llm": llm,
    }

    paper_analysis_agent = Agent(
        role="Paper Analyst",
        goal="Critically analysing scientific papers and how they relate to given user queries.",
        backstory="Loves reading and analysis",
        **agent_params,
    )

    task_description = """We are trying to determine the truth or falisty of the following query
    
    {query}
    
    Here is a paper related to the query. Please write a summary of the contents and how the paper
    either supports or refutes the query.
    
    {paper}
    """

    paper_analysis_task = Task(
        description=task_description,
        expected_output="A summary of the contents of the paper and whether it supports or refutes the query.",
        agent=paper_analysis_agent,
    )

    crew = Crew(
        agents=[paper_analysis_agent],
        tasks=[paper_analysis_task],
        verbose=False,
        process=Process.sequential,
        planning=False,
        **kwargs,
    )
    return crew


@validate_call
def final_answer_crew(
    model_name: Literal["openai", "anthropic", "groq"] = "groq",
    verbose: bool = False,
    **kwargs,
):
    llm = _get_llm(model_name)
    agent_params = {
        "memory": True,
        "verbose": verbose,
        "llm": llm,
    }

    summary_agent = Agent(
        role="Final Answer Agent",
        goal="Use the paper analyses to determine the truth or falsity of a given query.",
        backstory="Experienced in writing detailed reports weighing up different sides of an argument "
        "all with supporting evidence.",
        **agent_params,
    )

    task_description = """We are trying to determine the truth or falisty of the following query
    
    {query}
    
    We have already collected and analysed many scientific papers related to the query.
    Please summarise the scientific analyses and write a report determining the truth or falsity
    of the query.
    
    Paper analyses:
    {paper_analyses}
    """

    summary_task = Task(
        description=task_description,
        expected_output="A detailed report either confirming or falisfying the claim. All statements "
        "should be backed up with facts from the paper analyses and include links. "
        "Use markdown formatting",
        agent=summary_agent,
    )

    crew = Crew(
        agents=[summary_agent],
        tasks=[summary_task],
        verbose=False,
        process=Process.sequential,
        planning=False,
        **kwargs,
    )
    return crew


class SemanticScholarTool(BaseTool):
    name: str = "Search Semantic Scholar"
    description: str = (
        "A tool that can be used to call the Semantic Scholar API and obtain information about academic papers."
    )

    def _run(self, s2_query: str):
        try:
            ss = SemanticScholar(retry=True, api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY"))
            fields = [
                "paperId",
                "tldr",
                "abstract",
                "url",
                "title",
                "publicationDate",
                "journal",
                "referenceCount",
                "citationCount",
                "influentialCitationCount",
                "authors",
            ]
            results = ss.search_paper(s2_query, fields=fields, limit=20)
            return f"{results.items}"
        except Exception as e:
            return f"Failed to search Semantic Scholar for query '{s2_query}': {e}"


def fakt_ai_crew(**kwargs):
    """Create a crew to run the entire Fakt AI process on its own.

    Parameters
    ----------
    **kwargs
        Other arguments passed to Crew constructor
    """
    llm = _get_llm("groq")
    agent_params = {
        "memory": True,
        "verbose": True,
        "llm": llm,
    }

    fakt_ai_desc = """Our goal is to determine the truth or falisty of the following query
        submitted by a user:

        Query: {query}

        You are Fakt AI, an agent designed to scour the internet and determine whether a given
        user query is True or False. You exist to cut through the noise and combat the mass of
        information online that no single person can possibly handle themselves. It is your goal
        to gather relavant info and synthesis it into comprehensive answers.

        Some use cases include:
        - Alice listens to a podcast and hears Bob say: X is true and you can read scientific
        paper Y for proof. 

            - Fakt AI, will search online for that paper, read the paper, find the relevant portions
            and present back an argument as to whether that is true or not.

        It is perfectly fine to return back that a statement is False. 

        We rely on cold, hard facts and do not shy away from the tense topics of the day e.g.
        migration, vaccines, trans activism etc. 

        Our goal is to present the Truth as the current scientific literature understands it.
        We fully understand that this is always a 'best guess'.

        We do not purport to be able to answer all questions. Our goal is simply to gather
        the relevant data and present it in a way that is easy for a lay person to understand.

        In this, we fulfil a crucial role that most people today lack: the aility to perform
        deep research across the web, read technical papers from disparate disciplines, and 
        present conclusions. 

        Here is the query we will be working with:
        {query}
        """

    fakt_ai_agent = Agent(
        role="Fakt AI",
        goal="""You are Fakt AI, an agent designed to scour the internet and determine whether a given
        user query is True or False. You exist to cut through the noise and combat the mass of
        information online that no single person can possibly handle themselves. It is your goal
        to gather relavant info and synthesis it into comprehensive answers.""",
        backstory="",
        **agent_params,
    )

    fakt_ai_tools = get_all_tools()

    fakt_ai_task = Task(
        description=fakt_ai_desc,
        expected_output="A markdown document detailing whether a given query is True or False "
        "with accompanying references.",
        agent=fakt_ai_agent,
        tools=fakt_ai_tools,
    )

    crew = Crew(
        tasks=[fakt_ai_task],
        agents=[fakt_ai_agent],
        verbose=True,
        process=Process.sequential,
        planning=False,
        **kwargs,
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


def run_pro_and_con_crew(**kwargs):
    """Create and run a crew that finds pro and con arguments for given statement using the
    SerperDevTool and returns a summary of the results.

    Parameters
    ----------
    **kwargs
        Other arguments passed to Crew constructor
    """

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
        **kwargs,
    )

    statement = """While the CDC team found that 1 in 1,000,000 patients was injured by vaccines, the Lazarus team found that 1 in 37 kids had potential claims for vaccine injuries"""
    result = crew.kickoff({"statement": statement})

    print("######################")
    print(result)


# TODO - fill this in properly
def _query_crew():
    """WIP: crew to brainstorm different search queries given a user input."""
    query_creator_agent = Agent(
        role="",
        goal="",
        backstory="",
    )

    query_creator_task_desc = """
    You are Fakt AI, an agent designed to scour the internet and determine whether a given
    user query is True or False.

    However, users often input queries that can be difficult to interpret or could be
    better formulated. 

    So, given a user query and the following tool, brainstorm some search queries 
    that can express different aspects of the query. 

    Here is the query:
    {query}
    """

    query_creator_agent = Agent(
        role="Query Brainstorm Creator",
        goal="Reformulate the given query into a list of search queries.",
        backstory="Excellent at creating new ideas",
    )

    query_creator_task = Task(
        description=query_creator_task_desc,
        expected_output="A list of search queries to use with the given tool.",
        agent=query_creator_agent,
    )


@validate_call
def _get_llm(name: Literal["openai", "anthropic", "groq"]):
    llm_kwargs = {
        "temperature": 0.5,  # TODO: play with temperature - what do we want?
        "timeout": None,
        "max_retries": 3,
    }
    match name:
        case "openai":
            return ChatOpenAI(model="gpt-4o", **llm_kwargs)
        case "anthropic":
            return ChatAnthropic(model="claude-3-5-sonnet-20240620", **llm_kwargs)
        case "groq":
            return ChatGroq(model="llama-3.1-70b-versatile", **llm_kwargs)


if __name__ == "__main__":
    import warnings
    from time import time
    from pathlib import Path
    from datetime import datetime
    from loguru import logger
    from fakt_ai.utils import format_elapsed_time

    warnings.filterwarnings("ignore")

    global_start = time()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    OUTPUT_DIR = Path(f"../data/output/run_{stamp}")

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    log_file = OUTPUT_DIR / "log.log"
    logger.add(log_file)

    logger.info(f"Saving all outputs to {OUTPUT_DIR}")

    crew = fakt_ai_crew(output_log_file=str(log_file))
    r = crew.kickoff(
        {
            "query": "What treatments were available to treat covid before the vaccines came out?"
        }
    )

    markdown_output = r.raw
    with open(OUTPUT_DIR / "output.md", "w") as f:
        f.write(markdown_output)

    logger.info(f"Written output to {OUTPUT_DIR / 'output.md'}")

    logger.success(
        f"🥳 Finished processing everything in {format_elapsed_time(global_start)}! 🥳 "
        f"Outputs stored in {OUTPUT_DIR}."
    )
