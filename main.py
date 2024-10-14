import ast
import warnings
from concurrent.futures import ThreadPoolExecutor
from time import time

from IPython.display import Markdown
from dotenv import load_dotenv, find_dotenv
from fire import Fire
from loguru import logger
from rich.console import Console
from rich.markdown import Markdown

from fakt_ai.crewai_implementation import (
    semantic_scholar_crew,
    paper_analysis_crew,
    final_answer_crew,
)
from fakt_ai.utils import format_elapsed_time

load_dotenv(find_dotenv(raise_error_if_not_found=True, usecwd=True))

warnings.filterwarnings(
    "ignore", message="Overriding of current TracerProvider is not allowed*"
)


def main(query: str):
    start = time()
    logger.info(f"Searching for papers related to '{query}'")
    crew = semantic_scholar_crew()
    output = crew.kickoff({"query": query})
    papers: list[dict] = ast.literal_eval(output.raw)

    logger.info(f"Found {len(papers)} papers. Now analyzing each one...")
    analysis_crew = paper_analysis_crew()
    with ThreadPoolExecutor(max_workers=5) as executor:
        paper_analyses = list(
            executor.map(
                lambda paper: analysis_crew.kickoff({"query": query, "paper": paper}).raw,
                papers,
            )
        )

    logger.info("Generating final answer...")
    answer_crew = final_answer_crew()
    final_answer = answer_crew.kickoff({"query": query, "paper_analyses": paper_analyses})

    logger.success(f"Final answer generated in {format_elapsed_time(start)}")

    console = Console(width=150, color_system="auto")
    md = Markdown(final_answer.raw)
    console.print(md, justify="left")


if __name__ == "__main__":
    Fire(main)
