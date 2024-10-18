from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time

import streamlit as st
from dotenv import load_dotenv
from loguru import logger
from tqdm.auto import tqdm

from fakt_ai.langchain_implementation import (
    semantic_scholar_search_chain,
    paper_analysis_chain,
    final_answer_chain,
)
from fakt_ai.utils import format_elapsed_time

load_dotenv()


def main():
    st.set_page_config(page_title="Fakt AI MVP", page_icon="🔍", layout="centered")

    st.title("🔍 Fakt.ai")
    st.markdown("##### _Transparent, automated fact-checking powered by AI Agents_")
    st.markdown(
        """
    This is a very minimal MVP that uses AI Agents to fact check queries. It only supports academic papers search
    at the moment. So, please ask questions requiring academic papers to answer.
    
    Open source [repository here](https://github.com/codeananda/fakt_ai/)
    """
    )

    query = st.text_input(
        "Enter your query:",
        placeholder="e.g., What treatments were available to treat COVID before the vaccines came out?",
    )

    if query:
        start = time()
        st.write("Generating response...")

        response_container = st.empty()
        time_container = st.empty()

        with response_container.container():
            with st.expander(f"Step 1/3: Searching for papers related to '{query}'"):
                st.write("")

            search_chain = semantic_scholar_search_chain()
            papers = search_chain.invoke({"query": query})

            # TODO - figure out why more than 20 papers are being returned
            if len(papers) > 20:
                papers = papers[:20]

            with st.expander(f"Step 2/3: Found {len(papers)} papers. Analyzing each one..."):
                st.write("")

            analysis_chain = paper_analysis_chain()

            paper_analyses = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(analysis_chain.invoke, {"query": query, "paper": paper})
                    for paper in papers
                ]
                with tqdm(total=len(papers), desc="Analyzing papers") as progress_bar:
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            paper_analyses.append(result)
                        except Exception as e:
                            logger.error(f"An error occurred during paper analysis: {e}")
                        finally:
                            progress_bar.update(1)

            with st.expander("Step 3/3: Generating final answer..."):
                st.write("")

            answer_chain = final_answer_chain()
            final_answer = answer_chain.invoke(
                {"query": query, "paper_analyses": paper_analyses}
            )

            st.markdown(f"# Final Answer\n\n{final_answer.content}")

        time_container.markdown(f"**Fakt checked in: {format_elapsed_time(start)}**")


if __name__ == "__main__":
    main()
