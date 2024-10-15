import ast
from concurrent.futures import ThreadPoolExecutor
from time import time

import streamlit as st
from dotenv import load_dotenv

from fakt_ai.crewai_implementation import (
    semantic_scholar_crew,
    paper_analysis_crew,
    final_answer_crew,
)
from fakt_ai.utils import format_elapsed_time

load_dotenv()


def main():
    st.set_page_config(page_title="Fakt AI MVP", page_icon="🔍", layout="centered")

    st.title("🔍 Fakt.ai")
    st.markdown("##### _Transparent, automated fact-checking powered by AI Agents_")
    st.markdown("-------")

    st.markdown(
        """
    This is a very minimal MVP that uses AI Agents to fact check queries. It only supports academic papers search
    at the moment. So, please ask questions that require academic papers to answer them. It can be tempermental. 
    If you don't get an answer, re-run the query. Usually it works the second time.
    
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
            crew = semantic_scholar_crew()
            output = crew.kickoff({"query": query})
            papers: list[dict] = ast.literal_eval(output.raw)

            with st.expander(f"Step 2/3: Found {len(papers)} papers. Analyzing each one..."):
                st.write("")
            analysis_crew = paper_analysis_crew()
            with ThreadPoolExecutor(max_workers=5) as executor:
                paper_analyses = list(
                    executor.map(
                        lambda paper: analysis_crew.kickoff(
                            {"query": query, "paper": paper}
                        ).raw,
                        papers,
                    )
                )

            with st.expander("Step 3/3: Generating final answer..."):
                st.write("")
            answer_crew = final_answer_crew()
            final_answer = answer_crew.kickoff(
                {"query": query, "paper_analyses": paper_analyses}
            )

            st.markdown(f"# Final Answer\n\n{final_answer.raw}")

        time_container.markdown(f"**Fakt checked in: {format_elapsed_time(start)}**")


if __name__ == "__main__":
    main()
