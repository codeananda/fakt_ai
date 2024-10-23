import os

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.tools import tool
from semanticscholar import SemanticScholar

from fakt_ai.utils import _get_llm


def semantic_scholar_search_chain():
    prompt = PromptTemplate.from_template(
        "Search on semantic scholar for papers that support this query: {query}. "
        "\nNote that there may not be papers supporting a given query. In which case, "
        "it is ok and informative to return no papers."
    )

    llm = _get_llm("groq")
    tools = [semantic_scholar_search]
    llm = llm.bind_tools(tools, tool_choice=True)

    chain = prompt | llm | (lambda x: x.tool_calls[0]["args"]) | semantic_scholar_search
    return chain


def paper_analysis_chain():
    prompt = PromptTemplate.from_template(
        """We are trying to determine the truth or falisty of the following query
    
        {query}
        
        Here is a paper related to the query. Please write a summary of the contents and how the paper
        either supports or refutes the query.
    
        {paper}
        """
    )
    llm = _get_llm("groq")
    chain = prompt | llm
    return chain


def final_answer_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                """"We are trying to determine the truth or falisty of the following query
    
            {query}
    
            We have already collected and analysed many scientific papers related to the query.
            Please summarise the scientific analyses and write a report determining the truth or falsity
            of the query.
    
            Paper analyses:
            {paper_analyses}
        """,
            )
        ]
    )
    llm = _get_llm("openai")
    chain = prompt | llm
    return chain


@tool
def semantic_scholar_search(query: str):
    """A tool that can be used to call the Semantic Scholar API and obtain information about academic papers."""
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
        results = ss.search_paper(query, fields=fields, limit=20)
        return results.items
    except Exception as e:
        return f"Failed to search Semantic Scholar for query '{query}': {e}"
