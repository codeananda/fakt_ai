import os

import gradio as gr
from dotenv import load_dotenv
from gradio import ChatMessage
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

from fakt_ai.crewai_implementation import get_all_tools, _get_llm

load_dotenv()

# TODO - add elegant section for all API keys
if not (os.getenv("SERPAPI_API_KEY") and os.getenv("OPENAI_API_KEY")):
    with gr.Blocks() as demo:
        gr.Markdown(
            """
        # Chat with a LangChain Agent 🦜⛓️ and see its thoughts 💭
        In order to run this space, duplicate it and add the following space secrets:
        * SERPAPI_API_KEY - create an account at serpapi.com and get an API key
        * OPENAI_API_KEY - create an openai account and get an API key
        """
        )
    demo.launch()

model = _get_llm("groq")

tools = get_all_tools()

prompt = PromptTemplate.from_template(
    """
    Answer the following questions as best you can. You have access to the following tools:
    {tools} 
    
    with the following names:
    {tool_names}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

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

    Thought:{agent_scratchpad}
    """
)

agent = create_react_agent(model, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,
)


async def interact_with_langchain_agent(prompt, messages):
    messages.append(ChatMessage(role="user", content=prompt))
    yield messages
    async for chunk in agent_executor.astream({"query": prompt}):
        if "steps" in chunk:
            for step in chunk["steps"]:
                messages.append(
                    ChatMessage(
                        role="assistant",
                        content=step.action.log,
                        metadata={"title": f"🛠️ Used tool {step.action.tool}"},
                    )
                )
                yield messages
        if "output" in chunk:
            messages.append(ChatMessage(role="assistant", content=chunk["output"]))
            yield messages


with gr.Blocks() as demo:
    gr.Markdown("# Chat with a LangChain Agent 🦜⛓️ and see its thoughts 💭")
    chatbot_2 = gr.Chatbot(
        type="messages",
        label="Agent",
        avatar_images=(
            None,
            "https://em-content.zobj.net/source/twitter/141/parrot_1f99c.png",
        ),
    )
    input_2 = gr.Textbox(lines=1, label="Chat Message")
    input_2.submit(interact_with_langchain_agent, [input_2, chatbot_2], [chatbot_2])

if __name__ == "__main__":
    demo.launch()
