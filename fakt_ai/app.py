import os

import gradio as gr
from dotenv import load_dotenv
from gradio import ChatMessage
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_openai import ChatOpenAI
from fakt_ai.crewai_implementation import get_all_tools

load_dotenv()

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

model = ChatOpenAI(temperature=0, streaming=True)

tools = get_all_tools()

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-tools-agent")
# print(prompt.messages) -- to see the prompt
agent = create_openai_tools_agent(model.with_config({"tags": ["agent_llm"]}), tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools).with_config({"run_name": "Agent"})


async def interact_with_langchain_agent(prompt, messages):
    messages.append(ChatMessage(role="user", content=prompt))
    yield messages
    async for chunk in agent_executor.astream({"input": prompt}):
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
