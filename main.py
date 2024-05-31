# Import the required libraries
import os
import streamlit as st
from phi.assistant import Assistant  # type: ignore
from phi.tools.arxiv_toolkit import ArxivToolkit
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.hackernews import HackerNews
from phi.llm.openai import OpenAIChat  # type: ignore
from dotenv import load_dotenv

# Setup Streamlit App
# Page setting
st.set_page_config(
    layout="wide", page_title="Multi-Agent AI Research Assistant", page_icon="🤖"
)

# Load the CSS file
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Set sidebar
st.sidebar.title("Multi-Agent AI Research Assistant 🔎🤖")
st.sidebar.image("data/Libran4.png", width=300)
st.sidebar.title("Simple Demo of Multi-Agent AI LLM Application")
st.sidebar.subheader("Instructions:")
st.sidebar.markdown(
    """
    - Select an agent from the dropdown menu.
    - Follow the instructions provided by the agent on the main panel.
    - Enjoy the conversation with the AI models.
        """
)

# Set the agents
selection = st.sidebar.selectbox(
    "Select the type of data you want to retrieve",
    ["Arxiv Assistant", "Hackers News Assistant", "DuckDuckGo Assistant"],
)


# Get the OpenAI API key from the environment variables
load_dotenv()
api_key = os.environ["OPENAI_API_KEY"]
# Create an instance of the Assistant class
if api_key:
    if selection == "Arxiv Assistant":
        st.title("Arxiv Research Assistant 🔎🤖")
        st.write(
            "Welcome to the Arxiv Research Assistant! This app is designed to help you with your research by providing you with the latest articles and papers on AI and Machine Learning from Arxiv. Let's get started!"
        )
        arxiv_researcher = Assistant(
            name="Arxiv Researcher",
            run_id="Arxiv_researcher",
            role="Researches Arxiv Journal.",
            tools=[ArxivToolkit()],
            show_tool_calls=True,
        )
        hn_assistant = Assistant(
            name="Research Assistant",
            run_id="hn_assistant",
            team=[arxiv_researcher],
            llm=OpenAIChat(
                model="gpt-4o", max_tokens=1024, temperature=0.5, api_key=api_key
            ),
        )

    elif selection == "Hackers News Assistant":
        st.title("Hackers News Assistant 🔎👨‍💻")
        st.write(
            "Welcome to the Hackers News Assistant! This app is designed to help you with your research by providing you with the latest articles on Hacker News. Let's get started!"
        )
        story_researcher = Assistant(
            name="HarkerRank Researcher",
            run_id="hacker_researcher",
            role="Researches HackerNews Stories and Users.",
            tools=[HackerNews()],
            show_tool_calls=True,
        )
        user_researcher = Assistant(
            name="HarkerRank Researcher",
            run_id="hacker_researcher",
            role="Read Articles and Comments on HackerNews.",
            tools=[HackerNews()],
            show_tool_calls=True,
        )

        hn_assistant = Assistant(
            name="Research Assistant",
            run_id="hn_assistant",
            team=[story_researcher, user_researcher],
            llm=OpenAIChat(
                model="gpt-4o", max_tokens=1024, temperature=0.5, api_key=api_key
            ),
        )

    elif selection == "DuckDuckGo Assistant":
        st.title("DuckDuckGo Assistant 🔎🦆")
        st.write(
            "Welcome to the DuckDuckGo Assistant! This app is designed to help you with your research by providing you with the latest articles and information from DuckDuckGo. Let's get started!"
        )
        duck_researcher = Assistant(
            name="DuckDuckGo Researcher",
            run_id="duckduckgo_researcher",
            role="Researches DuckDuckGo.",
            tools=[DuckDuckGo()],
            show_tool_calls=True,
        )

        hn_assistant = Assistant(
            name="Research Assistant",
            run_id="hn_assistant",
            team=[duck_researcher],
            llm=OpenAIChat(
                model="gpt-4o", max_tokens=1024, temperature=0.5, api_key=api_key
            ),
        )

    # Input field for user to ask questions
    query = st.text_input("Enter your query:")
    run_button = st.button("Run Query")
    if run_button:
        # Get the response from the assistant
        response = hn_assistant.run(query, stream=False)
        st.write(response)
