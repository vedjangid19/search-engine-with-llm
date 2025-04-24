import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun, DuckDuckGoSearchRun

from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

###
load_dotenv()

# groq_api_key = os.getenv('GROQ_API_KEY')

arxiv_api_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
arxiv = ArxivQueryRun(api_wrapper=arxiv_api_wrapper)

wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wiki = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

search = DuckDuckGoSearchRun(name="search")


st.title("🔎 LangChain - Chat with search")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

# sidebar for setting
st.sidebar.title("Setting")
groq_api_key = st.sidebar.text_input("Enter Your Groq API Key", type="password")

if "messages" not in  st.session_state:
    st.session_state['messages'] = [
        {
            "role": "assisstant",
            "content": "Hi, I'm a chatbot who can search the web. how can i help you?"
        }
    ]


for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if prompt:=st.chat_input(placeholder="what is machine learning?"):
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )
    st.chat_message("user").write(prompt)

    llm = ChatGroq(model="Llama3-8b-8192", groq_api_key=groq_api_key, streaming=True)

    tools = [search, arxiv, wiki]

    search_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handling_parsing_errors=True
    )

    with st.chat_message("assistance"):
        st_cb = StreamlitCallbackHandler(
            st.container(),
            expand_new_thoughts=False
        )

        response = search_agent.run(
            st.session_state.messages,
            callbacks=[st_cb]
        )

        st.session_state.messages.append(
            {
                "role": "assistance",
                "content": response
            }
        )

        st.write(response)















