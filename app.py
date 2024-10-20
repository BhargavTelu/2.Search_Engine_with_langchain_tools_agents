import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

## Arxiv, Wikipedia, Duckduck tools initiation
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper = arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper= wiki_wrapper)

search = DuckDuckGoSearchRun(name = "Search")

st.title("Langchain - chat with search")
"""In this example, we are using streamlitcallbackhandler to display the thoughts 
and actions of an agent in an interactive streamlit app"""
##sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Api Key:", type = "password")

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role": "assistant",
         "content": "Hi, I'm a chatbot who can search the web"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query=st.chat_input(placeholder="Ask any question")
if user_query:
    st.session_state.messages.append({"role":"user", "content":user_query})
    st.chat_message("user").write(user_query)

    llm = ChatGroq(groq_api_key = api_key , model_name = "Llama3-8b-8192", streaming=True)
    tools = [search,arxiv,wiki]
    search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                     handling_parsing_errors=True)

    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response) 

