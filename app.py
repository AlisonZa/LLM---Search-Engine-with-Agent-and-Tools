import streamlit as st
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'End To End Search Engine GEN AI App using Tools And Agent With Open Source LLM'
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

## Tolls
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results= 1, doc_content_chars_max= 200)
wiki = WikipediaQueryRun(api_wrapper = api_wrapper_wiki)

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results= 1, doc_content_chars_max= 200)
arxiv = ArxivQueryRun(api_wrapper = api_wrapper_arxiv)

web_search = DuckDuckGoSearchRun(name = 'Search')

## Streamlit Config:
st.title('🔎 Search Engine GenAI App using Tools and Agents')
'''
In This Project We Are Going to Use 'StreamlitCallbackHandler' to Display the Actions that Our Model is Taking. 
'''

if 'messages ' not in st.session_state:
    st.session_state['messages'] = [
        {'role':'assistant','content':'Hi, I am a Chatbot that Can Search the Web. How Can I Help You?'}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if prompt:= st.chat_input(placeholder='What is Machine Learning?'):
    st.session_state.messages.append({'role':'user', 'content': prompt})
    st.chat_message('user').write(prompt)

    llm = ChatGroq(model_name = 'Llama3-8b-8192', streaming= True)
    tools = [web_search, wiki, arxiv]

    # Converting the tools in agents:
    search_agent = initialize_agent(tools, llm, agent= AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors = True)
    
    with st.chat_message('assistant'):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts = False)
        response = search_agent.run(st.session_state.messages, callbacks = [st_cb])
        st.session_state.messages.append({'role':'assistant','content':response})
        st.write(response)
