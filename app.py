import os
from datetime import datetime
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from bs4 import BeautifulSoup

# For dynamic content rendering
from playwright.sync_api import sync_playwright
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Playwright helper to fetch rendered HTML
def fetch_dynamic_html(url: str, timeout: int = 30000) -> str:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=timeout)
        page.wait_for_load_state('networkidle')
        html = page.content()
        browser.close()
    return html

# Save (static or dynamic) HTML to disk
def save_html_content(url: str, dynamic: bool = True) -> tuple[str, str]:
    if dynamic:
        html_content = fetch_dynamic_html(url)
    else:
        import requests
        html_content = requests.get(url).text

    os.makedirs('scraped', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    domain = url.replace('https://', '').replace('http://', '').split('/')[0]
    filename = f"scraped/{domain}_{timestamp}.html"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    return filename, html_content

# Build a vector store from a URL
def get_vectorstore_from_url(url: str, dynamic: bool = True):
    filename, html = save_html_content(url, dynamic=dynamic)
    st.sidebar.success(f"HTML saved to {filename}")

    # Parse text out of rendered HTML
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text(separator='\n')

    # Wrap in a single Document and split
    doc = Document(page_content=text, metadata={'source': url})
    splitter = RecursiveCharacterTextSplitter()
    docs = splitter.split_documents([doc])

    # Create vector store
    return Chroma.from_documents(docs, OpenAIEmbeddings())

# Build retrieval components as before
def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    return create_history_aware_retriever(llm, retriever, prompt)


def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_chain)

# Orchestrate a single response
def get_response(user_input: str) -> str:
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    rag_chain = get_conversational_rag_chain(retriever_chain)
    out = rag_chain.invoke({
        'chat_history': st.session_state.chat_history,
        'input': user_input
    })
    return out['answer']

# Streamlit app layout
st.set_page_config(page_title="Chat with Websites", page_icon="🤖")
st.title("Chat with Websites (Dynamic+Static)")

with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")
    dynamic = st.checkbox("Render dynamic content?", value=True)

if not website_url:
    st.info("Enter a website URL above to begin scraping...")
else:
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Hello! I'm your assistant.")]
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url, dynamic)

    # User input & response
    user_query = st.chat_input("Your message...")
    if user_query:
        ans = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=ans))

    # Display chat
    for msg in st.session_state.chat_history:
        role = "AI" if isinstance(msg, AIMessage) else "Human"
        with st.chat_message(role):
            st.write(msg.content)
