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
from urllib.parse import urljoin, urlparse

# For dynamic content rendering
from playwright.sync_api import sync_playwright
from langchain.schema import Document

# Load environment variables
load_dotenv()

# TODO: crawling can be multi-threaded or async
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

# Fetch static HTML via requests
def fetch_static_html(url: str) -> str:
    import requests
    response = requests.get(url)
    response.raise_for_status()
    return response.text

# Recursively crawl URLs up to max_depth using dfs
def crawl_urls(start_url: str, dynamic: bool, max_depth: int):
    visited = set()
    pages = []  # list of (url, html)

    def crawl(url: str, depth: int):
        if depth < 0 or url in visited:
            return
        visited.add(url)
        try:
            html = fetch_dynamic_html(url) if dynamic else fetch_static_html(url)
            pages.append((url, html))
            # tell users which pages are fetched
            st.sidebar.success(f"Fetched url: {url}")
            if depth == 0:
                return
            # parse links
            soup = BeautifulSoup(html, 'html.parser')
            base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
            for tag in soup.find_all('a', href=True):
                link = urljoin(base, tag['href'])
                # only follow same domain links
                if urlparse(link).netloc == urlparse(start_url).netloc:
                    crawl(link, depth - 1)
        except Exception as e:
            st.sidebar.warning(f"Failed to fetch {url}: {e}")

    crawl(start_url, max_depth)
    return pages

# Save HTML pages to disk and return combined text for reference
def save_and_extract_text(pages):
    os.makedirs('scraped', exist_ok=True)
    all_docs = []
    for url, html in pages:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        domain = urlparse(url).netloc.replace(':', '_')
        fname = f"scraped/{domain}_{timestamp}.html"
        with open(fname, 'w', encoding='utf-8') as f:
            f.write(html)
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text(separator='\n')
        metadata = {'source': url}
        all_docs.append(Document(page_content=text, metadata=metadata))
    return all_docs

# Build a vector store from a URL with recursion support
def get_vectorstore_from_url(url: str, dynamic: bool = True, max_depth: int = 1):
    pages = crawl_urls(url, dynamic=dynamic, max_depth=max_depth)
    st.sidebar.success(f"Fetched {len(pages)} pages (depth {max_depth})")
    docs = save_and_extract_text(pages)
    splitter = RecursiveCharacterTextSplitter()
    chunks = splitter.split_documents(docs)
    return Chroma.from_documents(chunks, OpenAIEmbeddings())

# Build retrieval components
def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    return create_history_aware_retriever(llm, retriever, prompt)

# TODO: provide external links as conversation source
def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}, and always reply back the source url of the answer"),
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
st.set_page_config(page_title="Chatime MVP", page_icon="🤖")
st.title("Chatime MVP (Recursive Scraping)")

with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")
    dynamic = st.checkbox("Render dynamic content?", value=True)
    max_depth = st.number_input("Max recursion depth", min_value=0, max_value=3, value=1, step=1)

if not website_url:
    st.info("Enter a website URL above to begin scraping...")
else:
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Hello! I'm your assistant.")]
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url, dynamic, max_depth)

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
