import os
from datetime import datetime
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
            # st.sidebar.success(f"Fetched url: {url}")
            print(f"Fetched url: {url}")
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
            # st.sidebar.warning(f"Failed to fetch {url}: {e}")
            print(f"Failed to fetch {url}: {e}")

    crawl(start_url, max_depth)
    print("crawling done!")
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
    # st.sidebar.success(f"Fetched {len(pages)} pages (depth {max_depth})")
    # stores metadata
    docs = save_and_extract_text(pages)
    # print("docs is")
    # print(docs[0])
    splitter = RecursiveCharacterTextSplitter()
    chunks = splitter.split_documents(docs)
    # print("Chunk is")
    # print(chunks[0])
    # metadata here
    # print(chunks[0].metadata)
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

from pydantic import BaseModel, Field
from typing import List

class StructuredAnswer(BaseModel):
    answer: str = Field(description="The answer to the user's question.")
    sources: List[str] = Field(description="A list of source URLs used to answer the question.")

from langchain_core.runnables import RunnableMap

# TODO: provide external links as conversation source
def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI(model="gpt-4o-2024-08-06").with_structured_output(StructuredAnswer)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}\n\nAlways include the source URL at the end of your response in the format: Source: [URL]"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    rag_chain = (
        RunnableMap({
            "context": retriever_chain,
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"]
        })
        | prompt
        | llm
    )

    return rag_chain

# TODO: update it with structured output
# Orchestrate a single response
def get_response(user_input: str, store) -> str:
    retriever_chain = get_context_retriever_chain(store)
    rag_chain = get_conversational_rag_chain(retriever_chain)
    response = rag_chain.invoke({
        'chat_history': [AIMessage(content="Hello! I'm your assistant.")],
        'input': user_input
    })

    answer = response.answer
    if response.sources:
        answer += "\n\nSources: " + ", ".join(response.sources)
    print("answer is: ", answer)
    return answer
    

store = get_vectorstore_from_url("https://qstommyshu.github.io/", dynamic=True, max_depth=3)
get_response("what is tommy's experience?", store)
print("\n\n")
get_response("How many blogs are there?", store)
print("\n\n")
get_response("Give me a piece of code snippets used in tommy's blog: grpc-introduction-chapter-2?", store)