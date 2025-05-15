import os
from datetime import datetime
from langchain_core.messages import AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pinecone import Pinecone, ServerlessSpec
import time
from langchain_pinecone import PineconeVectorStore

import asyncio
# For dynamic content rendering
from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright
from langchain.schema import Document

# Load environment variables
load_dotenv()

# pinecone init
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "chatime"  # change if desired

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        deletion_protection="enabled",  # Defaults to "disabled"
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=OpenAIEmbeddings())

# Playwright helper to fetch rendered HTML
async def fetch_dynamic_html(url: str, timeout: int = 30000) -> str:
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, wait_until="networkidle", timeout=timeout)
        html = await page.content()
        await browser.close()
    return html

# Recursively crawl URLs up to max_depth using dfs
async def crawl_urls(start_url: str, dynamic: bool, max_depth: int):
    visited = set()
    pages = []  # list of (url, html)

    async def crawl(url: str, depth: int):
        if depth < 0 or url in visited:
            return
        visited.add(url)
        try:
            html = await fetch_dynamic_html(url)

            # html = await fetch_dynbamic_html(url) if dynamic else fetch_static_html(url)
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
                    await crawl(link, depth - 1)
        except Exception as e:
            # st.sidebar.warning(f"Failed to fetch {url}: {e}")
            print(f"Failed to fetch {url}: {e}")

    await crawl(start_url, max_depth)
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
async def get_vectorstore_from_url(url: str, dynamic: bool = True, max_depth: int = 1):
    pages = await crawl_urls(url, dynamic=dynamic, max_depth=max_depth)
    # stores metadata
    docs = save_and_extract_text(pages)
    splitter = RecursiveCharacterTextSplitter()
    chunks = splitter.split_documents(docs)
    # return Chroma.from_documents(chunks, OpenAIEmbeddings())
    vector_store.add_documents(chunks)
    return vector_store

# Build retrieval components
def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever(
        # default 10 is too low to search up critical information
        search_kwargs={"k": 15, "namespace": ""}
    )
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
        ("system", "Answer the user's questions based on the below context:\n\n{context}\n\n"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    # python chain, map input to map, then feed data to prompt
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
    

store = asyncio.run(get_vectorstore_from_url("https://qstommyshu.github.io/", dynamic=True, max_depth=3))
get_response("what is tommy's experience?", store)
print("\n\n")
get_response("How many blogs are there?", store)
print("\n\n")
get_response("Give me a piece of code snippets used in tommy's blog: grpc-introduction-chapter-2?", store)


# ➜ python pinecone_store.py
# Fetched url: https://qstommyshu.github.io/
# Fetched url: https://qstommyshu.github.io/resume
# Fetched url: https://qstommyshu.github.io/blog
# Fetched url: https://qstommyshu.github.io/opensource
# Fetched url: https://qstommyshu.github.io/life
# Fetched url: https://qstommyshu.github.io/contact
# Fetched url: https://qstommyshu.github.io/blog/grpc-introduction-chapter-1
# Fetched url: https://qstommyshu.github.io/blog/grpc-introduction-chapter-2
# Fetched url: https://qstommyshu.github.io/blog/grpc-introduction-chapter-3
# crawling done!
# answer is:  Tommy Shu is a Full Stack Software Developer with experience in building modern web applications and distributed systems. Here are some positions he held:

# 1. **Full Stack Software Developer Intern at TD Securities (May 2023 - Dec 2023):**
#    - Led front-end development efforts and increased test coverage from 30% to 95% using Test Driven Development.
#    - Developed over 20 trader-facing pages using TypeScript, React, and AgGrid.
#    - Improved service reliability by 30% through the implementation of a backend data cache.

# 2. **Full Stack Software Developer Intern at TD Securities (May 2022 - Aug 2022):**
#    - Developed a graph database migration proof of concept with a 40% improvement in query performance.
#    - Built over 20 REST APIs and created a tool for generating trading reports using Elixir and GraphQL.

# 3. **Software Developer in Test Intern at Caseware International (May 2021 - Apr 2022):**
#    - Developed a validation system for AWS S3 parquet files, a Fintech BI Microservice, and over 20 end-to-end tests using Cypress.io.
#    - Designed more than 10 CI/CD pipelines using GitHub Actions.

# Sources: https://qstommyshu.github.io/, https://qstommyshu.github.io/resume



# answer is:  There are three blog posts on the topic of gRPC, each labeled as a chapter in a series. These blog posts are:

# 1. gRPC Introduction (Chapter 1) - December 14, 2024
# 2. gRPC Introduction (Chapter 2) - December 18, 2024
# 3. gRPC Introduction (Chapter 3) - January 4, 2025

# Sources: https://qstommyshu.github.io/blog



# answer is:  Certainly! Here's a code snippet used in Tommy's blog from the chapter "gRPC Introduction (Chapter 2)":

# ```typescript
# echo 'syntax = "proto3";

# service rpcExample {
#     rpc UnaryExample (MyRequest) returns (MyResponse);
#     rpc ServerStreamExample (MyRequest) returns (stream MyResponse);
#     rpc ClientStreamExample (stream MyRequest) returns (MyResponse);
#     rpc BidirectionalExample (stream MyRequest) returns (stream MyResponse);
# }

# message MyRequest {
#     int32 id = 1;
#     string msg = 2;
# }

# message MyResponse {
#     string msg = 1;
# }' > rpc_example.proto
# ```

# This snippet defines a `.proto` file defining the `rpcExample` service with several RPC methods and message types.

# Sources: https://qstommyshu.github.io/blog/grpc-introduction-chapter-2