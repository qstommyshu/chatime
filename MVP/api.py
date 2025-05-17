import os
from langchain_core.messages import AIMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import time
from langchain_pinecone import PineconeVectorStore

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


# Build retrieval components
def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever(
        # default 10 is too low to search up critical information
        search_kwargs={"k": 10, "namespace": ""}
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


def map_to_history(conversation):
    # [{ "role": "human", "content": "你好" },
#   { "role": "ai",    "content": "你好，我是 AI" },
#   { "role": "human", "content": "今天天气怎样？" },
#   { "role": "ai",    "content": "我不太清楚实时天气" }]
    return list(map(
        lambda message: AIMessage(content=message["content"]) if message["role"] == "ai" else AIMessage(content=message["content"]), 
        conversation
        ))


# Orchestrate a single response
def get_response(user_input: str, store, history_dict) -> str:
    history = map_to_history(history_dict)
    retriever_chain = get_context_retriever_chain(store)
    rag_chain = get_conversational_rag_chain(retriever_chain)
    response = rag_chain.invoke({
        'chat_history': history,
        'input': user_input
    })

    return response.dict()

    # print(f"response is {response}")
    # answer = response.answer
    # if response.sources:
    #     answer += "\n\nSources: " + ", ".join(response.sources)
    # print("answer is: ", answer)
    # return answer
    

# history = []

# question = "what is tommy's experience?"
# ans = get_response(question, vector_store, history)
# history.append({"role": "human", "content": question})
# history.append({"role": "ai", "content": ans})
# print("\n\n")

# question = "How many blogs are there?"
# ans = get_response(question, vector_store, history)
# history.append({"role": "human", "content": question})
# history.append({"role": "ai", "content": ans})
# print("\n\n")

# question = "Give me a piece of code snippets used in tommy's blog: grpc-introduction-chapter-2?"
# ans = get_response(question, vector_store, history)
# history.append({"role": "human", "content": question})
# history.append({"role": "ai", "content": ans})
# print("\n\n")

# question = "How many questions did I ask?"
# ans = get_response(question, vector_store, history)
# history.append({"role": "human", "content": question})
# history.append({"role": "ai", "content": ans})


from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/health')
def health():
    return {"status": "ok"}

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    history = data.get('history')
    question = data.get('question')
    ans = get_response(question, vector_store, history)
    history.append({"role": "human", "content": question})
    history.append({"role": "ai", "content": ans})
    return jsonify(ans), 200

if __name__ == '__main__':
    app.run(port=5000)
