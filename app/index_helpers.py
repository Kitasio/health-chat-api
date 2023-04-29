from typing import List, Optional
from langchain.schema import BaseMessage
from llama_index import GPTPineconeIndex, SimpleDirectoryReader
from langchain.memory import RedisChatMessageHistory
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain import OpenAI
from langchain.agents import initialize_agent
import pinecone
import os
import string
import secrets
import redis

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
REDIS_CONN = os.environ.get("REDIS_CONN") or os.environ.setdefault("REDIS_CONN", "redis://localhost:6379")

required_keys = ['PINECONE_API_KEY', 'OPENAI_API_KEY', 'REDIS_CONN']
for key in required_keys:
    if key not in os.environ:
        raise ValueError(f"The {key} environment variable is not set!")

index = None
pinecone_index = None

tools = [
    Tool(
        name = "GPT Index",
        func=lambda q: str(index.query(q)), # type: ignore
        description="useful for when you want to answer questions about what meal to eat. The input to this tool should be a complete english sentence.",
        return_direct=True
    ),
]

def init_pinecone():
    if PINECONE_API_KEY is None:
        return "PINECONE_API_KEY is not set"
    print("Initializing pinecone...")
    pinecone.init(api_key=PINECONE_API_KEY, environment="northamerica-northeast1-gcp")

def create_index(name: str):
    init_pinecone()
    pinecone.create_index(name, dimension=1536, metric="euclidean", pod_type="p1")

def init_index(name: str):
    global index, pinecone_index
    init_pinecone()
    pinecone_index = pinecone.Index(name) 

    print("Creating Index...")
    index = GPTPineconeIndex([], pinecone_index=pinecone_index)

def get_index() -> Optional[GPTPineconeIndex]:
    return index

def query_index(query_text: str, chat_id: str) -> str:
    """Query the index with the given text"""
    global index
    if index is None:
        return "index is empty"

    history = RedisChatMessageHistory(chat_id, url=REDIS_CONN, ttl=60*60*24)
    conversational_memory = ConversationBufferWindowMemory(
        chat_memory=history,
        memory_key='chat_history',
        k=5,
        return_messages=True
    )
    llm=OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name='gpt-3.5-turbo') # type: ignore
    agent_chain = initialize_agent(tools, llm, verbose=True, agent='chat-conversational-react-description', memory=conversational_memory) # type: ignore
    response = agent_chain.run(input=query_text)
    return str(response)

def get_chat_history(chat_id: str) -> List[BaseMessage] | str:
    global index
    if index is None:
        return "index is empty"

    history = RedisChatMessageHistory(chat_id)
    return history.messages

def insert_index(filepath: str) -> str | None:
    global index
    if index is None:
        return None

    doc = SimpleDirectoryReader(input_files=[filepath]).load_data()[0]

    if doc.doc_id is not None:
        save_to_redis(filepath, doc.doc_id)

    index.insert(doc)

    return "inserted to index"

def delete_index(doc_id: str) -> str | None:
    global index
    if index is None:
        return None

    index.delete(doc_id)
    delete_from_redis(doc_id)

    return f"document with id: {doc_id} deleted from index"

def delete_all_indices() -> str | None:
    global pinecone_index
    if pinecone_index is None:
        return None

    pinecone_index.delete(delete_all=True)
    r = redis.from_url(REDIS_CONN)
    r.delete("documents")

    return "all documents deleted from index"

def list_indices() -> List[str]:
    r = redis.from_url(REDIS_CONN)
    result = []
    for filename in r.hkeys("documents"):
        result.append({"filename": filename.decode(), "id": r.hget("documents", filename).decode('utf-8')}) # type: ignore
    return result

def save_to_redis(filepath: str, doc_id: str):
    r = redis.from_url(REDIS_CONN)
    r.hset("documents", get_random_name_from_path(filepath), doc_id)

def delete_from_redis(doc_id: str):
    r = redis.from_url(REDIS_CONN)
    for filename in r.hkeys("documents"):
        if r.hget("documents", filename).decode('utf-8') == doc_id: # type: ignore
            r.hdel("documents", filename)
            return

def get_random_string(length):
    alphabet = string.ascii_letters + string.digits + '-_'
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def get_random_name_from_path(path, length=5):
    filename = os.path.basename(path)
    name, ext = os.path.splitext(filename)
    random_suffix = get_random_string(length)
    return f"{name}_{random_suffix}{ext}"
