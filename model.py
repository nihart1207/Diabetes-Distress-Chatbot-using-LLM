from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_groq import ChatGroq
import chainlit as cl
import redis
import asyncio
from langchain_community.embeddings import HuggingFaceEmbeddings

# Establish a connection to the Redis server for caching responses.
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# Load the FAISS vector store from a predefined path.
DB_FAISS_PATH = 'vectorstore/db_faiss'
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})

# Load the FAISS vector store from a predefined path.
docsearch = FAISS.load_local(DB_FAISS_PATH, embeddings)

# Setup memory and history management for conversation continuity.
message_history = ChatMessageHistory()
memory = ConversationBufferMemory(
    memory_key="chat_history",
    output_key="answer",
    chat_memory=message_history,
    return_messages=True,
)

# Initialize the ChatGroq model.
llm_groq = ChatGroq(
    model_name='mixtral-8x7b-32768'
    # model_name='llama3-8b-8192'

)

# Configure the conversational chain with the text data and memory systems.
chain = ConversationalRetrievalChain.from_llm(
    llm=llm_groq,
    retriever=docsearch.as_retriever(),
    memory=memory,
    return_source_documents=True,
)

# Store the configured chain globally for application-wide access.
global_chain = chain

@cl.on_chat_start
async def on_chat_start():
    msg = cl.Message(content="Welcome to Iconcern, your diabetes chat assistant. How may I help you today?", author="Iconcern")
    await msg.send()
    
@cl.on_message
async def main(message: cl.Message):
    user_query = message.content.strip().lower()

    # Check cache first
    cached_response = redis_client.get(user_query)
    if cached_response:
        await cl.Message(content=cached_response, author="Iconcern").send()
        return

    # Indicate processing
    await cl.Message(content="Processing your request...", author="Iconcern").send()

    # Process the query using the conversational chain
    res = await global_chain.ainvoke(user_query)
    answer = res["answer"]
    source_documents = res["source_documents"]

    text_elements = []
    source_info = ""

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            text_elements.append(cl.Text(content=source_doc.page_content, name=source_name))
        source_names = [text_el.name for text_el in text_elements]
        source_info = f"\nSources: {', '.join(source_names)}" if source_names else "\nNo sources found"

    full_response = answer + source_info
    redis_client.set(user_query, full_response)

    # Send the final response to the user
    await cl.Message(content=full_response, author="Iconcern", elements=text_elements).send()

# # Start the chat server
# cl.run()
