from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.callbacks.manager import get_openai_callback
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import os
from dotenv import load_dotenv
from functools import lru_cache
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

MAX_TOKENS = 12000  # Reduced from 14000 to 12000 for an extra safety margin
MAX_QUERY_TOKENS = 500
MAX_DOCUMENT_TOKENS = 1000
MAX_HISTORY_TOKENS = 2000

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def truncate_text(text: str, max_tokens: int) -> str:
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    return encoding.decode(encoding.encode(text)[:max_tokens])

@lru_cache(maxsize=100)
def load_vector_store():
    embeddings = OpenAIEmbeddings()
    persist_directory = os.path.join(os.getcwd(), "vector_store")
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

def setup_qa_chain(vector_store):
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)
    
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
        k=3  # Only keep the last 3 interactions
    )
    
    prompt_template = """Use the following context to answer the question concisely. Provide a response that is between 2 and 4 sentences long. Be informative but brief:

Context: {context}

Question: {question}

Concise Answer (2-4 sentences):"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT},
    )

def get_response(qa_chain, query):
    logger.info(f"Processing query: {query}")
    query_tokens = num_tokens_from_string(query)
    logger.info(f"Query tokens: {query_tokens}")

    # Truncate query if it's too long
    if query_tokens > MAX_QUERY_TOKENS:
        query = truncate_text(query, MAX_QUERY_TOKENS)
        logger.info(f"Query truncated. New token count: {num_tokens_from_string(query)}")

    with get_openai_callback() as cb:
        # Estimate tokens from conversation history
        chat_history = qa_chain.memory.chat_memory.messages
        chat_history_tokens = sum(num_tokens_from_string(str(msg)) for msg in chat_history)
        
        # Truncate chat history if it's too long
        while chat_history_tokens > MAX_HISTORY_TOKENS and chat_history:
            removed_message = chat_history.pop(0)
            chat_history_tokens -= num_tokens_from_string(str(removed_message))

        # Run the query
        result = qa_chain({"question": query, "chat_history": chat_history})
        
        logger.info(f"Total tokens: {cb.total_tokens}")
        logger.info(f"Prompt tokens: {cb.prompt_tokens}")
        logger.info(f"Completion tokens: {cb.completion_tokens}")
        
        # Log retrieved documents
        for i, doc in enumerate(result.get('source_documents', [])):
            doc_tokens = num_tokens_from_string(doc.page_content)
            logger.info(f"Document {i+1} tokens: {doc_tokens}")
            logger.info(f"Document {i+1} content preview: {doc.page_content[:100]}...")

        logger.info(f"Chat history tokens: {chat_history_tokens}")

    answer = result['answer']

    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', answer)
    if len(sentences) < 2:
        logger.warning("Response too short, keeping as is.")
    elif len(sentences) > 4:
        answer = ' '.join(sentences[:4])
        logger.info("Response truncated to 4 sentences.")

    sources = [f"{doc.metadata.get('source', 'Unknown')} - {doc.page_content[:100]}..." for doc in result.get('source_documents', [])]

    return {
        'answer': answer,
        'sources': sources,
        'tokens_used': cb.total_tokens,
        'cost': cb.total_cost
    }

# TODO: This is never called from app.py, so it's not being used.
def main():
    print("This is an example change.")
    print("Loading vector store...")
    vector_store = load_vector_store()
    
    print("Setting up QA chain...")
    qa_chain = setup_qa_chain(vector_store)
    
    print("RAG system ready. Type 'exit' to quit.")
    
    while True:
        query = input("\nWhat would you like to know about restaurants in Raleigh, NC? ")
        if query.lower() == 'exit':
            break
        
        try:
            response = get_response(qa_chain, query)
            
            print("\nAnswer:", response['answer'])
            print("\nSources:")
            for source in response['sources']:
                print(source)
            print(f"\nTokens used: {response['tokens_used']}")
            print(f"Cost: ${response['cost']:.4f}")
            
            print("\nChat History:")
            for message in qa_chain.memory.chat_memory.messages[-2:]:
                print(f"{message.type}: {message.content[:50]}...")
        
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            logger.exception("Error in processing query")

if __name__ == "__main__":
    main()