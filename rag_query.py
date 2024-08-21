from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Annoy
from langchain_openai import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.cache import InMemoryCache
import tiktoken
import os
from dotenv import load_dotenv
from functools import lru_cache
import langchain

load_dotenv()

# Set up caching
langchain.llm_cache = InMemoryCache()

def truncate_text(text: str, max_tokens: int) -> str:
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return encoding.decode(encoding.encode(text)[:max_tokens])

@lru_cache(maxsize=100)
def load_vector_store():
    embeddings = OpenAIEmbeddings()
    return Annoy.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)

def setup_qa_chain(vector_store):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
        max_token_limit=50
    )
    
    prompt_template = """Use the following context to answer the question concisely:
Context:{context}
Question:{question}
Answer:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})  # Retrieving 2 documents
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT},
    )

@lru_cache(maxsize=1000)
def get_cached_response(query):
    # This function will cache responses for repeated queries
    return get_response(qa_chain, query)

def get_response(qa_chain, query):
    truncated_query = truncate_text(query, 50)
    with get_openai_callback() as cb:
        result = qa_chain({"question": truncated_query})
        tokens_used = cb.total_tokens
        cost = cb.total_cost
    
    answer = truncate_text(result['answer'], 100)
    
    sources = []
    if result.get('source_documents'):
        for i, doc in enumerate(result['source_documents'][:2], 1):  # Process up to 2 documents
            source = doc.metadata.get('source', 'Unknown')
            content = truncate_text(doc.page_content, 50)
            sources.append(f"Source {i}: {source} - Content: {content}")
    
    return {'answer': answer, 'sources': sources, 'tokens_used': tokens_used, 'cost': cost}

def chunk_documents(documents, chunk_size=100, chunk_overlap=10):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_documents(documents)

def main():
    print("Loading vector store...")
    vector_store = load_vector_store()
    
    print("Setting up QA chain...")
    qa_chain = setup_qa_chain(vector_store)
    
    print("RAG system ready. Type 'exit' to quit.")
    
    while True:
        query = input("\nWhat would you like to know about restaurants in Raleigh, NC? ")
        if query.lower() == 'exit':
            break
        
        response = get_cached_response(query)
        
        print("\nAnswer:", response['answer'])
        for source in response['sources']:
            print("\n" + source)
        print(f"\nTokens used: {response['tokens_used']}")
        print(f"Cost: ${response['cost']:.4f}")
        
        print("\nChat History:")
        for message in qa_chain.memory.chat_memory.messages[-2:]:
            print(f"{message.type}: {message.content[:30]}...")

if __name__ == "__main__":
    main()