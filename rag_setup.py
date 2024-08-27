import os
import time
import asyncio
import aiohttp
import ssl
import certifi
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from bs4 import BeautifulSoup
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Initialize OpenAI API
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("Please set OPENAI_API_KEY in your .env file")

def load_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory, filename)
            try:
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
                logging.info(f"Loaded PDF: {filename}")
            except Exception as e:
                logging.error(f"Error loading PDF {filename}: {str(e)}")
    
    logging.info(f"Loaded {len(documents)} documents from local PDF files")
    return documents

async def fetch_url(session, url, ssl_context):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        async with session.get(url, ssl=ssl_context, headers=headers, timeout=30) as response:
            if response.status == 200:
                html_content = await response.text()
                logging.info(f"Successfully fetched: {url}")
                return url, html_content
            else:
                logging.warning(f"Failed to fetch {url}: HTTP {response.status}")
                return url, None
    except asyncio.TimeoutError:
        logging.error(f"Timeout error fetching {url}")
        return url, None
    except Exception as e:
        logging.error(f"Error fetching {url}: {str(e)}")
        return url, None

def parse_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text

async def load_websites_async(urls, verify_ssl=True):
    if not verify_ssl:
        logging.warning("SSL verification is disabled. This may pose security risks.")
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
    else:
        ssl_context = ssl.create_default_context(cafile=certifi.where())

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url, ssl_context) for url in urls]
        results = await asyncio.gather(*tasks)
    
    documents = []
    for url, html_content in results:
        if html_content:
            try:
                text_content = parse_html(html_content)
                doc = Document(page_content=text_content, metadata={"source": url})
                documents.append(doc)
                logging.info(f"Successfully parsed: {url}")
            except Exception as e:
                logging.error(f"Error parsing {url}: {str(e)}")
        else:
            logging.warning(f"Skipping {url} due to failed fetch")
    logging.info(f"Loaded {len(documents)} documents from websites")
    return documents

def load_websites(urls_file, verify_ssl=True):
    with open(urls_file, 'r') as file:
        urls = [line.strip() for line in file if line.strip()]
    logging.info(f"Attempting to load {len(urls)} web pages...")
    return asyncio.run(load_websites_async(urls, verify_ssl))

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)
    logging.info(f"Created {len(split_docs)} text chunks after splitting")
    return split_docs

def create_vector_store(texts):
    embeddings = OpenAIEmbeddings()
    persist_directory = os.path.join(os.getcwd(), "vector_store")
    
    vector_store = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    # Persist the data
    vector_store.persist()
    
    return vector_store

def create_retriever(vector_store):
    base_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = ChatOpenAI(temperature=0)
    compressor = LLMChainExtractor.from_llm(llm)
    retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
    return retriever

def create_rag_chain(retriever, llm):
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

    Context: {context}

    Question: {input}

    Answer: """)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

def main():
    start_time = time.time()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    doc_directory = os.path.join(script_dir, "rag_documents")
    urls_file = os.path.join(script_dir, "urls.txt")
    
    if not os.path.exists(doc_directory):
        raise FileNotFoundError(f"The 'documents' directory does not exist in {script_dir}")
    if not os.path.exists(urls_file):
        raise FileNotFoundError(f"The 'urls.txt' file does not exist in {script_dir}")
    
    raw_documents = load_documents(doc_directory)
    
    web_load_start = time.time()
    web_documents = load_websites(urls_file, verify_ssl=False)
    web_load_end = time.time()
    logging.info(f"Web page loading took {web_load_end - web_load_start:.2f} seconds")
    
    all_documents = raw_documents + web_documents
    
    if not all_documents:
        logging.warning("No documents or web pages found.")
        return
    
    split_texts = split_documents(all_documents)
    
    vector_store_start = time.time()
    vector_store = create_vector_store(split_texts)
    vector_store_end = time.time()
    logging.info(f"Vector store creation took {vector_store_end - vector_store_start:.2f} seconds")
    
    retriever = create_retriever(vector_store)
    llm = ChatOpenAI(temperature=0)
    rag_chain = create_rag_chain(retriever, llm)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    logging.info(f"RAG setup completed with {len(split_texts)} text chunks.")
    logging.info(f"Total processing time: {total_time:.2f} seconds")

    # Log summary of loaded documents
    logging.info("Summary of loaded documents:")
    source_count = {}
    for doc in all_documents:
        source = doc.metadata.get('source', 'Unknown source')
        source_count[source] = source_count.get(source, 0) + 1
    for source, count in source_count.items():
        logging.info(f"  {source}: {count} chunks")

def ensure_vector_store_exists():
    vector_store_path = "vector_store"
    if not os.path.exists(vector_store_path):
        logging.info("Vector store not found. Creating new vector store...")
        main()
    else:
        logging.info("Vector store found.")

if __name__ == "__main__":
    main()

