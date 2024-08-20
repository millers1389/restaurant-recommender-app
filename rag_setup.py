import os
import time
import asyncio
import aiohttp
import ssl
import certifi
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Annoy
from langchain.docstore.document import Document
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check for tiktoken
try:
    import tiktoken
except ImportError:
    logging.error("tiktoken package is not installed. Please install it with 'pip install tiktoken'")
    exit(1)

# Load environment variables
load_dotenv()

# Initialize OpenAI API
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("Please set OPENAI_API_KEY in your .env file")

def load_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith('.txt'):
            logging.info(f"Loading TXT: {filename}")
            loader = TextLoader(file_path, encoding='utf8')
            documents.extend(loader.load())
        elif filename.endswith('.pdf'):
            logging.info(f"Loading PDF: {filename}")
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
    logging.info(f"Loaded {len(documents)} documents from local files")
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
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text()
    # Break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # Break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # Drop blank lines
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
    logging.info(f"Splitting {len(documents)} documents...")
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)  # Reduced chunk size
    split_docs = text_splitter.split_documents(documents)
    logging.info(f"Created {len(split_docs)} text chunks after splitting")
    return split_docs

def create_vector_store(texts):
    logging.info(f"Creating embeddings and vector store for {len(texts)} text chunks...")
    try:
        embeddings = OpenAIEmbeddings()
        vector_store = Annoy.from_documents(texts, embeddings)
        return vector_store
    except Exception as e:
        logging.error(f"Error creating vector store: {str(e)}")
        raise

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
    
    vector_store.save_local("vector_store")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    logging.info(f"Vector store created and saved with {len(split_texts)} text chunks.")
    logging.info(f"Total processing time: {total_time:.2f} seconds")

    # Log summary of loaded documents
    logging.info("Summary of loaded documents:")
    source_count = {}
    for doc in all_documents:
        source = doc.metadata.get('source', 'Unknown source')
        source_count[source] = source_count.get(source, 0) + 1
    for source, count in source_count.items():
        logging.info(f"  {source}: {count} chunks")

if __name__ == "__main__":
    main()