import streamlit as st
from rag_query import load_vector_store, setup_qa_chain, get_response
import os
from dotenv import load_dotenv
from arize_instrument import arize_instrument
from phoenix_instrument import setup_phoenix_instrumentation
from openai import OpenAIError
import traceback
import logging
from rag_setup import ensure_vector_store_exists
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@st.cache_resource
def setup_instrumentation(instrumentation_type):
    if instrumentation_type == 'arize':
        space_id = os.getenv("SPACE_ID")
        api_key = os.getenv("ARIZE_API_KEY")
        model_id = "raleigh-restaurant-recommender"
        model_version = "V1.01"
        arize_instrument(space_id, api_key, model_id, model_version)
    elif instrumentation_type == 'phoenix':
        setup_phoenix_instrumentation()
    else:
        logger.warning(f"Unknown instrumentation type: {instrumentation_type}. No instrumentation set up.")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Raleigh Restaurant Recommender")
    parser.add_argument('--instrument', choices=['arize', 'phoenix'], help='Choose instrumentation type: arize or phoenix')
    args = parser.parse_args()

    if args.instrument:
        setup_instrumentation(args.instrument)

    # Ensure vector store exists
    ensure_vector_store_exists()

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Please set your OpenAI API key as an environment variable 'OPENAI_API_KEY'")
        st.stop()

    

    # Initialize session state
    if 'qa_chain' not in st.session_state:
        with st.spinner("Loading knowledge base... This may take a moment."):
            try:
                vector_store = load_vector_store()
                st.session_state.qa_chain = setup_qa_chain(vector_store)
            except FileNotFoundError:
                st.error("Failed to load vector store. Please try refreshing the page.")
                st.stop()

    # Initialize other session state variables
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'sources' not in st.session_state:
        st.session_state.sources = []

    if 'last_tokens_used' not in st.session_state:
        st.session_state.last_tokens_used = 0

    if 'last_query_cost' not in st.session_state:
        st.session_state.last_query_cost = 0.0

    # Main UI
    st.title("🍽️ Raleigh Restaurant Recommender")

    st.write("""
Welcome to the Raleigh Restaurant Recommender! Ask me anything about restaurants in Raleigh, NC.
I can help with cuisine types, popular dishes, locations, and more.
""")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("What would you like to know about restaurants in Raleigh, NC?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                try:
                    response = get_response(st.session_state.qa_chain, prompt)
                    logger.info(f"Raw response: {response}")  # Log the raw response
                    
                    if not isinstance(response, dict):
                        raise ValueError(f"Expected dict response, got {type(response)}")
                    
                    if 'answer' not in response:
                        raise KeyError("'answer' key not found in response")
                    
                    answer = response['answer']
                    message_placeholder.markdown(answer)
                    
                    # Update sources and statistics in session state
                    st.session_state.sources = [source.split(' - ')[0] for source in response.get('sources', [])]
                    st.session_state.last_tokens_used = response.get('tokens_used', 0)
                    st.session_state.last_query_cost = response.get('cost', 0.0)
                    
                except OpenAIError as e:
                    logger.error(f"OpenAI API error: {str(e)}")
                    if "maximum context length" in str(e).lower():
                        answer = "I'm sorry, but that question is too complex for me to answer. Could you please ask a simpler or shorter question?"
                    else:
                        answer = f"I encountered an error while processing your request: {str(e)}"
                    message_placeholder.markdown(answer)
                    # Set token usage and sources to empty for errors
                    st.session_state.last_tokens_used = 0
                    st.session_state.last_query_cost = 0.0
                    st.session_state.sources = []
                
                except Exception as e:
                    logger.error(f"Unexpected error: {str(e)}")
                    logger.error(traceback.format_exc())
                    answer = f"An unexpected error occurred: {str(e)}. Please try asking your question again."
                    message_placeholder.markdown(answer)
                    # Set token usage and sources to empty for errors
                    st.session_state.last_tokens_used = 0
                    st.session_state.last_query_cost = 0.0
                    st.session_state.sources = []
    
        st.session_state.messages.append({"role": "assistant", "content": answer})

    # Sidebar
    st.sidebar.title("Sources and Statistics")


    # Display token usage and cost in the sidebar
    st.sidebar.subheader("Last Query Statistics")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Tokens Used", st.session_state.last_tokens_used)
    with col2:
        st.metric("Query Cost", f"${st.session_state.last_query_cost:.4f}")

    # Display sources in the sidebar
    if st.session_state.sources:
        st.sidebar.subheader("Sources for Last Query")
        for i, source in enumerate(st.session_state.sources, 1):
            st.sidebar.markdown(f"{i}. {source}")
    else:
        st.sidebar.info("No sources available for the last query.")

    # Additional information
    st.sidebar.title("About")
    st.sidebar.info(
        "This app uses AI to provide information about restaurants in Raleigh, NC. "
    )
    st.sidebar.warning(
        "Please note that the AI's knowledge may not include the very latest information about restaurants."
    )

    # Clear chat button
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.sources = []
        st.session_state.qa_chain.memory.clear()
        # Reset token statistics
        st.session_state.last_tokens_used = 0
        st.session_state.last_query_cost = 0.0

if __name__ == "__main__":
    # Set page configuration
    st.set_page_config(page_title="Raleigh Restaurant Recommender", page_icon="🍽️", layout="wide")
    main()