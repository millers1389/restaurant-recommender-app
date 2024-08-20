import streamlit as st
from rag_query import load_vector_store, setup_qa_chain, get_response
import os
from dotenv import load_dotenv
from arize_instrument import arize_instrument
from openai import OpenAIError

# Load environment variables
load_dotenv()

def setup_arize_if_needed():
    if os.getenv('ARIZE_INSTRUMENTED') != 'true':
        # Get api key, space id, and input model_id and model_version
        space_id = os.getenv("SPACE_ID")
        api_key = os.getenv("ARIZE_API_KEY")
        model_id = "raleigh-restaurant-recommender"
        model_version = "V1.01"

        # Run Arize setup
        arize_instrument(
            space_id=space_id,
            api_key=api_key,
            model_id=model_id,
            model_version=model_version
        )
        
        # Set the environment variable
        os.environ['ARIZE_INSTRUMENTED'] = 'true'
        
        print("Arize setup completed.")
    else:
        print("Arize already instrumented. Skipping setup.")

# Run the setup check
setup_arize_if_needed()

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("Please set your OpenAI API key as an environment variable 'OPENAI_API_KEY'")
    st.stop()

# Set page configuration
st.set_page_config(page_title="Raleigh Restaurant Recommender", page_icon="🍽️", layout="wide")

# Initialize session state
if 'qa_chain' not in st.session_state:
    with st.spinner("Loading knowledge base... This may take a moment."):
        vector_store = load_vector_store()
        st.session_state.qa_chain = setup_qa_chain(vector_store)

if 'messages' not in st.session_state:
    st.session_state.messages = []

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
                full_response = f"{response['answer']}\n\nSources:"
                for source in response['sources']:
                    full_response += f"\n- {source}"
                message_placeholder.markdown(full_response)
                
                # Display token usage and cost
                st.sidebar.subheader("Last Query Statistics:")
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    st.metric("Tokens Used", response['tokens_used'])
                with col2:
                    st.metric("Query Cost", f"${response['cost']:.4f}")
            
            except OpenAIError as e:
                if "maximum context length" in str(e).lower():
                    error_message = "I'm sorry, but that question is too complex for me to answer. Could you please ask a simpler or shorter question?"
                else:
                    error_message = "I encountered an error while processing your request. Could you please try asking your question differently?"
                message_placeholder.markdown(error_message)
                full_response = error_message
            
            except Exception as e:
                error_message = "An unexpected error occurred. Please try asking your question again."
                message_placeholder.markdown(error_message)
                full_response = error_message
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Additional information
st.sidebar.title("About")
st.sidebar.info(
    "This app uses AI to provide information about restaurants in Raleigh, NC. "
    "It's powered by a RAG (Retrieval-Augmented Generation) system, which combines "
    "a knowledge base about local restaurants with the ability to generate human-like responses."
)
st.sidebar.warning(
    "Please note that the AI's knowledge is based on its training data and may not include "
    "the very latest information about restaurants. Always verify important details directly "
    "with the restaurants."
)

# Clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.qa_chain.memory.clear()
    st.experimental_rerun()