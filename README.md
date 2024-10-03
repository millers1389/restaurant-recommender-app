# __Restaurant Recommender App__

## Description:
This repository contains files that build a rag-based LLM application that provides restaurant recommendations in and around Raleigh, North Carolina based on a user input.
* The application includes a UI in which the user will ask a question about what type of restaurant they are looking for, and the chat bot will reply with recommendations and details about the restaurant.
* The application references documents and websites about Raleigh restaurants to provide its response
* The application also instruments into the user's choice of Phoenix or Arize in order to provide visibility into the calls the application is making under the hood

## Files:
* [requirements.txt](https://github.com/millers1389/restaurant-recommender-app/blob/main/requirements.txt): list of dependencies to install
* [urls.txt](https://github.com/millers1389/restaurant-recommender-app/blob/main/urls.txt): list of urls for building RAG documentation
* [rag_documents](https://github.com/millers1389/restaurant-recommender-app/tree/main/rag_documents): list of pdfs to be added to the vector store in the rag_setup step
* [rag_setup.py](https://github.com/millers1389/restaurant-recommender-app/blob/main/rag_setup.py): creates the vector store with relevant documents and urls about Raleigh restaurants
* [rag_query.py](https://github.com/millers1389/restaurant-recommender-app/blob/main/rag_query.py): sets up the conditions/prompt for the llm response
* [phoenix_instrument.py](https://github.com/millers1389/restaurant-recommender-app/blob/main/phoenix_instrument.py): sets up instrumentation of traces/spans into Phoenix
* [arize_instrument.py](https://github.com/millers1389/restaurant-recommender-app/blob/main/arize_instrument.py): sets up instrumentation of traces/spans into Arize
* [app.py](https://github.com/millers1389/restaurant-recommender-app/blob/main/app.py): builds/runs the application including the UI where a user can directly converse with the chat bot
