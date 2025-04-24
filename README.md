## Simple RAG-based chatbot

Simple toy chatbot for testing different chat components.

To run do the following:
 - add documents to `rag_docs/` folder (should be `.txt`)
 - use `backend/.env_template` and create `.env`file, fill in the `OPENAI_API_KEY=`
 - build RAG database: `docker compose build loader` or 
`docker compose up --build loader`. 
 You don't have to reduild it every time you run the bot. 
Data is persistent once docker image has been built. Rebuild it when documents change.
 - build and run the bot: `docker compose up --build`
 - open `http://localhost:8000/` and chat