## Simple RAG-based chatbot

Simple toy chatbot for testing different chat components.

To run do the following:
 - git clone langfuse: `git clone https://github.com/langfuse/langfuse.git` and run  it
`cd langfuse` and `docker compose up`
 - open `http://localhost:3000/` and register a user and then create a project. Under project settings generate secret and public API keys
 - put them into `.env file` 
 - add documents to `rag_docs/` folder (should be `.pdf`)
 - use `backend/.env_template` and create `.env`file, fill in the `OPENAI_API_KEY=`
 - build RAG database: `docker compose build loader` or 
`docker compose up --build loader`. Do this once (or when docs change), every time all docs are embedded and inserted into the database.
 You don't have to reduild it every time you run the bot. 
Data is persistent once docker image has been built (you can see it in `weaviate_storage`folder). Rebuild it when documents change.
 - build and run the bot: `docker compose up --build backend` (for running only `docker compose up backend`)
 - open `http://localhost:8000/` and chat
 - open `http://localhost:3000/` to review the traces
 - to close chat: `docker compose stop backend` and `docker compose rm -f backend`