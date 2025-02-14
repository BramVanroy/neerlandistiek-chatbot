# INT Dummy RAG Chatbot

## 1. Installation

```shell
git clone 
python -m pip install -e .[dev]
```

## 2. Prepare data

Given our CSV file with our neerlandistiek.nl dumps, we should:

- extract text from HTML
- filter out potential samples without text
- save as JSONL file

```shell
python scripts/chat.py data/data.csv -j 16
```

## 3. Run chatbot (auto-generated database and vector store)

Running the chat functionality, we will automatically generated the required data files.

- database: we transfor the JSONL file into a SQLite database, useful if our LLM decides that SQL is a better choice than RAG
- vectorstore: we convert each *paragraph* of a webpage into its own Document, embed it with Jina v3 embeddings, and add it to a vector store. When using RAG, our question will be embedded too, and the most relevant paragraphs will be received from the vector store. These will then be added as context to our query and fed to the LLM. At this stage, metadata such as the URLs, are also included so the LLM can respond with the source.

These processing steps may take a long time the first time you run the script, but they will only need to be executed once. The database and vector store will be created in the same directory as the given data file.

```shell
python scripts/chat.py data/data_extracted.jsonl -d cuda:0 -b 8 --torch_compile -m llama3.3:70b-instruct-q3_K_M
```

You can specify the LLM interface model with `-m`. It should already be pulled with ollama and ollama must be running. I recommend `llama3.3:70b-instruct-q3_K_M`.