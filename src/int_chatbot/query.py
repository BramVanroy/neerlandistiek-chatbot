from sqlite3 import Cursor
from typing import Literal

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM

from int_chatbot.utils import format_results


def decide_approach(llm: OllamaLLM, question: str, columns: list[str]) -> Literal["SQL", "RAG"]:
    """
    Let the LLM decide whether to use SQL or RAG to answer a question.

    Args:
        llm: OllamaLLM instance
        question: str, the question to answer

    Returns:
        str: the approach to use, either "SQL" or "RAG"
    """
    prompt = f"""\
Je bent een expert in data-analyse. Beslis of de volgende vraag het beste beantwoord kan worden door:
- SQL: hiervoor zal een LLM een SQL query genereren om specifieke gegevens uit een database op te halen wanneer de gebruiker ernaar verwijst.\
 Beschikbare kolommen in de database zijn: {", ".join(columns)} \
 Dat kan bijvoorbeeld zijn wanneer ze vragen om specifieke kolommen of rijen, of sleutelwoorden gebruiken die relevant\
 zijn voor onze dataset waar SQL van pas kan komen, zoals "hoeveel", "tel", "lijst", "selecteer", "wanneer", "wie", enzovoort.
- RAG: een meer algemene aanpak waarmee de beschikbare tekstuele inhoud via Retrieval Augmented Generation (RAG) zal worden doorzocht.

Gebruikersvraag: {question}

Antwoord uitsluitend met 'SQL' of 'RAG'.\
    """
    response = llm.invoke(prompt).strip().upper()

    if "SQL" in response:
        return "SQL"
    else:
        return "RAG"


def get_table_column_names(cursor: Cursor, table_name: str = "data_table") -> list[str]:
    """
    Get the column names of a table in the SQLite database.

    Args:
        cursor: Cursor instance
        table_name: str, the name of the table

    Returns:
        list[str]: the column names
    """
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = [row[1] for row in cursor.fetchall()]
    return columns


def sql_query(
    llm: OllamaLLM,
    question: str,
    columns: list[str],
    cursor: Cursor,
    table_name: str = "data_table",
    num_retries: int = 3,
) -> list[dict[str, str]]:
    """
    Have an LLM rewrite a user question into a SQL query and execute it. Retry on fail.

    Args:
        llm: OllamaLLM instance
        question: str, the question to answer
        columns: list[str], the column names in the database table
        cursor: Cursor instance
        table_name: str, the name of the table
        num_retries: int, the number of retries to attempt

    Returns:
        list[dict[str, str]]: the formatted results in the form of a list of dictionaries
        where the keys are the column names and the values are the results
    """
    prompt = f"""Je bent een SQL-expert.
Zet de volgende gebruikersvraag om naar een SQLite SQL query zonder uitleg of extra tekst.
De database bevat een tabel '{table_name}' met deze kolommen: {", ".join(columns)}.

Vereisten:
- Gebruik alleen SELECT-queries.
- Gebruik Nederlandse kolomnamen in de output.
- Maak de query zo specifiek mogelijk.
- Gebruik WHERE-clausules en GROUP BY voor aggregaties waar logisch.
- Limiteer resultaten tot 10 rijen tenzij anders gevraagd.
- Let erop dat je spaties gebruikt rondom de SQL sleutelwoorden.
- Gebruik geen markdown of code-markeringen zoals ` of ```; je query begint met SELECT.

Vraag: {question}"""

    print("🤔 Denken over SQL-query...", flush=True)

    orig_prompt = prompt
    while num_retries > 0:
        query = llm.invoke(prompt).strip()

        print(f"🔍 We proberen SQL-query: {query}", flush=True)

        try:
            cursor.execute(query)
            results = cursor.fetchall()
        except Exception as exc:
            num_retries -= 1
            if num_retries == 0:
                print(f"❌ Fout tijdens uitvoeren van query '''{query}''': {exc}. Probeer een andere prompt.")
                return
            print(
                f"❌ Fout tijdens uitvoeren van query '''{query}''': {exc}. De LLM probeert nog een keer ({num_retries} pogingen over)."
            )
            prompt = f"{orig_prompt}\nMislukte poging: {query}\nFoutmelding: {exc}\nVerbeter deze query."
        else:
            break

    print(f"✅ SQL-query uitgevoerd: {query}", flush=True)

    if not results:
        print("🤷 Geen resultaten gevonden.", flush=True)
        return

    formatted_results = format_results([col[0] for col in cursor.description], results)

    print("📊 Databaseresultaten:", flush=True)
    for row in formatted_results:
        for column, value in row.items():
            print(f"- {column}: {value}", flush=True)

    return formatted_results


def rag_query(llm: OllamaLLM, question: str, vector_store: FAISS) -> str:
    """
    Use the RAG approach to answer a question. I.e., search for similar documents in the vector store.

    Args:
        llm: OllamaLLM instance
        question: str, the question to answer
        vector_store: FAISS instance

    Returns:
        str: the answer
    """
    docs = vector_store.similarity_search_with_score(question, k=5)
    print(f"🔍 {len(docs)} relevante passages gevonden! (afstand: lager = beter)", flush=True)
    context = []
    for doc, score in docs:
        title = doc.metadata["titel"]
        url = doc.metadata["link"]
        date = doc.metadata["datum"]
        para_idx = doc.metadata["paragraph_idx"] + 1
        print(f"  - 📄 {title} - {url} (alinea #{para_idx}) (afstand={score:.4f})", flush=True)
        context.append(f"Relevante passage uit artikel '{title}' ({url}) gepubliceerd op {date}: {doc.page_content}")

    context = "\n\n".join(context)

    prompt = f"""Je bent een expert in het analyseren van tekst en artikelen.
Hieronder staan enkele relevante passages uit de dataset en de bijhorende titel en URL. Nadien volgt een vraag die je moet beantwoorden aan de hand van de gegeven passages.\
 Gebruik de informatie uit de passages om de vraag te beantwoorden en citeer de bronnen aan de hand van hun URL met een voetnoot waar dat gepast is.\
 Als (en slechts als) je de informatie van een bron gebruikt, voeg je die toe in Markdown als URL in een voetnoot.\
 De voetnoten met URLs plaats je onderaan, met een lege witregel ervoor om hen te scheiden van je hoofdantwoord. Bijvoorbeeld::

```
Deze zin heeft een URL in een voetnoot.[^1] En deze zin heeft een andere bron in een voetnoot.[^2] En deze laatste zin verwijst ook naar de eerste bron.[^1]


[^1]: https://www.example.com
[^2]: https://www.example.org
```

Je geeft geen voetnoten als er geen relevante passages zijn. Je verwijst dus ENKEL naar de relevante passages en URLs waarnaar je in je tekst verwezen hebt.

---

Passages:

{context}

---

{question}"""

    print("\n🤖 RAG-assistent:", flush=True)

    response = ""
    for chunk in llm.stream(prompt):
        if chunk.strip():
            print(chunk, end="", flush=True)
            response += chunk
    print(flush=True)

    return response
