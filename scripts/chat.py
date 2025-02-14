import sqlite3
from pathlib import Path

from langchain_ollama import OllamaLLM

from int_chatbot.data import jsonl_to_sqlite
from int_chatbot.embeddings import build_vector_store
from int_chatbot.query import decide_approach, get_table_column_names, rag_query, sql_query


def main(
    jsonl_file: str | Path,
    ollama_model: str = "llama3",
    embedder_device: str = "cpu",
    batch_size: int = 8,
    torch_compile: bool = False,
    allow_sql: bool = True,
):
    print(f"📂 Laden JSONL-bestand en omzetten naar SQLite: {jsonl_file}", flush=True)
    db_file = str(Path(jsonl_file).with_suffix(".db"))
    db_file = jsonl_to_sqlite(jsonl_file, db_file=db_file)

    # Re-open connection in read-only
    conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
    cursor = conn.cursor()

    db_columns = get_table_column_names(cursor)

    print(f"ℹ️ Geladen kolommen in SQLite: {', '.join(db_columns)}", flush=True)

    dir_store_name = Path(jsonl_file).parent.joinpath("vector_store")
    vector_store = build_vector_store(
        jsonl_file,
        dir_store_name=dir_store_name,
        embedder_device=embedder_device,
        batch_size=batch_size,
        torch_compile=torch_compile,
    )

    llm = OllamaLLM(model=ollama_model)

    print("🤖 Chat Bot gestart!", flush=True)
    print("❌ Type 'stop' om te stoppen\n", flush=True)

    if allow_sql:
        print(
            "💡 De LLM zal zelf beslissen of er een exact antwoord in de database gezocht kan worden,"
            " of dat we toch eerder de RAG-toer op gaan!",
            flush=True,
        )

    while True:
        try:
            user_input = input("\n💭 Vraag: ").strip()

            if user_input.lower() in ["stop", "quit", "exit"]:
                print("👋 Tot ziens!", flush=True)
                break

            if allow_sql:
                approach = decide_approach(llm, user_input, db_columns)
                print(f"🔍 Gekozen aanpak: {approach}", flush=True)

                if approach == "SQL":
                    sql_query(llm, user_input, db_columns, cursor)
                else:
                    rag_query(llm, user_input, vector_store)
            else:
                rag_query(llm, user_input, vector_store)

        except KeyboardInterrupt:
            print("👋 Tot ziens!", flush=True)
            break
        except Exception as exc:
            raise Exception(f"❌ Fout: {exc}") from exc

    conn.close()


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(description="Chat Bot voor INT")
    cparser.add_argument("jsonl_file", type=str, help="Pad naar JSONL-bestand")

    cparser.add_argument("-m", "--ollama_model", type=str, help="Ollama model", default="llama3")
    cparser.add_argument(
        "-d",
        "--embedder_device",
        type=str,
        default="cpu",
        help="Apparaat voor embeddings (cpu of cuda of cuda:0, cuda:1, etc.)",
    )
    cparser.add_argument("-b", "--batch_size", type=int, default=8, help="Batchgrootte voor embeddings")
    cparser.add_argument("--torch_compile", action="store_true", help="Gebruik TorchScript compilatie voor embeddings")
    cparser.add_argument("--disable_sql", dest="allow_sql", action="store_false", help="Schakel SQL-query's uit")

    cargs = cparser.parse_args()
    main(**vars(cargs))

    # Wat is de status van de Friese taal?
    # Wat is het laatste nieuws over Papiaments?
    # Hoeveel unieke auteurs hebben gepubliceerd op neerlandistiek.nl?
    # Hoeveel artikelen staan er op neerlandistiek.nl?
