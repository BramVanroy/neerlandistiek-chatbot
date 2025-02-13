from pathlib import Path

from datasets import load_dataset
from trafilatura import extract


def main(csv_path: str, num_proc: int | None = None):
    ds = load_dataset("csv", data_files=csv_path, split="train")
    ds = ds.map(
        lambda item: {"inhoud": extract(item["inhoud"]), "excerpt": extract(item["excerpt"])}, num_proc=num_proc
    )
    ds = ds.filter(lambda item: item["inhoud"] is not None or item["excerpt"] is not None, num_proc=num_proc)
    pfin = Path(csv_path)
    ds.to_json(pfin.with_name(pfin.stem + "_extracted.jsonl"), orient="records", lines=True, index=False)


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser(description="Extract text from HTML content and save as JSONL")
    cparser.add_argument("csv_path", type=str, help="Path to the CSV file")
    cparser.add_argument(
        "-j", "--num_proc", type=int, help="Number of processes to use for parallel processing", default=None
    )
    cargs = cparser.parse_args()
    main(**vars(cargs))
