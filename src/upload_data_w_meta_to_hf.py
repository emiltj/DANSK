from datasets import Dataset, DatasetInfo, DatasetDict
from pathlib import Path
import json, os


def datasetinfo():
    return DatasetInfo(
        description="DANSK Danish Annotations for NLP Specific TasKs",
        citation="",
        homepage="https://huggingface.co/datasets/chcaa/DANSK",
        version="0.1.0",
        license="cc by-sa 4.0",
    )


def main(info):
    partitions = ["train", "dev", "test"]
    datasets = {}
    for p in partitions:
        json_path = f"data/{p}.jsonl"
        info.description = f"{p.capitalize()} partition of {str(info.description)}"
        datasets[f"{p}"] = Dataset.from_json(json_path, info=info)

    dataset_dict = DatasetDict(
        {"train": datasets["train"], "dev": datasets["dev"], "test": datasets["test"]}
    )
    dataset_dict.push_to_hub("chcaa/DANSK")


if __name__ == "__main__":
    info = datasetinfo()
    main(info)
