import spacy, re, json, datetime, os
import pandas as pd
from datasets import load_dataset
from spacy.tokens import DocBin
from multiprocessing import Pool, cpu_count
from functools import partial


# Create mapping of source to domain:
source_domain_mapping = {
    "retsinformationdk": "Legal",
    "skat": "Legal",
    "retspraksis": "Legal",
    "hest": "Social Media",
    "cc": "Web",
    "adl": "Wiki & Books",
    "botxt": "Other",
    "danavis": "News",
    "dannet": "dannet",
    "depbank": "Other",
    "ep": "Conversation",
    "ft": "Conversation",
    "gutenberg": "Wiki & Books",
    "jvj": "Wiki & Books",
    "naat": "Conversation",
    "opensub": "Conversation",
    "relig": "Wiki & Books",
    "spont": "Conversation",
    "synne": "Other",
    "tv2r": "News",
    "wiki": "Wiki & Books",
    "wikibooks": "Wiki & Books",
    "wikisource": "Wiki & Books",
    "twfv19": "Social Media",
}

# Create mapping of source to long_source names
source_extended_mapping = {
    "retsinformationdk": "retsinformation.dk (Danish legal information)",
    "skat": "Skat (Danish tax authority)",
    "retspraksis": "retspraksis (Danish legal information)",
    "hest": "Hestenettet (Danish debate forum)",
    "cc": "Common Crawl",
    "adl": " Archive for Danish Literature",
    "botxt": "Bornholmsk (Danish dialect)",
    "danavis": "Danish daily newspapers",
    "dannet": "DanNet (Danish WordNet)",
    "depbank": "Danish Dependency Treebank",
    "ep": "European Parliament",
    "ft": "Folketinget (Danish Parliament)",
    "gutenberg": "Gutenberg",
    "jvj": "Johannes V. Jensen (Danish poet)",
    "naat": "NAAT",
    "opensub": "Open Subtitles",
    "relig": "Religious texts",
    "spont": "Spontaneous speech",
    "synne": "Synderjysk (Danish dialect)",
    "tv2r": "TV 2 Radio (Danish news)",
    "wiki": "Wikipedia",
    "wikibooks": "Wikibooks",
    "wikisource": "Wikisource",
    "twfv19": "Twitter Folketingsvalget 2019 (Danish election tweets)",
}

# Define function for loading in DANSK:
def load_dansk():
    # Import DANSK and the partitions
    partitions = ["train", "dev", "test"]  # ["full", "train", "dev", "test"]
    nlp = spacy.blank("da")
    docs_dict = {}
    for p in partitions:
        doc_bin = DocBin().from_disk(f"data/{p}.spacy")
        docs = list(doc_bin.get_docs(nlp.vocab))
        docs_dict[f"{p}"] = docs
    return partitions, docs_dict


# Define func for making docs to json
def convert_list_of_docs_to_json(docs):
    return [doc.to_json() for doc in docs]


# Define func for loading all annotator data, where the metadata is
def load_meta_data_as_df():
    meta_data_info = pd.DataFrame()
    for i in list(range(1, 11)):
        df = pd.read_json(
            path_or_buf=f"data/prodigy_exports/prodigy{i}_db_exports/NER_merged_annotator{i}.jsonl",
            lines=True,
        )
        meta_data_info = pd.concat([meta_data_info, df])
    return meta_data_info


# Define func for making meta_data as dict
def meta_as_dict(meta_data):
    meta_data_dict = {}
    for index, row in meta_data.iterrows():
        text = row["text"]
        meta = row["meta"]
        meta_data_dict[f"{text}"] = meta
    return meta_data_dict


# Define func for removing any items that don't contain source
def dict_only_source(meta_data_dict):
    new_dict = {}
    for key, value in list(meta_data_dict.items()):
        if "source" in list(value.keys()):
            new_dict[f"{key}"] = value["source"]
    return new_dict


# Define func for values only including source
def meta_data_only_source(meta_data_dict):
    sources = list(source_domain_mapping.keys())
    new_values = {}
    for key, value in list(meta_data_dict.items()):
        for source in sources:
            if re.search(f"{str(source)}_", f"{str(value)}_"):
                new_values[f"{key}"] = source
    return new_values


# Define func for adding dagw_source
def add_meta_to_json_docs(json_docs, meta_data_dict):
    new_docs_json = json_docs
    for i, d in enumerate(json_docs):
        text = d["text"]
        for key, val in meta_data_dict.items():
            if text == key:
                new_docs_json[i]["dagw_source"] = val
                new_docs_json[i]["dagw_domain"] = source_domain_mapping[f"{val}"]
                new_docs_json[i]["dagw_source_full"] = source_extended_mapping[f"{val}"]
    return new_docs_json


# Defining func for writing jsonl files from a list of dicts (output from doc.to_json())
def write_to_jsonl(json_docs, outpath):
    with open(f"{outpath}", "w") as outfile:
        for json_doc in json_docs:
            json.dump(json_doc, outfile)
            outfile.write("\n")


# Defining main function
def main():
    meta_data = load_meta_data_as_df()
    meta_data_dict = meta_as_dict(meta_data)
    meta_data_dict = dict_only_source(meta_data_dict)
    meta_data_dict = meta_data_only_source(meta_data_dict)
    partitions, docs_dict = load_dansk()

    for p in partitions:
        docs = docs_dict[f"{p}"]
        docs_json = convert_list_of_docs_to_json(docs)
        docs_json_w_meta = add_meta_to_json_docs(docs_json, meta_data_dict)
        write_to_jsonl(docs_json_w_meta, f"data/{p}.jsonl")


if __name__ == "__main__":
    main()
