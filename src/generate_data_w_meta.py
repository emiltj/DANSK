import spacy, re, json
from datasets import load_dataset
from spacy.tokens import DocBin, Doc
import datetime

# from multiprocess import Pool, cpu_count

from multiprocessing import Pool, cpu_count
from functools import partial


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

# Defining func for retrieve source from dagw for a list of docs
def get_meta_list_of_docs(
    dagw_docs, docs, source_domain_mapping, source_extended_mapping
):
    json_docs_w_meta = docs
    for json_doc_w_meta in json_docs_w_meta:
        json_doc_w_meta["dagw_source"] = []
        json_doc_w_meta["dagw_source_extended"] = []
        json_doc_w_meta["dagw_domain"] = []
    dagw_len = len(dagw_docs)
    for i, dagw_doc in enumerate(dagw_docs):
        if i % 100 == 0:
            print(
                f"\nProgress: {round((i/dagw_len)*100, 5)} percent done with current partition"
            )
            print(datetime.datetime.now())
        for d, doc in enumerate(docs):
            # print(dagw_doc)
            # print("\n")
            # print(dagw_doc["text"])
            # print("\n")
            # print(re.escape(dagw_doc["text"]))
            # print("\n")
            # print(i)
            # print("\n")
            # print(re.search(re.escape(doc["text"]), re.escape(dagw_doc["text"])))
            # print("\n")
            # print(d)
            # print("\n")
            # print(doc["text"])
            # print("\n")
            # print(re.escape(doc["text"]))
            # print("\n")
            if (
                re.search(re.escape(doc["text"]), re.escape(dagw_doc["text"]))
                and dagw_doc["source"] not in json_docs_w_meta[d]["dagw_source"]
            ):
                json_docs_w_meta[d]["dagw_source"].append(dagw_doc["source"])
                source = dagw_doc["source"]
                json_docs_w_meta[d]["dagw_domain"].append(source_domain_mapping[source])
                json_docs_w_meta[d]["dagw_source_extended"].append(
                    source_extended_mapping[source]
                )
    return json_docs_w_meta


# Defining func for writing jsonl files from a list of doc.to_json() objects (dicts)
def write_to_jsonl(json_docs, outpath):
    with open(f"{outpath}", "w") as outfile:
        for json_doc in json_docs:
            json.dump(json_doc, outfile)
            outfile.write("\n")


# Defining main function
def main(docs_dict, partitions, dagw, source_domain_mapping, source_extended_mapping):
    for p in partitions:
        print(f"Retrieving meta data for the partition: {p}")
        docs = docs_dict[f"{p}"]
        json_docs = convert_list_of_docs_to_json(docs)
        json_docs_w_meta = get_meta_list_of_docs(
            dagw, json_docs, source_domain_mapping, source_extended_mapping
        )
        write_to_jsonl(json_docs_w_meta, f"data/{p}.jsonl")


# https://stackoverflow.com/questions/25553919/passing-multiple-parameters-to-pool-map-function-in-python
# Defining main function with parallel processing
def main_parallel(
    docs_dict, partitions, dagw, source_domain_mapping, source_extended_mapping
):
    for p in partitions:
        print(f"Retrieving meta data for the partition: {p}")
        docs = docs_dict[f"{p}"]
        json_docs = convert_list_of_docs_to_json(docs)
        # Create a partial function to pass to the pool
        partial_func = partial(
            get_meta_list_of_docs,
            source_domain_mapping=source_domain_mapping,
            source_extended_mapping=source_extended_mapping,
            docs=json_docs,
            # dagw_docs=dagw,
        )
        with Pool(processes=cpu_count()) as pool:
            json_docs_w_meta = pool.map(partial_func, dagw)
        write_to_jsonl(json_docs_w_meta, f"data/{p}.jsonl")


if __name__ == "__main__":
    subset = False
    parallel_processing = True

    if not subset:
        # Import DAGW-no-twitter
        dagw = load_dataset("DDSC/partial-danish-gigaword-no-twitter")["train"]

        # Import DANSK
        partitions, docs_dict = load_dansk()

        # Get metadata for docs and save as jsonl
        if parallel_processing:
            main_parallel(
                docs_dict,
                partitions,
                dagw,
                source_domain_mapping,
                source_extended_mapping,
            )
        else:
            main(
                docs_dict,
                partitions,
                dagw,
                source_domain_mapping,
                source_extended_mapping,
            )

    else:
        dagw_subset = load_dataset(
            "DDSC/partial-danish-gigaword-no-twitter", split="train[:20%]"
        )

        # Import DANSK
        partitions, docs_dict = load_dansk()
        docs_dict_subset = {f"{p}": docs_dict[f"{p}"][:5] for p in partitions}

        # Get metadata for docs and save as jsonl
        if parallel_processing:
            main_parallel(
                docs_dict_subset,
                partitions,
                dagw_subset,
                source_domain_mapping,
                source_extended_mapping,
            )

        else:
            main(
                docs_dict_subset,
                partitions,
                dagw_subset,
                source_domain_mapping,
                source_extended_mapping,
            )
