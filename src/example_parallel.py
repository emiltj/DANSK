"""
A simple example of parallel processing using the multiprocessing module.
"""

import multiprocess as mp

# import multiprocessing as mp
from functools import partial

# from multiprocess import Pool, cpu_count


def check_string(document, strings_to_check):
    """
    Check if a string is in a corpus of documents.

    returns strings in document
    """
    return [string for string in strings_to_check if string in document]


corpus_of_documents = [
    ("hello my name is kenneth", "categrory1"),
    ("hello bye bye", "category2"),
    ("hello hello hello", "category3"),
]

documents = [document[0] for document in corpus_of_documents]
categories = [document[1] for document in corpus_of_documents]

strings_to_check = [
    "hello",
    "bye",
]

# Create a partial function to pass to the pool
partial_check_string = partial(check_string, strings_to_check=strings_to_check)

# Create a pool of processes
cpus_to_use = mp.cpu_count() - 1

# # Run the pool
# pool = mp.Pool(processes=cpus_to_use)
# results = pool.map(partial_check_string, corpus_of_documents)
# # Close the pool
# pool.close()

# or simpler (and safer) using a with statement
with mp.Pool(processes=cpus_to_use) as pool:
    results = pool.map(partial_check_string, documents)

string_results = {s: {"category": []} for s in strings_to_check}
for cat, result in zip(categories, results):
    for string in result:
        string_results[string]["category"].append(cat)

print(string_results)
