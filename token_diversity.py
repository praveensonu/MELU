import os
import json
import pandas as pd
from typing import List

def distinct_n(sentences: List[str], n: int = 1) -> float:
    """
    Compute Distinct-N for a list of generated sentences.

    Args:
        sentences (List[str]): A list of generated text strings.
        n (int): The n-gram size (e.g. 1 for Distinct-1, 2 for Distinct-2).

    Returns:
        float: The distinct-n score (number of unique n-grams / total n-grams).
    """
    all_ngrams = []
    for sentence in sentences:
        tokens = sentence.strip().split()
        ngrams = zip(*[tokens[i:] for i in range(n)])
        all_ngrams.extend(ngrams)

    total_ngrams = len(all_ngrams)
    unique_ngrams = len(set(all_ngrams))

    if total_ngrams == 0:
        return 0.0
    return unique_ngrams / total_ngrams


def calculate_token_diversity(directory_path: str, column_name: str, n: int = 2) -> None:
    results = {}

    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            try:
                df = pd.read_csv(file_path)
                if column_name in df.columns:
                    sentences = df[column_name].dropna().astype(str).tolist()
                    score = distinct_n(sentences, n=n)
                    results[filename] = {"distinct_n": score}
                else:
                    results[filename] = {"error": f"Column '{column_name}' not found"}
            except Exception as e:
                results[filename] = {"error": str(e)}

    # Save all results to a JSON file
    output_path = os.path.join(directory_path, "distinct_n_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_path}")


calculate_token_diversity("./results/datasets", column_name="gen_answer", n=2)
