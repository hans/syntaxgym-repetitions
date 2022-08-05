from argparse import ArgumentParser
from copy import deepcopy
import itertools
import re

import datasets
import evaluate
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def regions_to_string(regions):
    ret = " ".join([region.lstrip()
                    for region in regions["content"]
                    if region.strip() != ""])
    ret = re.sub(r"\s+([.,])", r"\1", ret)

    return ret


def expand_suite(suite: datasets.Dataset, max_length,
                 grammatical_conditions, ungrammatical_conditions,
                 target_size=10000, subsample_pct=None):
    """
    Expand the examples in an item to contain prefixes consisting of grammatical sentences
    from other items. Expand until all inputs under `max_length` tokens (based on
    whitespace split) are generated.

    Adds a single condition to the item storing the entire prefix content.
    """

    condition_names = suite[0]["conditions"]["condition_name"]

    # Retrieve all grammatical sentences that could participate in prefix.
    grammatical_sentences = []
    for item in suite:
        sentence_map = dict(zip(item["conditions"]["condition_name"],
                                item["conditions"]["content"]))
        for cond in grammatical_conditions:
            grammatical_sentences.append((item["item_number"], sentence_map[cond]))

    grammatical_sentence_lengths = np.array([sentence.count(" ") + 1 for _, sentence in grammatical_sentences])

    # Add empty prefix region to each item.
    items = []
    for item in suite:
        item = deepcopy(item)
        item["conditions"]["content"].insert(0, "")
        for region in item["conditions"]["regions"]:
            region["region_number"] = [1] + [x + 1 for x in region["region_number"]]
            region["content"].insert(0, "")

        # Also prepare to track accumulated items in the prefix.
        item["used_item_numbers"] = [item["item_number"]]

        # Update predictions to account for shifted region number.
        item["predictions"] = [
            re.sub(r"(\d+)", lambda match: str(int(match.group(1)) + 1), prediction_formula)
            for prediction_formula in item["predictions"]
        ]

        items.append(item)

    from tqdm.auto import tqdm
    new_datasets = []
    acc_dataset_size = len(suite)
    while acc_dataset_size < target_size and len(items) > 0:
        items_next = []

        for item in tqdm(items):
            item_sentence_maxlen = max(content_str.count(" ") + 1
                for content_str in item["conditions"]["content"])

            compatible_prefixes = []
            for (prefix_item_idx, prefix_sentence), prefix_length in zip(grammatical_sentences, grammatical_sentence_lengths):
                if prefix_item_idx in item["used_item_numbers"]:
                    continue
                if prefix_length + item_sentence_maxlen > max_length:
                    continue
                if subsample_pct is not None and np.random.random() >= subsample_pct:
                    continue
                compatible_prefixes.append(prefix_sentence)

            for prefix_sentence in compatible_prefixes:
                # Add prefix sentence to item.
                ret = deepcopy(item)

                for cond_name, cond_regions in zip(condition_names, ret["conditions"]["regions"]):
                    additional_prefix = prefix_sentence + ". " if cond_regions["content"][0] != "" else prefix_sentence + "."
                    cond_regions["content"][0] = additional_prefix + cond_regions["content"][0]

                ret["conditions"]["content"] = [regions_to_string(regions)
                        for regions in ret["conditions"]["regions"]]
                ret["used_item_numbers"].append(prefix_item_idx)

                items_next.append(ret)

            # Early exit
            if acc_dataset_size + len(items_next) > target_size:
                break

        new_dataset = datasets.Dataset.from_dict({
            feature: [item[feature] for item in items_next]
            for feature in suite.features
        }, info=suite.info, split=suite.split)

        new_datasets.append(new_dataset)
        acc_dataset_size += len(new_dataset)

        items = items_next

    # Update item numbers to be unique.
    ret_dataset = datasets.concatenate_datasets([suite] + new_datasets)
    for idx, item in enumerate(ret_dataset):
        item["item_number"] = idx + 1

    return ret_dataset


def main(args):
    suite = datasets.load_dataset("cpllab/syntaxgym", args.suite)["test"]

    # TODO generalize
    grammatical_conditions = ["match_sing", "match_plural"]
    ungrammatical_conditions = ["mismatch_sing", "mismatch_plural"]

    expanded = expand_suite(suite, args.max_length, grammatical_conditions, ungrammatical_conditions,
                            target_size=args.target_size, subsample_pct=args.subsample_pct)

    metric = evaluate.load("cpllab/syntaxgym")
    result = metric.compute(dataset=expanded, model_id=args.model_id)[args.suite]

    prediction_df = pd.DataFrame(result.prediction_results)
    prediction_df.index = expanded["item_number"]
    prediction_df.index.name = "item_number"
    prediction_df.to_csv(args.output_file + ".predictions.csv")

    regions_df = pd.DataFrame(result.region_totals)
    regions_df.index = expanded["item_number"]
    regions_df.index.name = "item_number"
    regions_df = regions_df.reset_index().melt(id_vars=["item_number"])
    regions_df["condition"], regions_df["region_number"] = regions_df["variable"].str
    regions_df.drop("variable", axis=1, inplace=True)
    regions_df.to_csv(args.output_file + ".regions.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--suite", default="number_prep")
    parser.add_argument("--max-length", type=int, default=40)
    parser.add_argument("--target-size", type=int, default=1000)
    parser.add_argument("--subsample-pct", type=float, default=None)
    parser.add_argument("-m", "--model-id", default="gpt2")
    parser.add_argument("-o", "--output-file", required=True)
    args = parser.parse_args()

    main(args)
