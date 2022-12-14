"""
Run SyntaxGym evaluation with arbitrary prefix data.
"""


from argparse import ArgumentParser
from copy import deepcopy
import json
from pathlib import Path
import re
from readline import get_history_length
from typing import Optional, List, Tuple, Dict, Any
import warnings

import datasets
import evaluate
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm, trange


whitespace_punct_re = re.compile(r"\s+([.,])")
def regions_to_string(regions):
    ret = " ".join([region.lstrip()
                    for region in regions["content"]
                    if region.strip() != ""])
    ret = whitespace_punct_re.sub(r"\1", ret)

    return ret


ItemNumber = int
ConditionName = str


def expand_suite(suite: datasets.Dataset,
                 prefixes: Dict[int, List[str]],
                 target_lengths: List[int],
                 target_size=1000):
    """
    Expand the examples in an item to contain prefixes consisting of prefix strings from
    `prefix_data` (which are grouped by approximate length in tokens).
    Expand until all inputs under `target_length` tokens (based on whitespace split) are
    generated.)

    Adds a single condition to the item storing the entire prefix content.
    """

    size_per_length = target_size // len(target_lengths)

    results: List[Tuple[int, int, int]] = []

    for target_length in target_lengths:
        if size_per_length > len(prefixes[target_length]):
            raise ValueError(f"Not enough prefixes at length {target_length} "
                             f"({len(prefixes[target_length])}) to generate "
                             f"unique items (target size is {size_per_length}). "
                              "Reduce target size or add prefixes.")

        # Sample items to use
        item_sample = np.random.choice(
            range(len(suite)),
            size=size_per_length,
            replace=True)

        # Sample prefixes to use
        prefix_sample = np.random.choice(len(prefixes[target_length]),
                                         size=size_per_length,
                                         replace=False)

        results.extend(list(zip(np.repeat(target_length, len(prefix_sample)),
                                prefix_sample.tolist(), item_sample.tolist())))

    #########

    # Update existing dataset with null data.
    suite = suite.add_column("prefix_length", [0] * len(suite))
    suite = suite.add_column("prefix_source", [""] * len(suite))

    def add_empty_region(item):
        """
        Add an empty prefix region to the given item.
        """
        for region in item["conditions"]["regions"]:
            region["region_number"] = [1] + [x + 1 for x in region["region_number"]]
            region["content"].insert(0, "")

        # Update predictions to account for shifted region number.
        item["predictions"] = [
            re.sub(r"(\d+)", lambda match: str(int(match.group(1)) + 1), prediction_formula)
            for prediction_formula in item["predictions"]
        ]

        # Also split conjunction predictions.
        item["predictions"] = [
            subprediction.strip()
            for prediction in item["predictions"]
            for subprediction in prediction.split("&")
        ]

        return item

    suite = suite.map(add_empty_region)

    ########

    acc = len(suite)
    new_items = []
    for prefix_length, prefix_idx, item_idx in tqdm(results, desc="Generating items"):
        item = deepcopy(suite[item_idx])
        item["item_number"] = acc + 1
        acc += 1

        prefix_str = prefixes[prefix_length][prefix_idx]
        for cond_regions in item["conditions"]["regions"]:
            cond_regions["content"][0] = prefix_str

        item["prefix_length"] = prefix_str.count(" ") + 1
        item["prefix_source"] = f"{prefix_length}/{prefix_idx}"
        item["conditions"]["content"] = [
            regions_to_string(regions) for regions in item["conditions"]["regions"]]

        new_items.append(item)

    new_dataset = datasets.Dataset.from_dict(
        {feature: [item[feature] for item in new_items]
         for feature in suite.features},
        info=suite.info, split=suite.split)
    ret_dataset = datasets.concatenate_datasets([suite, new_dataset])

    return ret_dataset


def main(args):
    print("Running with device: ", args.device)

    suite = datasets.load_dataset("cpllab/syntaxgym", args.suite)["test"]

    # TODO generalize
    suite_conditions = {
        "grammatical": {
            "number": ["match_sing", "match_plural"],
            "reflexive": ["match_sing", "match_plural"],
            "subordination": ["no-sub_no-matrix", "sub_matrix"],
            "mvrr": ["reduced_ambig", "unreduced_ambig", "reduced_unambig", "unreduced_unambig"],  # all are grammatical!
            "npi": ["neg_pos", "neg_neg"],
            "npz": ["no-obj_comma", "obj_no-comma", "no-obj_no-comma", "obj_comma"],  # all are grammatical!
            "fgd": ["that_nogap", "what_gap"],
        },
        "ungrammatical": {
            "number": ["mismatch_sing", "mismatch_plural"],
            "reflexive": ["mismatch_sing", "mismatch_plural"],
            "subordination": ["sub_no-matrix", "no-sub_matrix"],
            "mvrr": [],
            "npi": ["pos_neg", "pos_pos"],
            "npz": [],
            "fgd": ["that_gap", "what_nogap"],
        }
    }

    prefix_label = args.prefix_file.stem if args.prefix_label is None else args.prefix_label
    with args.prefix_file.open() as prefix_f:
        prefixes: Dict[int, List[str]] = {
            int(prefix_length): sentences
            for prefix_length, sentences in json.load(prefix_f).items()
        }
        
        # Fix detokenization issue -- leftover @,@ , @.@, etc.
        # from Moses tokenizer, not caught by detokenizer
        moses_punct_escape_re = re.compile(r"\s+@([^@]+)@\s+")
        prefixes = {
            prefix_length: [moses_punct_escape_re.sub(r"\1", sentence) for sentence in sentences]
            for prefix_length, sentences in prefixes.items()
        }

    target_lengths = [int(x) for x in args.target_lengths.split(",")] \
        if args.target_lengths is not None else prefixes.keys()

    expanded = expand_suite(suite, prefixes, target_lengths=target_lengths,
                            target_size=args.target_size)

    # The input to the metric needs to match the expected feature spec.
    expanded_input = expanded.map(remove_columns=["prefix_length", "prefix_source"])

    metric = evaluate.load("cpllab/syntaxgym")
    result = metric.compute(dataset=expanded_input, model_id=args.model_id,
                            batch_size=32)[args.suite]

    prediction_df = pd.DataFrame(
        result.prediction_results,
        columns=[f"prediction_{i}" for i in range(len(result.prediction_results[0]))])
    prediction_df["prefix_length"] = expanded["prefix_length"]
    prediction_df["prefix_label"] = prefix_label
    prediction_df.index = expanded["item_number"]
    prediction_df.index.name = "item_number"
    prediction_df.to_csv(args.output_dir / "predictions.csv")

    regions_df = pd.DataFrame(result.region_totals)
    regions_df.index = expanded["item_number"]
    regions_df.index.name = "item_number"
    regions_df = regions_df.reset_index().melt(id_vars=["item_number"])
    regions_df["condition"], regions_df["region_number"] = regions_df["variable"].str
    regions_df["prefix_label"] = prefix_label
    regions_df.drop("variable", axis=1, inplace=True)
    regions_df.to_csv(args.output_dir / "regions.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--suite", default="number_prep")
    parser.add_argument("--prefix_file", type=Path, required=True)
    parser.add_argument("--prefix_label")
    parser.add_argument("--target-lengths", type=str, help="comma-separated list of target lengths")
    parser.add_argument("--target-size", type=int, default=1000)
    parser.add_argument("-m", "--model-id", default="gpt2")
    parser.add_argument("-o", "--output_dir", type=Path, required=True)
    parser.add_argument("-d", "--device", default="gpu" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    main(args)
