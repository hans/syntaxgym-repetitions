from argparse import ArgumentParser
from copy import deepcopy
import itertools
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

def get_grammatical_sentences(suite: datasets.Dataset,
                              grammatical_conditions: List[str],
                              ) -> Tuple[np.ndarray,
                                         List[Tuple[ItemNumber, ConditionName, str]],
                                         np.ndarray]:
    condition_names = suite[0]["conditions"]["condition_name"]

    # Retrieve all grammatical sentences that could participate in prefix.
    grammatical_sentences = []
    grammatical_sentence_item_numbers = []
    for item in suite:
        sentence_map = dict(zip(item["conditions"]["condition_name"],
                                item["conditions"]["content"]))
        for cond in grammatical_conditions:
            grammatical_sentences.append((item["item_number"], cond, sentence_map[cond]))
            grammatical_sentence_item_numbers.append(item["item_number"])

    grammatical_sentence_item_numbers = np.array(grammatical_sentence_item_numbers)
    grammatical_sentence_lengths = np.array([
        sentence.count(" ") + 1 for _, _, sentence in grammatical_sentences])

    return (grammatical_sentence_item_numbers,
            grammatical_sentences,
            grammatical_sentence_lengths)


def expand_suite(suite: datasets.Dataset,
                 target_length: int,
                 grammatical_conditions: List[str],
                 ungrammatical_conditions: List[str],
                 other_suite: Optional[datasets.Dataset] = None,
                 other_grammatical_conditions: Optional[List[str]] = None,
                 target_size=1000):
    """
    Expand the examples in an item to contain prefixes consisting of grammatical sentences
    from other items from `other_suite`, or the same suite if `other_suite` is not specified.
    Expand until all inputs under `target_length` tokens (based on whitespace split) are
    generated.

    Adds a single condition to the item storing the entire prefix content.
    """

    if other_suite is None:
        expanding_self = True
        other_suite = suite
        other_grammatical_conditions = grammatical_conditions
    else:
        expanding_self = False
        assert other_grammatical_conditions is not None, "Must specify grammatical conditions for other suite."
    gr_numbers, gr_sentences, gr_lengths = get_grammatical_sentences(
        other_suite, other_grammatical_conditions)

    # About many prefixes will we need to reach target_length?
    num_prefixes = int(np.ceil(target_length / gr_lengths.mean()))
    max_possible_prefixes = len(other_suite) * 2 - 1
    if num_prefixes > max_possible_prefixes:
        warnings.warn(f"{num_prefixes} prefixes required to reach target length {target_length} "
                      f"but only {max_possible_prefixes} possible prefixes.")
    num_prefixes = min(num_prefixes, max_possible_prefixes)
    num_samples_per_prefix = int(np.ceil(target_size / num_prefixes))

    results = []
    for num_prefixes_i in trange(1, num_prefixes, desc="Prefix lengths"):
        # Sample sentences to use in this prefix.
        item_sample = np.random.choice(
            range(len(other_suite)),
            size=num_samples_per_prefix,
            replace=True)

        # Now sample prefix sentences for each item.
        for item_idx in item_sample:
            if expanding_self:
                # Make sure we don't draw prefix sentences from the current item.
                possible_prefixes_mask = gr_numbers != item_idx
                if possible_prefixes_mask.sum() < num_prefixes_i:
                    raise RuntimeError(f"Not enough unique prefix items to make {num_prefixes_i} prefixes.")
                candidate_prefix_items = np.where(possible_prefixes_mask)[0]
            else:
                candidate_prefix_items = np.arange(len(gr_sentences))

            prefixes = np.random.choice(candidate_prefix_items, size=num_prefixes_i, replace=False)
            results.append((prefixes, int(item_idx)))

    #########

    # Update existing dataset with null data.
    suite = suite.add_column("prefix_length", [0] * len(suite))
    suite = suite.add_column("used_item_numbers", [[item["item_number"]] for item in suite])
    suite = suite.add_column("used_conditions", [[""] for item in suite])

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
    for prefixes, item_idx in tqdm(results, desc="Generating items"):
        item = deepcopy(suite[item_idx])
        item["item_number"] = acc + 1
        acc += 1

        prefix_str = ". ".join([gr_sentences[idx][2] for idx in prefixes]) + "."
        for cond_regions in item["conditions"]["regions"]:
            cond_regions["content"][0] = prefix_str

        item["prefix_length"] = len(prefix_str.split(" "))
        item["used_item_numbers"] = gr_numbers[prefixes]
        item["used_conditions"] = [gr_sentences[idx][1] for idx in prefixes]

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
    prefix_suite = None if args.prefix_suite is None else \
        datasets.load_dataset("cpllab/syntaxgym", args.prefix_suite)["test"]

    # TODO generalize
    grammatical_conditions = {
        "number": ["match_sing", "match_plural"],
        "reflexive": ["match_sing", "match_plural"],
        "subordination": ["no-sub_no-matrix", "sub_matrix"],
        "mvrr": ["reduced_ambig", "unreduced_ambig", "reduced_unambig", "unreduced_unambig"],  # all are grammatical!
        "fgd": ["that_nogap", "what_gap"],
    }
    grammatical_conditions_suite = grammatical_conditions[args.suite.split("_")[0]]
    grammatical_conditions_prefix_suite = None if args.prefix_suite is None else \
        grammatical_conditions[args.prefix_suite.split("_")[0]]

    expanded = expand_suite(suite, args.target_length, grammatical_conditions_suite, None,
                            other_suite=prefix_suite, other_grammatical_conditions=grammatical_conditions_prefix_suite,
                            target_size=args.target_size)

    # The input to the metric needs to match the expected feature spec.
    expanded_input = expanded.map(remove_columns=["used_item_numbers", "used_conditions", "prefix_length"])

    metric = evaluate.load("cpllab/syntaxgym")
    result = metric.compute(dataset=expanded_input, model_id=args.model_id,
                            batch_size=32)[args.suite]

    # # DEV
    # from collections import namedtuple
    # result_cls = namedtuple("result", ["suite_name", "prediction_results", "region_totals"])
    # result = result_cls(args.suite,
    #     [np.random.random() > 0.5 for _ in range(len(expanded))],
    #     [{(cond, region_number): np.random.random()
    #       for cond in item["conditions"]["condition_name"]
    #       for region_number in item["conditions"]["regions"][0]["region_number"]}
    #      for item in expanded])

    prefix_suite_name = args.prefix_suite if args.prefix_suite is not None else args.suite

    prediction_df = pd.DataFrame(
        result.prediction_results,
        columns=[f"prediction_{i}" for i in range(len(result.prediction_results[0]))])
    prediction_df["used_item_numbers"] = [" ".join(map(str, nums)) for nums in expanded["used_item_numbers"]]
    prediction_df["used_conditions"] = [" ".join(conds) for conds in expanded["used_conditions"]]
    prediction_df["prefix_length"] = expanded["prefix_length"]
    prediction_df["prefix_suite"] = prefix_suite_name
    prediction_df.index = expanded["item_number"]
    prediction_df.index.name = "item_number"
    prediction_df.to_csv(f"{args.output_file}.predictions.csv")

    regions_df = pd.DataFrame(result.region_totals)
    regions_df.index = expanded["item_number"]
    regions_df.index.name = "item_number"
    regions_df = regions_df.reset_index().melt(id_vars=["item_number"])
    regions_df["condition"], regions_df["region_number"] = regions_df["variable"].str
    regions_df["prefix_suite"] = prefix_suite_name
    regions_df.drop("variable", axis=1, inplace=True)
    regions_df.to_csv(f"{args.output_file}.regions.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--suite", default="number_prep")
    parser.add_argument("--prefix_suite")
    parser.add_argument("--target-length", type=int, default=40)
    parser.add_argument("--target-size", type=int, default=1000)
    parser.add_argument("-m", "--model-id", default="gpt2")
    parser.add_argument("-o", "--output-file", required=True)
    parser.add_argument("-d", "--device", default="gpu" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    main(args)
