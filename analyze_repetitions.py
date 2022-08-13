from argparse import ArgumentParser
from copy import deepcopy
import itertools
import re
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


def expand_suite(suite: datasets.Dataset, target_length,
                 grammatical_conditions, ungrammatical_conditions,
                 target_size=1000):
    """
    Expand the examples in an item to contain prefixes consisting of grammatical sentences
    from other items. Expand until all inputs under `max_length` tokens (based on
    whitespace split) are generated.

    Adds a single condition to the item storing the entire prefix content.
    """

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

    # About many prefixes will we need to reach max_length?
    num_prefixes = int(np.ceil(target_length / grammatical_sentence_lengths.mean()))
    max_possible_prefixes = len(suite) * 2 - 1
    if num_prefixes > max_possible_prefixes:
        warnings.warn(f"{num_prefixes} prefixes required to reach target length {target_length} "
                      f"but only {max_possible_prefixes} possible prefixes.")
    num_prefixes = min(num_prefixes, max_possible_prefixes)
    num_samples_per_prefix = int(np.ceil(target_size / num_prefixes))

    results = []
    for num_prefixes_i in trange(num_prefixes, desc="Prefix lengths"):
        # Sample sentences to use in this prefix.
        item_sample = np.random.choice(
            range(len(suite)),
            size=num_samples_per_prefix,
            replace=True)

        # Now sample prefix sentences for each item.
        for item_idx in item_sample:
            # We can use prefix sentences from any other item.
            possible_prefixes_mask = grammatical_sentence_item_numbers != item_idx
            if possible_prefixes_mask.sum() < num_prefixes_i:
                raise RuntimeError(f"Not enough unique prefix items to make {num_prefixes_i} prefixes.")

            prefixes = np.random.choice(np.where(possible_prefixes_mask)[0], size=num_prefixes_i, replace=False)
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

        return item

    suite = suite.map(add_empty_region)

    ########

    acc = len(suite)
    new_items = []
    for prefixes, item_idx in tqdm(results, desc="Generating items"):
        item = deepcopy(suite[item_idx])
        item["item_number"] = acc + 1
        acc += 1

        item["prefix_length"] = len(prefixes)
        item["used_item_numbers"] = grammatical_sentence_item_numbers[prefixes]
        item["used_conditions"] = [grammatical_sentences[idx][1] for idx in prefixes]

        prefix_str = ". ".join([grammatical_sentences[idx][2] for idx in prefixes]) + "."
        for cond_name, cond_regions in zip(condition_names, item["conditions"]["regions"]):
            cond_regions["content"][0] = prefix_str

        item["conditions"]["content"] = [
            regions_to_string(regions) for regions in item["conditions"]["regions"]]

        new_items.append(item)

    # new_datasets = []
    # acc_dataset_size = len(suite)
    # items = list(suite)
    # longest_sentence = 0
    # # while acc_dataset_size < target_size and len(items) > 0:
    # while longest_sentence < max_length and len(items) > 0:
    #     items_next = []

    #     for item in tqdm(items):
    #         item_sentence_maxlen = max(content_str.count(" ") + 1
    #             for content_str in item["conditions"]["content"])
    #         longest_sentence = max(longest_sentence, item_sentence_maxlen)

    #         compatible_prefixes = []
    #         for (prefix_item_idx, prefix_cond, prefix_sentence), prefix_length in zip(grammatical_sentences, grammatical_sentence_lengths):
    #             if prefix_item_idx in item["used_item_numbers"]:
    #                 continue
    #             if prefix_length + item_sentence_maxlen > max_length:
    #                 continue
    #             if subsample_pct is not None and np.random.random() >= subsample_pct:
    #                 continue
    #             compatible_prefixes.append((prefix_item_idx, prefix_cond, prefix_sentence, prefix_length))

    #         for prefix_item_idx, prefix_cond, prefix_sentence, prefix_length in compatible_prefixes:
    #             # Add prefix sentence to item.
    #             ret = deepcopy(item)
    #             ret["item_number"] = acc_dataset_size + 1

    #             for cond_name, cond_regions in zip(condition_names, ret["conditions"]["regions"]):
    #                 additional_prefix = prefix_sentence + ". " if cond_regions["content"][0] != "" else prefix_sentence + "."
    #                 cond_regions["content"][0] = additional_prefix + cond_regions["content"][0]

    #             ret["conditions"]["content"] = [regions_to_string(regions)
    #                     for regions in ret["conditions"]["regions"]]
    #             ret["used_item_numbers"].append(prefix_item_idx)
    #             ret["used_conditions"].append(prefix_cond)
    #             ret["prefix_length"] += prefix_length

    #             acc_dataset_size += 1
    #             items_next.append(ret)

    #         # # Early exit
    #         # if acc_dataset_size + len(items_next) > target_size:
    #         #     break

    #     new_dataset_dict = {
    #         feature: [item[feature] for item in items_next]
    #         for feature in suite.features
    #     }
    #     new_dataset = datasets.Dataset.from_dict(new_dataset_dict,
    #         info=suite.info,
    #         split=suite.split)

    #     new_datasets.append(new_dataset)
    #     acc_dataset_size += len(new_dataset)

    #     items = items_next

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
    grammatical_conditions = ["match_sing", "match_plural"]
    ungrammatical_conditions = ["mismatch_sing", "mismatch_plural"]

    expanded = expand_suite(suite, args.max_length, grammatical_conditions, ungrammatical_conditions,
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

    prediction_df = pd.DataFrame(
        result.prediction_results,
        columns=[f"prediction_{i}" for i in range(len(result.prediction_results[0]))])
    prediction_df["used_item_numbers"] = [" ".join(map(str, nums)) for nums in expanded["used_item_numbers"]]
    prediction_df["used_conditions"] = [" ".join(conds) for conds in expanded["used_conditions"]]
    prediction_df["prefix_length"] = expanded["prefix_length"]
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
    parser.add_argument("--target-length", type=int, default=40)
    parser.add_argument("--target-size", type=int, default=1000)
    parser.add_argument("-m", "--model-id", default="gpt2")
    parser.add_argument("-o", "--output-file", required=True)
    parser.add_argument("-d", "--device", default="gpu" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    main(args)
