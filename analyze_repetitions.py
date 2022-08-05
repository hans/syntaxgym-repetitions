from argparse import ArgumentParser
from copy import deepcopy
import itertools
import re

import datasets
from datasets.table import InMemoryTable, concat_tables
import evaluate
import numpy as np
from pkg_resources import compatible_platforms


def regions_to_string(regions):
    ret = " ".join([region.lstrip()
                    for region in regions["content"]
                    if region.strip() != ""])
    ret = re.sub(r"\s+([.,])", r"\1", ret)

    return ret



def add_dataset_batch(dataset: datasets.Dataset,
                      new_items):
    item_table = InMemoryTable.from_pydict({k: [v] for k, v in item.items()})
    # # We don't call _check_if_features_can_be_aligned here so this cast is "unsafe"
    # dset_features, item_features = _align_features([self.features, Features.from_arrow_schema(item_table.schema)])
    # Cast to align the schemas of the tables and concatenate the tables
    table = concat_tables(
        [
            self._data.cast(dset_features.arrow_schema) if self.features != dset_features else self._data,
            item_table.cast(item_features.arrow_schema),
        ]
    )
    if self._indices is None:
        indices_table = None
    else:
        item_indices_array = pa.array([len(self._data)], type=pa.uint64())
        item_indices_table = InMemoryTable.from_arrays([item_indices_array], names=["indices"])
        indices_table = concat_tables([self._indices, item_indices_table])
    info = self.info.copy()
    info.features.update(item_features)
    table = update_metadata_with_features(table, info.features)
    return Dataset(
        table,
        info=info,
        split=self.split,
        indices_table=indices_table,
        fingerprint=new_fingerprint,
    )


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
            region["region_number"].insert(0, 0)
            region["content"].insert(0, "")

        # Also prepare to track accumulated items in the prefix.
        item["used_item_numbers"] = [item["item_number"]]

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

    # TODO fix item numbers

    return datasets.concatenate_datasets([suite] + new_datasets)


def main(args):
    pass