"""Acquisition asserts (CLAUDE.md §2.5). Run first; fails loudly on a wrong ID/schema."""
from datasets import load_dataset

import config

qqp = load_dataset(*config.QQP)
assert set(qqp.keys()) == {"train", "validation", "test"}
assert qqp["train"].num_rows == 363_846
assert qqp["validation"].num_rows == 40_430
assert set(qqp["validation"]["label"]) == {0, 1}  # test split is all -1
assert set(qqp["validation"].column_names) >= {"question1", "question2", "label", "idx"}

maha = load_dataset(config.MAHA)
assert maha["train"].column_names == ["Sentence1", "Sentence2", "Label"]
assert set(maha["train"]["Label"]) == {"P", "NP"}
assert sum(maha[s].num_rows for s in maha) == 7_999
print("acquisition OK")
