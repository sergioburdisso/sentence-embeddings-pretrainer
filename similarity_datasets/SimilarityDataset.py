import csv
import gzip

from torch.utils.data import Dataset
from sentence_transformers import InputExample

class SimilarityDataset(Dataset):
    def __init__(self, path_csv, delimiter=",", col_sent0="sentence1", col_sent1="sentence2",
                 col_label="value", col_split="split", use_split=None,
                 is_regression=False, encoding='utf8'):

        self.samples = []
        self.num_labels = 0
        self.label2ix = {}
        self.ix2label = {}

        f_open = gzip.open if path_csv.endswith(".gz") else open  # TODO: improve "if type == gzip" detection
        with f_open(path_csv, 'rt', encoding=encoding) as reader:
            csv_file = csv.DictReader(reader, delimiter=delimiter, quoting=csv.QUOTE_NONE)

            max_value, min_value = float("-inf"), float("inf")
            for row in csv_file:
                if not use_split or row[col_split] == use_split:
                    label = float(row[col_label]) if is_regression else row[col_label]

                    if is_regression:
                        max_value = max(max_value, label)
                        min_value = min(min_value, label)
                    else:
                        if label not in self.label2ix:
                            self.label2ix[label] = len(self.label2ix)
                            self.ix2label[self.label2ix[label]] = label
                        label = self.label2ix[label]

                    self.samples.append(InputExample(texts=[row[col_sent0], row[col_sent1]], label=label))

        # if regression normalize values between [0, 1]
        if is_regression:
            for sample in self.samples:
                sample.label = (sample.label - min_value) / (max_value - min_value)

        self.num_labels = len(self.label2ix)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

