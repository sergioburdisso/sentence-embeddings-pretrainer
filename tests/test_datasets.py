import pytest
import random

from spretrainer.datasets import SimilarityDataReader, SimilarityDataset, SimilarityDatasetContrastive

PATH_DATASET_LABEL = "tests/data/dataset_labels.csv"
PATH_DATASET_LABEL_SPLITS = "tests/data/dataset_labels_splits.csv"
PATH_DATASET_REGRESSION = "tests/data/dataset_regression.csv"

random.seed(13)


def test_dataset_raw():
    # Error cases
    with pytest.raises(KeyError):
        data = SimilarityDataset(
            SimilarityDataReader.read_csv(PATH_DATASET_LABEL, col_sent0="sentence1", col_sent1="sentence2")
        )
    with pytest.raises(KeyError):
        data = SimilarityDataset(
            SimilarityDataReader.read_csv(PATH_DATASET_LABEL, delimiter='\t')
        )

    # Load while dataset as is
    data = SimilarityDataset(
        SimilarityDataReader.read_csv(PATH_DATASET_LABEL, col_sent0="sent1", col_sent1="sent2")
    )
    assert len(data) == 7
    assert data[0].texts == ["hello", "world"]
    assert data.ix2label[data[0].label] == "positive"
    assert data[-1].texts == ["jsalt", "ai"]
    assert data.ix2label[data[-1].label] == "positive"

    # Load only a subset, the data for a given split ("train")
    data = SimilarityDataset(
        SimilarityDataReader.read_csv(PATH_DATASET_LABEL_SPLITS,
                                      col_sent0="sent1", col_sent1="sent2",
                                      col_split="split", use_split="train")
    )
    assert len(data) == 4
    assert data[-1].texts == ["apple", "car"]
    assert data.ix2label[data[-1].label] == "negative"

    # Load regression data (i.e. ground truth is a number) with normalized values (default)
    data = SimilarityDataset(
        SimilarityDataReader.read_csv(PATH_DATASET_REGRESSION, col_sent0="sent1", col_sent1="sent2"),
        is_regression=True
    )
    assert len(data) == 7
    assert data[0].texts == ["hello", "world"]
    # check that values are returned normalized
    assert data[0].label == 5 / 5
    assert data[-1].label == 3 / 5

    # Load regression data without normalizing values (raw values)
    data = SimilarityDataset(
        SimilarityDataReader.read_csv(PATH_DATASET_REGRESSION, col_sent0="sent1", col_sent1="sent2"),
        is_regression=True, normalize_value=False
    )
    assert len(data) == 7
    # check that values are returned as given
    assert data[0].label == 5
    assert data[-1].label == 3
    assert isinstance(data[-1].label, float)


def test_dataset_contrastive():
    # with pytest.raises(KeyError):
    #     data = SimilarityDatasetContrastive(PATH_DATASET_LABEL)
    with pytest.raises(KeyError):
        data = SimilarityDatasetContrastive(
            SimilarityDataReader.read_csv(PATH_DATASET_LABEL, delimiter='\t')
        )

    # positive and explicit negatives
    data = SimilarityDatasetContrastive(
        SimilarityDataReader.read_csv(PATH_DATASET_LABEL, col_sent0="sent1", col_sent1="sent2", col_label="value"),
        label_pos="positive", label_neg="negative"
    )

    assert len(data) == 6
    assert data[0].texts == ["hello", "world", "moon"]
    assert data[1].texts == ["world", "hello", "moon"]
    assert data[2].texts == ["apple", "orange", "car"]
    assert data[3].texts == ["orange", "apple", "car"]
    assert data[4].texts == ["orange", "apple", "truck"]
    assert data[5].texts == ["apple", "orange", "truck"]

    # only positive pairs (non-existing label colum) other pairs will be considered as negative (inside the batch)
    data = SimilarityDatasetContrastive(
        SimilarityDataReader.read_csv(PATH_DATASET_LABEL, col_sent0="sent1", col_sent1="sent2", col_label="value")
    )
    assert len(data) == 7
    assert data[0].texts == ["hello", "world"]
    assert data[1].texts == ["hello", "moon"]
    assert data[2].texts == ["apple", "orange"]
    assert data[3].texts == ["apple", "car"]
    assert data[4].texts == ["orange", "truck"]
    assert data[5].texts == ["person", "dog"]
    assert data[6].texts == ["jsalt", "ai"]

    data = SimilarityDatasetContrastive(
        SimilarityDataReader.read_csv(PATH_DATASET_LABEL, col_sent0="sent1", col_sent1="sent2", col_label=None)
    )
    assert len(data) == 7
