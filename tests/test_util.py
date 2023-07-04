import pytest
import random

from similarity_datasets import SimilarityDataset, SimilarityDatasetContrastive

PATH_DATASET_LABEL = "tests/data/dataset_labels.csv"

random.seed(13)

def test_dataset_raw():
    with pytest.raises(KeyError):
        data = SimilarityDataset(PATH_DATASET_LABEL)
    with pytest.raises(KeyError):
        data = SimilarityDataset(PATH_DATASET_LABEL, delimiter='\t')

    data = SimilarityDataset(PATH_DATASET_LABEL, col_sent0="sent1", col_sent1="sent2")

    assert data[0].texts == ["hello", "world"]
    assert data.ix2label[data[0].label] == "positive"

def test_dataset_contrastive():
    # with pytest.raises(KeyError):
    #     data = SimilarityDatasetContrastive(PATH_DATASET_LABEL)
    with pytest.raises(KeyError):
        data = SimilarityDatasetContrastive(PATH_DATASET_LABEL, delimiter='\t')

    # positive and explicit negatives
    data = SimilarityDatasetContrastive(PATH_DATASET_LABEL,
                                      col_sent0="sent1", col_sent1="sent2", col_label="value",
                                      label_pos="positive", label_neg="negative")

    assert len(data) == 6
    assert data[0].texts == ["hello", "world", "moon"]
    assert data[1].texts == ["world", "hello", "moon"]
    assert data[2].texts == ["apple", "orange", "car"]
    assert data[3].texts == ["orange", "apple", "car"]
    assert data[4].texts == ["orange", "apple", "truck"]
    assert data[5].texts == ["apple", "orange", "truck"]

    # only positive pairs (non-existing label colum) other pairs will be considered as negative (inside the batch)
    data = SimilarityDatasetContrastive(PATH_DATASET_LABEL,
                                      col_sent0="sent1", col_sent1="sent2",
                                      col_label=None)
    assert len(data) == 7
    assert data[0].texts == ["hello", "world"]
    assert data[1].texts == ["hello", "moon"]
    assert data[2].texts == ["apple", "orange"]
    assert data[3].texts == ["apple", "car"]
    assert data[4].texts == ["orange", "truck"]
    assert data[5].texts == ["person", "dog"]
    assert data[6].texts == ["jsalt", "ai"]

    data = SimilarityDatasetContrastive(PATH_DATASET_LABEL,
                                      col_sent0="sent1", col_sent1="sent2",
                                      col_label="asd")
    assert len(data) == 7
