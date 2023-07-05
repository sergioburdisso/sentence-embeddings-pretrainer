"""
Script to train sentence-BERT models with the provided training set(s) and
loss function(s). At every given number of training steps, the learned
embeddings are evaluated on the provided similarity task.

Usage: python main.py .... [TODO]

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
import os
import sys
import math
import torch
import wandb
import random
import logging
import argparse

from datetime import datetime
from typing import Any, Union, Optional, List
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import models, losses, util, datasets
from sentence_transformers import SentenceTransformer, LoggingHandler, InputExample
from sentence_transformers.evaluation import SimilarityFunction, EmbeddingSimilarityEvaluator

from similarity_datasets import SimilarityDataReader, SimilarityDataset, SimilarityDatasetContrastive


DEFAULT_SEED = 13
MIN_EVALUATION_STEP = 100

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


# parser = argparse.ArgumentParser()
# parser.add_argument("--train_batch_size", default=64, type=int)
# parser.add_argument("--max_seq_length", default=300, type=int)
# parser.add_argument("--model_name", required=True)
# parser.add_argument("--max_passages", default=0, type=int)
# parser.add_argument("--epochs", default=30, type=int)
# parser.add_argument("--pooling", default="mean")
# parser.add_argument("--negs_to_use", default=None, help="From which systems should negatives be used? Multiple systems seperated by comma. None = all")
# parser.add_argument("--warmup_steps", default=1000, type=int)
# parser.add_argument("--lr", default=2e-5, type=float)
# parser.add_argument("--num_negs_per_system", default=5, type=int)
# parser.add_argument("--use_pre_trained_model", default=False, action="store_true")
# parser.add_argument("--use_all_queries", default=False, action="store_true")
# args = parser.parse_args()

# logging.info(str(args))

model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-uncased'
pooling_mode = 'cls'  # ['mean', 'max', 'cls', 'weightedmean', 'lasttoken']
loss = ["denoising-autoencoder", 'multi-neg-ranking', 'cosine-similarity']  # multi-neg-ranking only positive pairs or positive pair + strong negative.
batch_size = 16
num_epochs = 6
evals_per_epoch = 50
warmup_pct = 0.1
learning_rate = 2e-5
log_interval = 100
optimizer = torch.optim.AdamW
path_trainset = ['data/dialogue.txt', 'data/AllNLI_train.csv', 'data/stsbenchmark_train.csv']
path_devset = 'data/stsbenchmark_dev.csv'
path_testset = 'data/stsbenchmark_test.csv'
checkpoint_dir = "output"
checkpoint_name = "checkpoint"
special_tokens = []  # ["[USR]", "[SYS]"]
project_name = None

torch.manual_seed(DEFAULT_SEED)
# np.random.seed(DEFAULT_SEED)
random.seed(DEFAULT_SEED)

if isinstance(path_trainset, str):
    path_trainset = [path_trainset]
if isinstance(loss, str):
    loss = [loss]


def get_dataset_name(paths:Union[List[str], str]) -> str:
    """Given a path, or a list of paths, return string with file name(s)."""
    if isinstance(paths, str):
        paths = [paths]

    filenames = [os.path.split(path)[1] for path in paths]
    return '|'.join([filename[:filename.find('.')] for filename in filenames])


def get_study_name(path_trainset:str, path_devset:str, model_name:str, pooling_mode:str, loss:str) -> str:
    """Given the provided parameters return a string to identify the study."""
    return f"train[{get_dataset_name(path_trainset)}]eval[{get_dataset_name(path_devset)}]" + model_name.replace("/", "-") + f"[pooling-{pooling_mode}][loss-{'|'.join(loss)}]"


def on_evaluation(score:float, epoch:int, steps:int) -> None:
    # score is Spearman coorelation between predicted cosine-similarity and ground truth values

    # if it's the evaluation perform automatically after finishing the epoch, use "custom epoch" step
    if steps == -1:
        wandb.log({"epoch_score": score, "epoch": epoch + 1})
    else:  # if not just use default wandb steps
        wandb.log({"score": score})

project_name = (project_name or get_study_name(path_trainset, path_devset, model_name, pooling_mode, loss))[:128]
wandb.init(
    project=project_name,
    config={
        "learning_rate": learning_rate,
        "loss": loss,
        "model": model_name,
        "pooling_mode": pooling_mode,
        "trainset": path_trainset,
        "devset": path_devset,
        "warmup_data_percentage": warmup_pct,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "optimizer": str(optimizer)
    }
)
wandb.define_metric("epoch")
wandb.define_metric("epoch_score", step_metric="epoch")

path_output = os.path.join(checkpoint_dir, (checkpoint_name or project_name) + '-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

transformer_seq_encoder = models.Transformer(model_name)

if special_tokens:
    transformer_seq_encoder.tokenizer.add_tokens(special_tokens, special_tokens=True)
    transformer_seq_encoder.auto_model.resize_token_embeddings(len(transformer_seq_encoder.tokenizer))

sentence_vector = models.Pooling(transformer_seq_encoder.get_word_embedding_dimension(), pooling_mode=pooling_mode)
model = SentenceTransformer(modules=[transformer_seq_encoder, sentence_vector])
wandb.watch(model, log_freq=100)

logging.info(f"Reading training sets ({path_trainset})")
train_objectives = []
for ix, path in enumerate(path_trainset):
    loss_name = loss[:ix + 1][-1]
    data = SimilarityDataReader.read_csv(path, col_sent0="sent1", col_sent1="sent2", col_label="value")

    if loss_name == "softmax":
        data = SimilarityDataset(data)
        loss_fn = losses.SoftmaxLoss(model=model,
                                     sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                     num_labels=data.num_labels)
    elif loss_name == "multi-neg-ranking":
        data = SimilarityDatasetContrastive(data, label_pos="entailment", label_neg="contradiction")
        loss_fn = losses.MultipleNegativesRankingLoss(model=model)
    elif loss_name == "cosine-similarity":
        data = SimilarityDataset(data, is_regression=True, normalize_value=True)
        loss_fn = losses.CosineSimilarityLoss(model=model)
    elif loss_name == "denoising-autoencoder":  # unsupervised
        sentences = SimilarityDataReader.read_docs(path, lines_are_documents=True)
        data = datasets.DenoisingAutoEncoderDataset(list(sentences))
        loss_fn = losses.DenoisingAutoEncoderLoss(model, tie_encoder_decoder=True)
    else:
        raise ValueError(f"Loss {loss_name} not supported.")

    if loss_name == "multi-neg-ranking":
        data_loader = datasets.NoDuplicatesDataLoader(data, batch_size=batch_size)
    else:
        data_loader = DataLoader(data, shuffle=True, batch_size=batch_size, drop_last=True)

    train_objectives.append((data_loader, loss_fn))

logging.info(f"Reading development set ({path_devset})")
devset = SimilarityDataset(
    SimilarityDataReader.read_csv(path_devset, col_sent0="sent1", col_sent1="sent2", col_label="value"),
    is_regression=True,
    normalize_value=True
)
dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    devset, main_similarity=SimilarityFunction.COSINE, batch_size=batch_size, name='devset'
)

steps_per_epoch = min([len(data_loader) for data_loader, _ in train_objectives])  # trainingsets will be repeated as in a round-robin queue, 1 epochs = full smallest dataset, increase epoch to cover more parts of the bigger ones
warmup_steps = math.ceil(len(data) * num_epochs * warmup_pct)
logging.info("Warmup steps: {}".format(warmup_steps))

model.fit(train_objectives=train_objectives,
          evaluator=dev_evaluator,
          epochs=num_epochs,
          steps_per_epoch=steps_per_epoch,
          evaluation_steps=max(steps_per_epoch // evals_per_epoch, MIN_EVALUATION_STEP),
          warmup_steps=warmup_steps,
          output_path=path_output,
          optimizer_class=optimizer,
          optimizer_params={'lr': learning_rate},
        #   use_amp=False,          # True, if your GPU supports FP16 operations
          callback=on_evaluation)

logging.info(f"Loading best checkpoint from disk ({path_output})")
model = SentenceTransformer(path_output)

logging.info(f"Reading the test set ({path_testset})")
testset = SimilarityDataset(
    SimilarityDataReader.read_csv(path_testset, col_sent0="sent1", col_sent1="sent2", col_label="value"),
    is_regression=True,
    normalize_value=True
)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    testset, main_similarity=SimilarityFunction.COSINE, batch_size=batch_size, name='testset'
)
logging.info(f"Evaluating model on the test set data...")
test_evaluator(model, output_path=path_output)
