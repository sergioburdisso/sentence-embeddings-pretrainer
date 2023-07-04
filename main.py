"""
Script to train sentence-BERT models with the provided trainset(s) and
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
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import models, losses, util, datasets
from sentence_transformers import SentenceTransformer, LoggingHandler, InputExample
from sentence_transformers.evaluation import SimilarityFunction, EmbeddingSimilarityEvaluator

from similarity_datasets import SimilarityDataset, SimilarityDatasetContrastive


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
loss = ["multi-neg-ranking", "cosine-similarity"]  # ['softmax', 'multi-neg-ranking', 'cosine-similarity']  # multi-neg-ranking only positive pairs or positive pair + strong negative.
# loss = "cosine-similarity"  # ['softmax', 'multi-neg-ranking', 'cosine-similarity']  # multi-neg-ranking only positive pairs or positive pair + strong negative.
batch_size = 16
num_epochs = 5
evals_per_epoch = 50
warmup_pct = 0.1
learning_rate = 2e-5
log_interval = 100
optimizer = torch.optim.AdamW
# trainset = 'data/stsbenchmark.tsv.gz'
trainset = ['data/AllNLI.tsv.gz', 'data/stsbenchmark.tsv.gz']
evalset = 'data/stsbenchmark.tsv.gz'
output_path = "output"

torch.manual_seed(DEFAULT_SEED)
# np.random.seed(DEFAULT_SEED)
random.seed(DEFAULT_SEED)

if isinstance(trainset, str):
    trainset = [trainset]
if isinstance(loss, str):
    loss = [loss]


def get_dataset_name(paths) -> str:
    """Given a path return the file name."""
    if isinstance(paths, str):
        paths = [paths]

    filenames = [os.path.split(path)[1] for path in paths]
    return '|'.join([filename[:filename.find('.')] for filename in filenames])


def get_study_name(trainset:str, evalset:str, model_name:str, pooling_mode:str, loss:str) -> str:
    """Given the provided parameters return a string to identify the study."""
    return f"train[{get_dataset_name(trainset)}]eval[{get_dataset_name(evalset)}]" + model_name.replace("/", "-") + f"[pooling-{pooling_mode}][loss-{'|'.join(loss)}]"


def on_evaluation(score, epoch, steps):
    # score is Spearman coorelation between predicted cosine-similarity and ground truth values

    # if it's the evaluation perform automatically after finishing the epoch, use "custom epoch" step
    if steps == -1:
        wandb.log({"epoch_score": score, "epoch": epoch + 1})
    else:  # if not just use default wandb steps
        wandb.log({"score": score})


wandb.init(
    project=get_study_name(trainset, evalset, model_name, pooling_mode, loss),  # maybe only evaluation set? so all runs are different models/configs evaluated on the same dataset
    config={
        "learning_rate": learning_rate,
        "loss": loss,
        "model": model_name,
        "pooling_mode": pooling_mode,
        "trainset": trainset,
        "devset": evalset,
        "warmup_data_percentage": warmup_pct,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "optimizer": str(optimizer)
    }
)
wandb.define_metric("epoch")
wandb.define_metric("epoch_score", step_metric="epoch")

output_path = os.path.join(output_path, get_study_name(trainset, evalset, model_name, pooling_mode, loss) + '-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

transformer_seq_encoder = models.Transformer(model_name)

# TODO: in case we need to add new special tokens
# tokens = ["[USR]", "[SYS]"]
# transformer_seq_encoder.tokenizer.add_tokens(tokens, special_tokens=True)
# transformer_seq_encoder.auto_model.resize_token_embeddings(len(transformer_seq_encoder.tokenizer))

sentence_vector = models.Pooling(transformer_seq_encoder.get_word_embedding_dimension(), pooling_mode=pooling_mode)
model = SentenceTransformer(modules=[transformer_seq_encoder, sentence_vector])
wandb.watch(model, log_freq=100)

logging.info(f"Reading training set ({trainset})")

train_objectives = []
for ix, path in enumerate(trainset):
    loss_name = loss[:ix + 1][-1]
    if loss_name == "softmax":
        data = SimilarityDataset(path, use_split="train", delimiter="\t")
        num_labels = data.num_labels
        data = DataLoader(data, shuffle=True, batch_size=batch_size)
        loss_fn = losses.SoftmaxLoss(model=model,
                                     sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                     num_labels=num_labels)
    elif loss_name == "multi-neg-ranking":
        data = datasets.NoDuplicatesDataLoader(
            SimilarityDatasetContrastive(path, use_split="train", delimiter="\t", label_pos="entailment", label_neg="contradiction"),
            batch_size=batch_size
        )
        loss_fn = losses.MultipleNegativesRankingLoss(model=model)
    elif loss_name == "cosine-similarity":
        data = DataLoader(SimilarityDataset(path, col_label="score", use_split="train", is_regression=True, delimiter="\t"),
                        shuffle=True,
                        batch_size=batch_size)
        loss_fn = losses.CosineSimilarityLoss(model=model)
    else:
        raise ValueError(f"Loss {loss_name} not supported.")
    
    train_objectives.append((data, loss_fn))


dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    SimilarityDataset(evalset, col_label="score", use_split="dev", is_regression=True, delimiter="\t"),
    main_similarity=SimilarityFunction.COSINE, batch_size=batch_size, name='devset'
)

steps_per_epoch = min([len(trainset) for trainset, _ in train_objectives])  # trainingsets will be repeated as in a round-robin queue, 1 epochs = full smallest dataset, increase epoch to cover more parts of the bigger ones
warmup_steps = math.ceil(len(data) * num_epochs * warmup_pct)
logging.info("Warmup steps: {}".format(warmup_steps))


# Train the model
model.fit(train_objectives=train_objectives,
          evaluator=dev_evaluator,
          epochs=num_epochs,
          steps_per_epoch=steps_per_epoch,
          evaluation_steps=max(steps_per_epoch // evals_per_epoch, MIN_EVALUATION_STEP),
          warmup_steps=warmup_steps,
          output_path=output_path,
          optimizer_class=optimizer,
          optimizer_params={'lr': learning_rate},
        #   use_amp=False,          #True, if your GPU supports FP16 operations
          callback=on_evaluation)

model = SentenceTransformer(output_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    SimilarityDataset(evalset, col_label="score", use_split="test", is_regression=True, delimiter="\t"),
    batch_size=batch_size, name='testset')

test_evaluator(model, output_path=output_path)
