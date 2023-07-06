"""
Script to pre-train sentence-BERT-like models for task-oriented dialogue.

Usage: python main.py .... [TODO]
"""
import os
import sys
import json
import math
import torch
import wandb
import random
import logging
import argparse
import numpy as np

from datetime import datetime

from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from tqdm.autonotebook import trange
from typing import Union, Optional, List, Iterable, Tuple, Type, Dict, Callable
from sentence_transformers import models, losses, datasets
from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers.util import fullname, batch_to_device
from sentence_transformers.evaluation import SentenceEvaluator, SimilarityFunction, EmbeddingSimilarityEvaluator
from sentence_transformers.model_card_templates import ModelCardTemplate

from similarity_evaluation import ClassificationEvaluator, LossEvaluator
from similarity_datasets import SimilarityDataReader, SimilarityDataset, SimilarityDatasetContrastive


# TODO: use config file instead for below
DEFAULT_SEED = 13
DEFAULT_NAME_COL_SENT1 = "sent1"
DEFAULT_NAME_COL_SENT2 = "sent2"
DEFAULT_NAME_COL_LABEL = "value"
MIN_EVALUATION_STEPS = 100
MIN_CHECKPOINT_SAVE_STEPS = 50
WANDB_LOG_FREQ = 100

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


# TODO: use argparse.ArgumentParser() below!!!
model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-uncased'
pooling_mode = 'cls'  # ['mean', 'max', 'cls', 'weightedmean', 'lasttoken']
# loss = "softmax"
# eval_metric = "f1-score"
# path_trainset = 'data/AllNLI_train.csv'
# path_devset = 'data/AllNLI_dev.csv'
# path_testset = 'data/AllNLI_test.csv'

loss = 'multi-neg-ranking'  # softmax, "denoising-autoencoder", 'multi-neg-ranking', 'cosine-similarity', multi-neg-ranking only positive pairs or positive pair + strong negative.
# loss = ["denoising-autoencoder", 'multi-neg-ranking', 'cosine-similarity']  # softmax, "denoising-autoencoder", 'multi-neg-ranking', 'cosine-similarity', multi-neg-ranking only positive pairs or positive pair + strong negative.
loss_contrastive_label_pos="entailment"
loss_contrastive_label_neg="contradiction"
special_tokens = []  # ["[USR]", "[SYS]"]
eval_metric = "loss"  # "coorelation-score"  (Spearman correlation) + sklearn classification_report metrics ("f1-score", "accuracy", "recall", "precision")
# eval_metric = "coorelation-score"  # "coorelation-score"  (Spearman correlation) + sklearn classification_report metrics ("f1-score", "accuracy", "recall", "precision")
eval_metric_avg = "macro"  # "macro", "micro", "weighted" (ignore if not classification, )
max_seq_length = None
batch_size = 16
num_epochs = 1
evals_per_epoch = 50
warmup_pct = 0.1
learning_rate = 2e-5
log_interval = 100
optimizer = torch.optim.AdamW
path_trainset = "data/AllNLI_train.csv"
path_devset = "data/AllNLI_dev.csv"
path_testset = "data/AllNLI_test.csv"
# path_trainset = ["data/dialogue.txt", "data/AllNLI_train.csv", "data/stsbenchmark_train.csv"]
# path_devset = "data/stsbenchmark_dev.csv"
# path_testset = "data/stsbenchmark_test.csv"
path_output = "output/test"
checkpoint_path = "output/test/checkpoints"
checkpoint_saves_per_epoch = 0
checkpoint_save_total_limit = 0
checkpoint_save_after_each_epoch = True
project_name = None

torch.manual_seed(DEFAULT_SEED)
np.random.seed(DEFAULT_SEED)
random.seed(DEFAULT_SEED)

if isinstance(path_trainset, str):
    path_trainset = [path_trainset]
if isinstance(loss, str):
    loss = [loss]
if not path_devset:
    path_devset = ''
if not path_testset:
    path_testset = ''


def get_dataset_name(paths:Union[List[str], str]) -> str:
    """Given a path, or a list of paths, return string with file name(s)."""
    if isinstance(paths, str):
        paths = [paths]

    filenames = [os.path.split(path)[1] for path in paths]
    return '|'.join([filename[:filename.find('.')] for filename in filenames])


def get_project_name(path_trainset:str, path_devset:str, model_name:str, pooling_mode:str, loss:str) -> str:
    """Given the provided parameters return a string to identify the project."""
    return f"train[{get_dataset_name(path_trainset)}]eval[{get_dataset_name(path_devset)}]" + model_name.replace("/", "-") + f"[pooling-{pooling_mode}][loss-{'|'.join(loss)}]"


def get_dataset_by_loss(loss_name, data):
    if loss_name == "softmax":
        return SimilarityDataset(data)
    elif loss_name == "multi-neg-ranking":
        return SimilarityDatasetContrastive(data, label_pos=loss_contrastive_label_pos, label_neg=loss_contrastive_label_neg)
    elif loss_name == "cosine-similarity":
        return SimilarityDataset(data, is_regression=True, normalize_value=True)
    elif loss_name == "denoising-autoencoder":  # unsupervised
        return datasets.DenoisingAutoEncoderDataset(data)


def get_dataloader_by_loss(loss_name, dataset, shuffle=True):
    if loss_name == "multi-neg-ranking":
        return datasets.NoDuplicatesDataLoader(dataset, batch_size=batch_size)

    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, drop_last=True)


def get_evaluator_by_metric(path_evalset, metric, evaluator_name):
    data = SimilarityDataReader.read_csv(path_evalset, col_sent0=DEFAULT_NAME_COL_SENT1, col_sent1=DEFAULT_NAME_COL_SENT2, col_label=DEFAULT_NAME_COL_LABEL)

    if metric == "coorelation-score":
        evalset = SimilarityDataset(data, is_regression=True, normalize_value=True)
        return EmbeddingSimilarityEvaluator.from_input_examples(
            evalset, main_similarity=SimilarityFunction.COSINE,
            batch_size=batch_size, name=evaluator_name
        )
    elif metric in ["accuracy", "f1-score", "recall", "precision"]:
        evalset = DataLoader(SimilarityDataset(data), shuffle=False, batch_size=batch_size)

        _, softmax_loss = train_objectives[0]  # TODO: search for the right softmax loss in the list (not necessarily has to be the one at index [0])
        return ClassificationEvaluator(evalset, softmax_model=softmax_loss,
                                       metric=metric, metric_avg=eval_metric_avg,
                                       name=evaluator_name)
    elif metric == "loss":
        # TODO: as above, allow the user to specify which loss from the provided list (not necessarily [0])
        loss_name = loss[0]
        _, target_loss = train_objectives[0]
        evalset = get_dataloader_by_loss(loss_name,
                                         get_dataset_by_loss(loss_name, data),
                                         shuffle=False)
        return LossEvaluator(evalset, target_loss, name=evaluator_name)
    else:
        raise ValueError(f"evaluation metric '{metric}' is not supported.")


def on_evaluation(score:float, avg_losses:List[float], epoch:int, steps:int) -> None:
    # if it's the evaluation perform automatically after finishing the epoch, use "custom epoch" step
    if steps == -1:
        wandb.log({f"{eval_metric}_epoch": score, "epoch": epoch + 1})
    else:  # if not just use default wandb steps
        metrics = {eval_metric: score}
        if len(avg_losses) > 1:
            metrics.update({f"train_loss_obj{ix}" : avg_loss for ix, avg_loss in enumerate(avg_losses)})
        elif len(avg_losses) == 1:
            metrics.update({"train_loss" : avg_losses[0]})
        wandb.log(metrics)


def eval_during_training(model, evaluator, output_path, save_best_model, epoch, steps, loss_values, callback):
    """Runs evaluation during the training"""
    eval_path = output_path
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        eval_path = os.path.join(output_path, "eval")
        os.makedirs(eval_path, exist_ok=True)

    # average loss per objective/task
    avg_losses = [sum(ll) / float(len(ll)) if len(ll) else 0 for ll in loss_values]

    if evaluator is not None:
        score = evaluator(model, output_path=eval_path, epoch=epoch, steps=steps)
        if callback is not None:
            callback(score, avg_losses, epoch, steps)
        if eval_metric != "loss" and score > model.best_score:
            model.best_score = score
            if save_best_model:
                model.save(output_path)
    else:
        if callback is not None:
            callback(None, avg_losses, epoch, steps)


def train(model,
          train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
          evaluator: SentenceEvaluator = None,
          epochs: int = 1,
          steps_per_epoch = None,
          scheduler: str = 'WarmupLinear',
          warmup_steps: int = 10000,
          optimizer_class: Type[Optimizer] = torch.optim.AdamW,
          optimizer_params : Dict[str, object]= {'lr': 2e-5},
          weight_decay: float = 0.01,
          evaluation_steps: int = 0,
          output_path: str = None,
          save_best_model: bool = True,
          max_grad_norm: float = 1,
          use_amp: bool = False,
          callback: Callable[[float, int, int], None] = None,
          show_progress_bar: bool = True,
          checkpoint_path: str = None,
          checkpoint_save_steps: int = 500,
          checkpoint_save_total_limit: int = 0,
          checkpoint_save_after_each_epoch: bool = False,
        ):
    """
    Train the model with the given training objective
    Each training objective is sampled in turn for one batch.
    We sample only as many batches from each objective as there are in the smallest one
    to make sure of equal training with each dataset.

    :param train_objectives: Tuples of (DataLoader, LossFunction). Pass more than one for multi-task learning
    :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
    :param epochs: Number of epochs for training
    :param steps_per_epoch: Number of training steps per epoch. If set to None (default), one epoch is equal the DataLoader size from train_objectives.
    :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
    :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
    :param optimizer_class: Optimizer
    :param optimizer_params: Optimizer parameters
    :param weight_decay: Weight decay for model parameters
    :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
    :param output_path: Storage path for the model and evaluation files
    :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
    :param max_grad_norm: Used for gradient normalization.
    :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
    :param callback: Callback function that is invoked after each evaluation.
            It must accept the following three parameters in this order:
            `score`, `epoch`, `steps`
    :param show_progress_bar: If True, output a tqdm progress bar
    :param checkpoint_path: Folder to save checkpoints during training
    :param checkpoint_save_steps: Will save a checkpoint after so many steps
    :param checkpoint_save_total_limit: Total number of checkpoints to store
    """

    info_loss_functions =  []
    for dataloader, loss in train_objectives:
        info_loss_functions.extend(ModelCardTemplate.get_train_objective_info(dataloader, loss))
    info_loss_functions = "\n\n".join([text for text in info_loss_functions])

    info_fit_parameters = json.dumps({"evaluator": fullname(evaluator), "epochs": epochs, "steps_per_epoch": steps_per_epoch, "scheduler": scheduler, "warmup_steps": warmup_steps, "optimizer_class": str(optimizer_class),  "optimizer_params": optimizer_params, "weight_decay": weight_decay, "evaluation_steps": evaluation_steps, "max_grad_norm": max_grad_norm }, indent=4, sort_keys=True)
    model._model_card_text = None
    model._model_card_vars['{TRAINING_SECTION}'] = ModelCardTemplate.__TRAINING_SECTION__.replace("{LOSS_FUNCTIONS}", info_loss_functions).replace("{FIT_PARAMETERS}", info_fit_parameters)

    if use_amp:
        from torch.cuda.amp import autocast
        scaler = torch.cuda.amp.GradScaler()

    model.to(model._target_device)

    dataloaders = [dataloader for dataloader, _ in train_objectives]

    # Use smart batching
    for dataloader in dataloaders:
        dataloader.collate_fn = model.smart_batching_collate

    loss_models = [loss for _, loss in train_objectives]
    for loss_model in loss_models:
        loss_model.to(model._target_device)

    model.best_score = -9999999

    if steps_per_epoch is None or steps_per_epoch == 0:
        steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

    num_train_steps = int(steps_per_epoch * epochs)

    # Prepare optimizers
    optimizers = []
    schedulers = []
    for loss_model in loss_models:
        param_optimizer = list(loss_model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
        scheduler_obj = model._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

        optimizers.append(optimizer)
        schedulers.append(scheduler_obj)

    global_step = 0
    data_iterators = [iter(dataloader) for dataloader in dataloaders]

    num_train_objectives = len(train_objectives)
    loss_value_lists = [[] for _ in range(num_train_objectives)]

    skip_scheduler = False
    for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
        training_steps = 0

        for loss_model in loss_models:
            loss_model.zero_grad()
            loss_model.train()

        for _ in trange(steps_per_epoch, desc="Step", smoothing=0.05, disable=not show_progress_bar):
            for train_idx in range(num_train_objectives):
                loss_model = loss_models[train_idx]
                loss_values = loss_value_lists[train_idx]
                optimizer = optimizers[train_idx]
                scheduler = schedulers[train_idx]
                data_iterator = data_iterators[train_idx]

                try:
                    data = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(dataloaders[train_idx])
                    data_iterators[train_idx] = data_iterator
                    data = next(data_iterator)

                features, labels = data
                labels = labels.to(model._target_device)
                features = list(map(lambda batch: batch_to_device(batch, model._target_device), features))

                if use_amp:
                    with autocast():
                        loss_value = loss_model(features, labels)

                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss_value).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    skip_scheduler = scaler.get_scale() != scale_before_step
                else:
                    loss_value = loss_model(features, labels)
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                    optimizer.step()

                loss_values.append(loss_value.item())
                optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()

            training_steps += 1
            global_step += 1

            if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                eval_during_training(model, evaluator, output_path, save_best_model, epoch, training_steps, loss_value_lists, callback)

                for ix, loss_model in enumerate(loss_models):
                    loss_model.zero_grad()
                    loss_model.train()

                    loss_value_lists[ix][:] = []

            if checkpoint_path is not None and checkpoint_save_steps is not None and checkpoint_save_steps > 0 and global_step % checkpoint_save_steps == 0:
                model._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)

        if checkpoint_save_after_each_epoch:
            model._save_checkpoint(checkpoint_path, None, f"epoch-{epoch}")

        eval_during_training(model, evaluator, output_path, save_best_model, epoch, -1, loss_value_lists, callback)

    if (evaluator is None or eval_metric == "loss") and output_path is not None:
        model.save(output_path)

    if checkpoint_path is not None:
        model._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)


# Set up wandb
project_name = (project_name or get_project_name(path_trainset, path_devset, model_name, pooling_mode, loss))[:128]
wandb.init(
    project=project_name,
    config={
        # TODO: add more (all the arguments that are important)
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

# Setting up the model
transformer_seq_encoder = models.Transformer(model_name, max_seq_length=max_seq_length)

if special_tokens:
    transformer_seq_encoder.tokenizer.add_tokens(special_tokens, special_tokens=True)
    transformer_seq_encoder.auto_model.resize_token_embeddings(len(transformer_seq_encoder.tokenizer))

sentence_vector = models.Pooling(transformer_seq_encoder.get_word_embedding_dimension(), pooling_mode=pooling_mode)
model = SentenceTransformer(modules=[transformer_seq_encoder, sentence_vector])
wandb.watch(model, log_freq=WANDB_LOG_FREQ)

# Loading datasets, data loaders and losses
# Training sets
logging.info(f"Reading training sets ({path_trainset})")
train_objectives = []
for ix, path in enumerate(path_trainset):
    loss_name = loss[:ix + 1][-1]

    # if it's unsupervised
    if loss_name == "denoising-autoencoder":
        # read raw txt file, each line is a sample sentence
        data =list(SimilarityDataReader.read_docs(path, lines_are_documents=True))
    else:
        data = SimilarityDataReader.read_csv(path, col_sent0=DEFAULT_NAME_COL_SENT1, col_sent1=DEFAULT_NAME_COL_SENT2, col_label=DEFAULT_NAME_COL_LABEL)

    data = get_dataset_by_loss(loss_name, data)

    if loss_name == "softmax":
        loss_fn = losses.SoftmaxLoss(model=model,
                                     sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                     num_labels=data.num_labels)
    elif loss_name == "multi-neg-ranking":
        loss_fn = losses.MultipleNegativesRankingLoss(model=model)
    elif loss_name == "cosine-similarity":
        loss_fn = losses.CosineSimilarityLoss(model=model)
    elif loss_name == "denoising-autoencoder":  # unsupervised
        loss_fn = losses.DenoisingAutoEncoderLoss(model, tie_encoder_decoder=True)
    else:
        raise ValueError(f"Loss {loss_name} not supported.")

    train_objectives.append((get_dataloader_by_loss(loss_name, data), loss_fn))


# Evaluation/development set
dev_evaluator = None
if path_devset:
    logging.info(f"Reading development set ({path_devset})")
    dev_evaluator = get_evaluator_by_metric(path_devset, eval_metric, evaluator_name="devset")

# Training
steps_per_epoch = min([len(data_loader) for data_loader, _ in train_objectives])  # if multiple trainingsets, sets will be repeated as in a round-robin queue, 1 epochs = full smallest dataset, increase epoch to cover more parts of the bigger ones
warmup_steps = math.ceil(steps_per_epoch * num_epochs * warmup_pct)
logging.info("Warmup steps: {}".format(warmup_steps))

train(model, train_objectives=train_objectives,
      evaluator=dev_evaluator,
      epochs=num_epochs,
      steps_per_epoch=steps_per_epoch,
      evaluation_steps=max(steps_per_epoch // evals_per_epoch, MIN_EVALUATION_STEPS) if evals_per_epoch > 0 else 0,
      warmup_steps=warmup_steps,
      output_path=path_output,
      optimizer_class=optimizer,
      optimizer_params={'lr': learning_rate},
      checkpoint_path=checkpoint_path,
      checkpoint_save_steps=max(steps_per_epoch // checkpoint_saves_per_epoch, MIN_CHECKPOINT_SAVE_STEPS) if checkpoint_saves_per_epoch else 0,
      checkpoint_save_total_limit=checkpoint_save_total_limit,
      checkpoint_save_after_each_epoch=checkpoint_save_after_each_epoch,
      callback=on_evaluation)


# If test set, then evaluate on test set...
if path_testset:
    logging.info(f"Loading final model from disk ({path_output})")
    model = SentenceTransformer(path_output)

    torch.cuda.empty_cache()
    model.to(model._target_device)

    logging.info(f"Reading the test set ({path_testset})")
    test_evaluator = get_evaluator_by_metric(path_testset, eval_metric, evaluator_name="testset")

    logging.info(f"Evaluating model on the test set data...")
    test_evaluator(model, output_path=path_output)
