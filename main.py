"""
Script to pre-train sentence-BERT-like models for task-oriented dialogue.
"""
import os
import json
import math
import torch
import wandb
import hydra
import random
import logging
import numpy as np

from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from omegaconf import DictConfig
from tqdm.autonotebook import trange
from typing import Union, List, Iterable, Tuple, Type, Dict

from sentence_transformers import models, datasets
from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers.util import fullname, batch_to_device
from sentence_transformers.evaluation import SentenceEvaluator, SimilarityFunction, EmbeddingSimilarityEvaluator
from sentence_transformers.model_card_templates import ModelCardTemplate

from similarity_evaluation import ClassificationEvaluator, LossEvaluator
from similarity_datasets import SimilarityDataReader, SimilarityDataset, SimilarityDatasetContrastive
from similiraty_losses import BaseLoss, SoftmaxLoss, CosineSimilarityLoss, DenoisingAutoEncoderLoss, MultipleNegativesRankingLoss

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt=r'%Y-%m-%d %H:%M:%S',
                    level=logging.INFO, handlers=[LoggingHandler()])

DEFAULT_OPTIMIZER = torch.optim.AdamW

# Globals to be populated in main()
CONTRASTIVE_LABEL_POS = None
CONTRASTIVE_LABEL_NEG = None
CSV_COL_SENT1 = None
CSV_COL_SENT2 = None
CSV_COL_LABEL = None


def get_loss_class_name(loss_name: str) -> str:
    """
    Convert to original-name to UpperCamelCase class name.
    TODO: :param NAME: description to this and all the functions below...
    """
    return ''.join([w.capitalize() for w in loss_name.split('-')]) + "Loss"


def get_dataset_name(paths: Union[List[str], str]) -> str:
    """Given a path, or a list of paths, return string with file name(s)."""
    if isinstance(paths, str):
        paths = [paths]

    filenames = [os.path.split(path)[1] for path in paths]
    return '|'.join([filename[:filename.find('.')] for filename in filenames])


def get_default_wandb_project_name(path_trainset: str, path_devset: str, model_name: str, pooling_mode: str, loss: str) -> str:
    """Given the provided parameters return a string to identify the project."""
    project_name = f"train[{get_dataset_name(path_trainset)}]eval[{get_dataset_name(path_devset)}]" + model_name.replace("/", "-") + f"[pooling-{pooling_mode}][loss-{'|'.join(loss)}]"
    return project_name[:128]  # wandb project name can't have more than 128 characters


def get_dataset_by_loss(loss_name: str, data: Iterable) -> Dataset:
    """Get the proper Dataset object for the given loss."""
    if loss_name == SoftmaxLoss.__name__:
        return SimilarityDataset(data)
    elif loss_name == MultipleNegativesRankingLoss.__name__:
        return SimilarityDatasetContrastive(data, label_pos=CONTRASTIVE_LABEL_POS, label_neg=CONTRASTIVE_LABEL_NEG)
    elif loss_name == CosineSimilarityLoss.__name__:
        return SimilarityDataset(data, is_regression=True, normalize_value=True)
    elif loss_name == DenoisingAutoEncoderLoss.__name__:  # unsupervised
        return datasets.DenoisingAutoEncoderDataset(data)


def get_dataloader_by_loss(loss_name: str, dataset: Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    """Get the proper DataLoader for the given loss."""
    # If contrastive unsupervised...
    if loss_name == MultipleNegativesRankingLoss.__name__:
        # Make sure there are no duplicate instances in the batch
        return datasets.NoDuplicatesDataLoader(dataset, batch_size=batch_size)

    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, drop_last=True)


def get_loss_by_name(loss_name: str, data: Dataset, model: SentenceTransformer) -> BaseLoss:
    """Get the Loss object given its name."""
    if loss_name == SoftmaxLoss.__name__:
        return SoftmaxLoss(model=model,
                           sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                           num_labels=data.num_labels)
    elif loss_name == MultipleNegativesRankingLoss.__name__:
        return MultipleNegativesRankingLoss(model=model)
    elif loss_name == CosineSimilarityLoss.__name__:
        return CosineSimilarityLoss(model=model)
    elif loss_name == DenoisingAutoEncoderLoss.__name__:  # unsupervised
        return DenoisingAutoEncoderLoss(model, tie_encoder_decoder=True)
    else:
        raise ValueError(f"Loss {loss_name} not supported.")


def get_evaluator_by_metric(path_evalset: str, metric: str, metric_avg: str = "",
                            loss_model: BaseLoss = None, batch_size: int = None,
                            evaluator_name: str = '') -> SentenceEvaluator:
    """
    Get the Evaluator object for the given metric name.

    If `metric` == 'loss' then the `loss_model` should contain the concrete loss object to use for evaluation
    """

    # if it's unsupervised
    if metric == "loss" and isinstance(loss_model, DenoisingAutoEncoderLoss):
        # read raw txt file, each line is a sample sentence
        data = list(SimilarityDataReader.read_docs(path_evalset, lines_are_documents=True))
    else:
        data = SimilarityDataReader.read_csv(path_evalset, col_sent0=CSV_COL_SENT1, col_sent1=CSV_COL_SENT2, col_label=CSV_COL_LABEL)

    if metric == "coorelation-score":
        evalset = SimilarityDataset(data, is_regression=True, normalize_value=True)
        return EmbeddingSimilarityEvaluator.from_input_examples(
            evalset, main_similarity=SimilarityFunction.COSINE,
            batch_size=batch_size, name=evaluator_name
        )
    elif metric in ["accuracy", "f1-score", "recall", "precision"]:
        evalset = DataLoader(SimilarityDataset(data), shuffle=False, batch_size=batch_size)
        return ClassificationEvaluator(evalset, softmax_model=loss_model,
                                       metric=metric, metric_avg=metric_avg,
                                       name=evaluator_name)
    elif metric == "loss":
        loss_name = loss_model.__name__
        evalset = get_dataloader_by_loss(loss_name,
                                         get_dataset_by_loss(loss_name, data),
                                         batch_size=batch_size, shuffle=False)
        return LossEvaluator(evalset, loss_model, name=evaluator_name)
    else:
        raise ValueError(f"evaluation metric '{metric}' is not supported.")


def wandb_log(score: float, avg_losses: List[float], metric_name: str, epoch: int, steps: int) -> None:
    """
    Log evaluation results in WandB.

    When called after finishing each bach, `steps` will be equal to -1.
    """

    # if it's the evaluation perform automatically **after finishing the epoch**, use "epoch" as x-axis
    if steps == -1:
        wandb.log({f"{metric_name}_epoch": score, "epoch": epoch + 1})
    else:  # if not just use default wandb steps as x-axis
        metrics = {metric_name: score}
        if len(avg_losses) > 1:
            metrics.update({f"train_loss_obj{ix}": avg_loss for ix, avg_loss in enumerate(avg_losses)})
        elif len(avg_losses) == 1:
            metrics.update({"train_loss": avg_losses[0]})
        wandb.log(metrics)


def eval_during_training(model: SentenceTransformer, evaluator: SentenceEvaluator,
                         output_path: str, save_best_model: bool, epoch: int, steps: int,
                         loss_values: list) -> None:
    """Runs evaluation during the training using the provided evalutor object."""
    eval_path = output_path
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        eval_path = os.path.join(output_path, "eval")
        os.makedirs(eval_path, exist_ok=True)

    # average loss per objective/task since the last evaluation/call
    avg_losses = [sum(ll) / float(len(ll)) if len(ll) else 0 for ll in loss_values]

    if evaluator is not None:
        score = evaluator(model, output_path=eval_path, epoch=epoch, steps=steps)
        wandb_log(score, avg_losses, evaluator.metric_name, epoch, steps)
        if evaluator.metric_name != "loss" and score > model.best_score:
            model.best_score = score
            if save_best_model:
                model.save(output_path)
    else:
        wandb_log(None, avg_losses, evaluator.metric_name, epoch, steps)


# Modified from the original sentence-bert model.fit() implementation
# (https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py#L575)
def train(
        model,
        train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
        evaluator: SentenceEvaluator = None,
        epochs: int = 1,
        steps_per_epoch: int = None,
        scheduler: str = 'WarmupLinear',
        warmup_steps: int = 10000,
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        optimizer_params: Dict[str, object] = {'lr': 2e-5},
        weight_decay: float = 0.01,
        evaluation_steps: int = 0,
        output_path: str = None,
        save_best_model: bool = True,
        max_grad_norm: float = 1,
        use_amp: bool = False,
        show_progress_bar: bool = True,
        checkpoint_path: str = None,
        checkpoint_save_steps: int = 500,
        checkpoint_save_total_limit: int = 0,
        checkpoint_save_after_each_epoch: bool = False,
):
    """
    Train the model with the given training objective(s).

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
    :param show_progress_bar: If True, output a tqdm progress bar
    :param checkpoint_path: Folder to save checkpoints during training
    :param checkpoint_save_steps: Will save a checkpoint after so many steps
    :param checkpoint_save_total_limit: Total number of checkpoints to store
    """
    info_loss_functions = []
    for dataloader, loss in train_objectives:
        info_loss_functions.extend(ModelCardTemplate.get_train_objective_info(dataloader, loss))
    info_loss_functions = "\n\n".join([text for text in info_loss_functions])

    info_fit_parameters = json.dumps({"evaluator": fullname(evaluator), "epochs": epochs, "steps_per_epoch": steps_per_epoch,
                                      "scheduler": scheduler, "warmup_steps": warmup_steps, "optimizer_class": str(optimizer_class),
                                      "optimizer_params": optimizer_params, "weight_decay": weight_decay, "evaluation_steps": evaluation_steps,
                                      "max_grad_norm": max_grad_norm}, indent=4, sort_keys=True)
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

    model.best_score = float("-inf")

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
                eval_during_training(model, evaluator, output_path, save_best_model, epoch, training_steps, loss_value_lists)

                for ix, loss_model in enumerate(loss_models):
                    loss_model.zero_grad()
                    loss_model.train()

                    loss_value_lists[ix][:] = []

            if checkpoint_path is not None and checkpoint_save_steps is not None and checkpoint_save_steps > 0 and global_step % checkpoint_save_steps == 0:
                model._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)

        if checkpoint_save_after_each_epoch:
            model._save_checkpoint(checkpoint_path, None, f"epoch-{epoch}")

        eval_during_training(model, evaluator, output_path, save_best_model, epoch, -1, loss_value_lists)

    if (evaluator is None or evaluator.metric_name == "loss") and output_path is not None:
        model.save(output_path)

    if checkpoint_path is not None:
        model._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    global CONTRASTIVE_LABEL_POS, CONTRASTIVE_LABEL_NEG, CSV_COL_SENT1, CSV_COL_SENT2, CSV_COL_LABEL

    CONTRASTIVE_LABEL_POS = cfg.contrastive_learning.label_pos
    CONTRASTIVE_LABEL_NEG = cfg.contrastive_learning.label_neg
    CSV_COL_SENT1 = cfg.datasets.csv.column_name_sent1
    CSV_COL_SENT2 = cfg.datasets.csv.column_name_sent2
    CSV_COL_LABEL = cfg.datasets.csv.column_name_ground_truth

    num_epochs = cfg.training.num_epochs
    batch_size = cfg.training.batch_size
    warmup_pct = cfg.training.warmup_pct
    learning_rate = cfg.training.learning_rate
    evals_per_epoch = cfg.evaluation.evaluations_per_epoch
    checkpoint_saves_per_epoch = cfg.checkpointing.saves_per_epoch
    optimizer = DEFAULT_OPTIMIZER

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    if isinstance(cfg.target.trainsets, str):
        cfg.target.trainsets = [cfg.target.trainsets]
    if isinstance(cfg.target.losses, str):
        cfg.target.losses = [cfg.target.losses]
    if not cfg.evaluation.devset:
        cfg.evaluation.devset = ''
    if not cfg.evaluation.testset:
        cfg.evaluation.testset = ''

    # 1. Set up wandb
    project_name = (cfg.wandb.project_name or get_default_wandb_project_name(cfg.target.trainsets, cfg.evaluation.devset,
                                                                             cfg.model.base, cfg.model.pooling_mode, cfg.target.losses))
    wandb.init(
        project=project_name,
        config={
            # TODO: add more (all we consider important to be added)
            "learning_rate": learning_rate,
            "loss": cfg.target.losses,
            "model": cfg.model.base,
            "pooling_mode": cfg.model.pooling_mode,
            "trainset": cfg.target.trainsets,
            "devset": cfg.evaluation.devset,
            "warmup_data_percentage": warmup_pct,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "optimizer": str(optimizer)
        }
    )
    wandb.define_metric("epoch")
    wandb.define_metric("epoch_score", step_metric="epoch")

    # 2. Set up the model
    transformer_seq_encoder = models.Transformer(cfg.model.base, max_seq_length=cfg.model.max_seq_length)

    if cfg.model.special_tokens:
        transformer_seq_encoder.tokenizer.add_tokens(cfg.model.special_tokens, special_tokens=True)
        transformer_seq_encoder.auto_model.resize_token_embeddings(len(transformer_seq_encoder.tokenizer))

    sentence_vector = models.Pooling(transformer_seq_encoder.get_word_embedding_dimension(), pooling_mode=cfg.model.pooling_mode)
    model = SentenceTransformer(modules=[transformer_seq_encoder, sentence_vector])
    wandb.watch(model, log_freq=cfg.wandb.log_freq)

    # 3. Loading datasets, data loaders and losses
    # 3.1. Training sets
    logging.info(f"Reading training sets ({cfg.target.trainsets})")
    train_objectives = []
    target_losses = [get_loss_class_name(loss_name) for loss_name in cfg.target.losses]
    for ix, path in enumerate(cfg.target.trainsets):
        loss_name = target_losses[:ix + 1][-1]  # trick to avoid IndexError when there are more datasets than losses by returning the last one

        # if it's unsupervised
        if loss_name == DenoisingAutoEncoderLoss.__name__:
            # read raw txt file, each line is a sample sentence
            data = list(SimilarityDataReader.read_docs(path, lines_are_documents=True))
        else:
            data = SimilarityDataReader.read_csv(path, col_sent0=CSV_COL_SENT1, col_sent1=CSV_COL_SENT2, col_label=CSV_COL_LABEL)

        data = get_dataset_by_loss(loss_name, data)
        loss_fn = get_loss_by_name(loss_name, data, model)

        train_objectives.append((get_dataloader_by_loss(loss_name, data, batch_size=batch_size), loss_fn))

    # Assuming here the first loss at [0] is the one is used
    # For evaluation, should be somehow allow the user to specify a different one?
    _, evaluation_loss = train_objectives[0]

    # 3.2. Evaluation/development set
    dev_evaluator = None
    if cfg.evaluation.devset:
        logging.info(f"Reading development set ({cfg.evaluation.devset})")
        dev_evaluator = get_evaluator_by_metric(cfg.evaluation.devset, cfg.evaluation.metric, cfg.evaluation.metric_avg, evaluation_loss, batch_size=batch_size, evaluator_name="devset")
        dev_evaluator.metric_name = cfg.evaluation.metric

    # 4. Training
    steps_per_epoch = min([len(data_loader) for data_loader, _ in train_objectives])  # if multiple trainingsets, sets will be repeated as in a round-robin queue, 1 epochs = full smallest dataset, increase epoch to cover more parts of the bigger ones
    warmup_steps = math.ceil(steps_per_epoch * num_epochs * warmup_pct)
    logging.info("Warmup steps: {}".format(warmup_steps))

    train(model,
          train_objectives=train_objectives,
          evaluator=dev_evaluator,
          epochs=num_epochs,
          steps_per_epoch=steps_per_epoch,
          evaluation_steps=max(steps_per_epoch // evals_per_epoch, cfg.evaluation.min_steps) if evals_per_epoch > 0 else 0,
          warmup_steps=warmup_steps,
          output_path=cfg.evaluation.best_model_output_path,
          optimizer_class=optimizer,
          optimizer_params={'lr': learning_rate},
          checkpoint_path=cfg.checkpointing.path,
          checkpoint_save_steps=max(steps_per_epoch // checkpoint_saves_per_epoch, cfg.checkpointing.min_steps) if checkpoint_saves_per_epoch else 0,
          checkpoint_save_total_limit=cfg.checkpointing.total_limit,
          checkpoint_save_after_each_epoch=cfg.checkpointing.always_save_after_each_epoch)

    # 5. If test set, then evaluate model on it...
    if cfg.evaluation.testset:
        logging.info(f"Loading final model from disk ({cfg.evaluation.best_model_output_path})")
        model = SentenceTransformer(cfg.evaluation.best_model_output_path)

        torch.cuda.empty_cache()
        model.to(model._target_device)

        logging.info(f"Reading the test set ({cfg.evaluation.testset})")
        test_evaluator = get_evaluator_by_metric(cfg.evaluation.testset, cfg.evaluation.metric, cfg.evaluation.metric_avg, evaluation_loss, batch_size=batch_size, evaluator_name="testset")

        logging.info("Evaluating model on the test set data...")
        test_evaluator(model, output_path=cfg.evaluation.best_model_output_path)


if __name__ == '__main__':
    main()
