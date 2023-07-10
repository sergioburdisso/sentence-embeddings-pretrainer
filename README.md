# :speech_balloon: Sentence Embeddings Pre-Trainer

The sentence embedding pre-training pipeline assumes the input are pairs of sentences (sent1, sent2) or pair of sentences plus an optional grand truth value (sent1, sent2, value) where value could be a label or a similarity numerical value. These pairs or triplets can be passed to the pipeline as dataset files (e.g. csv file) or using custom `Dataset`/`DataLoaders` that will provide them. Note here, in the context of dialogue, by "sentences" or "pair of sentences" we don't necessarily mean literally sentences but rather any piece of text, could be pairs `(context, next turn)`, pairs of consecutive turns `(turn_i, turn_{i+1})`, etc.

This pipeline follows the structure described in [Sentence-BERT paper](https://arxiv.org/pdf/1908.10084.pdf) and thus is built on top `sentence-bert` codebase. Instead of having two different encoders for context/query and label/response respectively or one cross-encoder, here we have one single encoder that will be trained as a bi-encoder[*].

To use the pipeline we need to specify: (a) the base model used to encode the input tokens; (b) the pooling strategy to convert the the encodings of all the input tokens (last hidden states) into a single sentence embedding vector (e.g. [CLS] token, mean, max, etc.); and (c) how to compute the loss and on which dataset. The full input arguments are described in the next section.



> _[*] We can change this later, modify the code to allow having actually two different models though not sure it is a good idea (we'll need to discuss about it)._



## :electric_plug: Input arguments

The main script uses [Hydra](https://hydra.cc/) to manage input arguments. The arguments hierarchical structure with example default values are in the `config.yaml`:

```yaml
seed: 13

model:
  base: "bert-base-uncased"  # e.g. "TODBERT/TOD-BERT-JNT-V1"
  pooling_mode: "cls"  # "cls"; "mean"; "max"; "weightedmean"; "lasttoken";
  special_tokens: []  # e.g. ["[USR]", "[SYS]"]
  max_seq_length: null  # if null, then min(auto_model.config.max_position_embeddings, tokenizer.model_max_length)

target: # if multiple losses and traininsets are given as below, it will be treated as multi-task learning.
  losses: ["denoising-auto-encoder", "multiple-negatives-ranking", "cosine-similarity"]  # "softmax"; "denoising-auto-encoder"; "multiple-negatives-ranking"; "cosine-similarity";
  trainsets: ["data/testing/dialogue.txt", "data/testing/AllNLI_train.csv", "data/testing/stsbenchmark_train.csv"]

contrastive_learning:
  label_pos: "entailment"
  label_neg: "contradiction"

evaluation:
  evaluations_per_epoch: 50  # how many evaluations to perform per epoch
  min_steps: 100  # minimum number of steps for evaluations to happend
  metric: "coorelation-score"  # "coorelation-score"  (Spearman correlation) or "f1-score", "accuracy", "recall", "precision" (sklearn classification_report metrics)
  metric_avg: "macro"  # "macro", "micro", "weighted" (ignore if not classification)
  devset: "data/testing/stsbenchmark_dev.csv"
  testset: "data/testing/stsbenchmark_test.csv"
  best_model_output_path: "outputs/testing"

training:
  batch_size: 16
  num_epochs: 2
  warmup_pct: 0.1  # %
  learning_rate: 2e-5

checkpointing:
  saves_per_epoch: 0
  min_steps: 50  # minimum number of steps for checkpoint saving to happend
  total_limit: 0
  always_save_after_each_epoch: True
  path: "outputs/test/checkpoints"

wandb:
  project_name: null  # if null a defult name will be given using the provided parameters (traininset, model name, etc.)
  log_freq: 100

datasets:
  csv:
    column_name_sent1: "sent1"
    column_name_sent2: "sent2"
    column_name_ground_truth: "value"
```

From above we can see the input files are expected to be either a txt file in which each line is a different sentence or a csv file with two or three columns named: "sent1", "sent2", and "value".


## :open_file_folder: Files and directories

- `main.py` script to run the sentence embedding pre-training, input arguments described above.
- `config.yaml` contains input arguments/configuration for `main.py` script.
- `conda_env.yaml` conda environment name and dependencies.

- The `spretrainer` module/folder is divided into three submodules/subfolder:
    1. `datasets` contains everything related to input data:
        - `DataReader`s to read raw input files (e.g. csv, jsonl, txt) and return it into standard format.
        - `Dataset`s to return the actual dataset used for training as a list of `InputExample` instances. This `Dataset` objects can return samples from the reader as is or apply transformations to the original one (e.g. `DataReader` reads a dialogue, then `Dataset` convert the dialogue into pairs of consecutive turns `(turn_i, turn_{i + 1})` as positive pairs for contrastive learning).
        - `DataLoader`(s) used to iterate over the `Dataset`(s) to return the actual batches (e.g. for instance, a `DataLoader` to remove duplicates in batches when using the rest of the batch as negative samples).
    2. `evaluation` contains everything needed for model evaluation (evaluated as classification, regression or simply report the loss on provided dataset). To achieve this, an evaluator must be created/added for each new evaluation type is needed by defining a class that inherits from sentence-bert `SentenceEvaluator` class (e.g. `MyEvaluator(SentenceEvaluator):`) with whatever custom code and class constructor we want. The only constrain is that the `__call__(model, output_path, epoch, steps)` method has to be defined respecting those input arguments.
    3. `losses` contains losses that can be used for training the models. Each loss implemented as a class that inherits from `BaseLoss` providing custom code for the `forward(sentence_features, labels)` method (class constructor and anyother part of the class can be implemented with no restriction). So far there are available 4 of the sentence-bert losses (added as examples):
        - [`CosineSimilarityLoss`](https://www.sbert.net/docs/package_reference/losses.html#cosinesimilarityloss) (from [sbert original paper](https://arxiv.org/pdf/1908.10084.pdf), regression head) - _requires ground truth value_.
        - [`SoftmaxLoss`](https://www.sbert.net/docs/package_reference/losses.html#softmaxloss) (from [sbert original paper](https://arxiv.org/pdf/1908.10084.pdf), classification head) - _requires ground truth value_.
        - [`MultipleNegativesRankingLoss`](https://www.sbert.net/docs/package_reference/losses.html#multiplenegativesrankingloss) - Contrastive loss assuming provided pairs (sent1, sent2) are positive and the rest in the batch are negatives. It can accept hard negatives when labels are provided (sent1, sent2, label). To specify which value of label is pos and neg the `contrastive_learning.label_pos/neg` can be used when calling `main.py`.
        - [`DenoisingAutoEncoderLoss`](https://www.sbert.net/docs/package_reference/losses.html#denoisingautoencoderloss) _(unsupervised)_  uses a decoder to try to regenerate the corrupted input sentence only from the sentence embedding, described in the paper ["TSDAE: Using Transformer-based Sequential Denoising Auto-Encoder for Unsupervised Sentence Embedding Learning"](https://arxiv.org/pdf/2104.06979.pdf); this loss can be used to replace the MLM task and its variations like Masked Utterance Modeling described in [DialogueBERT](https://arxiv.org/ftp/arxiv/papers/2109/2109.10480.pdf), it is expected to produce better quality embeddings since we force the decoder to regenerate the missing pieces ("the [MASK]s") only from the single sentence embedding.

NOTE: Loss classes defined in the `losses` we'll be added automatically to input argument values by converting the UpperCamelCase name into Kebab case, so if you add a new loss, for instance, named "super magic" loss as the `SuperMagicLoss` class, the value `"super-magic"` will be automatically accepted as input argument in `target.losses`.


## :computer: How to run the code?

First, make sure your environment and dependencies are all set up:

```bash
$ conda env create -f conda_env.yaml
$ conda activate spretrainer
(spretrainer)$ pip install -r requirements.txt
```

_(regression)_ Now let's train our sentence embeddings using the default `config.yaml` by overritting the target values to train and evaluate with only Semantic Text Similarity task using the [STSbenchmark dataset](http://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark) and using "cosine similarity" loss:

```bash
$ python main.py target.trainsets="data/testing/stsbenchmark_train.csv" target.losses="cosine-similarity"
```
> (Spearman correlation on dev set: 0.8412)  
> (Spearman correlation on test set: 0.8100)

_(classification)_ Now, instead of training on STS task let's change the target to train the model on Natural Language Inference task (trained on small joint subset of [SNLI](https://nlp.stanford.edu/projects/snli/) and [MultiNLI](https://cims.nyu.edu/~sbowman/multinli/) dataset) with "softmax" loss but still evaluate it on STS.

```bash
$ python main.py target.trainsets="data/testing/allnli_train_small_20k.csv" \
                 target.losses="softmax"
```
> (Spearman correlation on dev set: 0.5251)  
> (Spearman correlation on test set: 0.4730)

_(classification + contrastive)_ Let's switch the loss to a contrastive loss using "Multiple Negatives Ranking" loss and the "entailment" and "contradiction" as the positive and negative labels respectively (by default in `config.yaml`):

```bash
$ python main.py target.trainsets="data/testing/allnli_train_small_20k.csv" \
                 target.losses="multiple-negatives-ranking"
```
> (Spearman correlation on dev set: 0.8180)  
> (Spearman correlation on test set: 0.7985)

_(unsupervised)_ Now let's train it without labels, using only a list of sentences and using Denoising Auto-Encoder loss as the target loss:

```bash
$ python main.py target.trainsets="data/testing/unlabeled_sentences.txt" \
                 target.losses="denoising-auto-encoder"
```
> (Spearman correlation on test set: 0.79668)  
> (Spearman correlation on test set: 0.7342)  
> -- not bad for being completely unsupervised :)



_(multi-task)_ Finally, let's learn sentence embeddings using multi-task learning with the previous task altogether:

```bash
$ python main.py 'target.trainsets=["data/testing/stsbenchmark_train.csv", "data/testing/allnli_train_small_20k.csv", "data/testing/unlabeled_sentences.txt"]' \
                 'target.losses=["cosine-similarity", "multiple-negatives-ranking", "denoising-auto-encoder"]'
```
> (Spearman correlation on dev set: 0.8651)  
> (Spearman correlation on test set: 0.8416)



## :pill: Testing

1. Unit tests

```bash
$ pytest
```

2. Source Code Styling Check

```bash
$ flake8
```

---
_(This code is related to [Keya's issue #22](https://github.com/keya-dialog/keya-dialog/issues/22))_