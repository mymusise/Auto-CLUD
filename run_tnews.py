import configs
import os
import json
import numpy as np

import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import TFGPT2LMHeadModel, TFGPT2ForSequenceClassification, GPT2Config
from transformers import TFTrainer, TFTrainingArguments
from transformers import XLNetTokenizer


# if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
#     tf.keras.mixed_precision.set_global_policy("mixed_float16")


def load_json2list(filename):
    with open(filename) as f:
        for line in f.readlines():
            if line:
                data = json.loads(line)
                yield data


def load_dataset(task_name: str):
    path = os.path.join(configs.DATA_PATH, task_name)
    train_file = os.path.join(path, "train.json")
    dev_file = os.path.join(path, "dev.json")
    test_file = os.path.join(path, "test.json")
    train_dataset = list(load_json2list(train_file))
    dev_dataset = list(load_json2list(dev_file))
    test_dataset = list(load_json2list(test_file))
    return train_dataset, dev_dataset, test_dataset


def get_label_nums(task_name: str):
    lines = open(os.path.join(configs.DATA_PATH, task_name, "labels.json")).readlines()
    return len(lines) + 2


def load_tokenizer():
    tokenizer = XLNetTokenizer.from_pretrained(configs.MODEL_PATH, padding_side="right")
    return tokenizer


def cover_dataset2np(dataset, is_test=False):
    tokenizer = load_tokenizer()
    inputs = [
        tokenizer.bos_token
        + " || ".join((item["sentence"], item["keywords"]))
        + tokenizer.eos_token
        for item in dataset
    ]
    if not is_test:
        labels = [int(item["label"]) - 100 for item in dataset]

        labels = np.array(labels, dtype=np.int)
        labels = labels.reshape(-1, 1)
    else:
        labels = None

    max_length = 0
    for i in inputs:
        if len(i) > max_length:
            max_length = len(i)
    inputs = tokenizer(
        inputs,
        max_length=max_length,
        padding="max_length",
        return_tensors="np",
        return_token_type_ids=False,
        return_attention_mask=False,
        add_special_tokens=False,
    )["input_ids"]
    return inputs, labels


def cover_np2tfds(inputs, labels):
    return (
        tf.data.Dataset.from_tensor_slices((inputs, labels))
        # .shuffle(inputs.shape[0], reshuffle_each_iteration=True)
        # .batch(configs.BATCH_SIZE)
    )


def load_model(task_name: str) -> TFGPT2ForSequenceClassification:
    model_config = GPT2Config.from_pretrained(
        configs.MODEL_PATH, num_labels=get_label_nums(task_name)
    )
    model = TFGPT2ForSequenceClassification.from_pretrained(
        configs.MODEL_PATH, config=model_config
    )
    return model


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="micro"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def init_trainer(train_dataset, dev_dataset, task_name):
    training_args = TFTrainingArguments(
        output_dir=os.path.join(configs.OUTPUT_PATH, task_name),
        do_eval=True,
        num_train_epochs=20,  # total # of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,
        weight_decay=0.01,  # strength of weight decay
        logging_dir=os.path.join(
            configs.OUTPUT_PATH, task_name, "logs"
        ),  # directory for storing logs
        save_steps=5000,
        logging_strategy="steps",
        logging_steps=20,
        eval_steps=500,
        evaluation_strategy="steps",
    )
    with training_args.strategy.scope():
        model = load_model(task_name)
    trainer = TFTrainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=dev_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,
    )
    return trainer


if __name__ == "__main__":
    task_name = "tnews"
    train_dataset, dev_dataset, test_dataset = load_dataset(task_name)
    train_dataset = cover_dataset2np(train_dataset)
    dev_dataset = cover_dataset2np(dev_dataset)
    test_dataset = cover_dataset2np(test_dataset, is_test=True)
    train_dataset = cover_np2tfds(*train_dataset)
    dev_dataset = cover_np2tfds(*dev_dataset)

    # optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    # model.compile(optimizer=optimizer, loss=model.compute_loss)
    # model.fit(train_dataset)
    trainer = init_trainer(train_dataset, dev_dataset, task_name)
    trainer.train()
    res = trainer.evaluate()
    print(res)
