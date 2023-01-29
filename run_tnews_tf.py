import configs
import os
import json

import numpy as np
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    TFTrainer,
    TFTrainingArguments,
    TFAutoModelForSequenceClassification,
    AutoConfig,
)
from transformers import XLNetTokenizer


def get_label_config(task_name: str):
    lines = open(os.path.join(configs.DATA_PATH, task_name, "labels.json")).readlines()
    labels = [json.loads(line)["label_desc"] for line in lines]
    label2id = {}
    id2label = {}
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label
    return {
        "label2id": label2id,
        "id2label": id2label,
        "num_labels": len(lines),
    }


def load_json2list(filename, task_name):
    label_configs = get_label_config(task_name)
    label2id = label_configs["label2id"]
    with open(filename) as f:
        for line in f.readlines():
            if line:
                data = json.loads(line)
                if data.get("label_desc"):
                    data["label"] = label2id[data["label_desc"]]
                yield data


def load_dataset(task_name: str):
    path = os.path.join(configs.DATA_PATH, task_name)

    train_file = os.path.join(path, "train.json")
    dev_file = os.path.join(path, "dev.json")
    test_file = os.path.join(path, "test.json")
    train_dataset = list(load_json2list(train_file, task_name))
    dev_dataset = list(load_json2list(dev_file, task_name))
    test_dataset = list(load_json2list(test_file, task_name))
    return train_dataset, dev_dataset, test_dataset


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
        labels = [item["label"] for item in dataset]

        labels = np.array(labels)
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
        tf.data.Dataset.from_tensor_slices((inputs, labels)).shuffle(
            inputs.shape[0], reshuffle_each_iteration=False
        )
    )


def load_model(task_name: str) -> TFAutoModelForSequenceClassification:
    label_config = get_label_config(task_name)
    model_config = AutoConfig.from_pretrained(configs.MODEL_PATH, **label_config)
    model = TFAutoModelForSequenceClassification.from_pretrained(
        configs.MODEL_PATH, config=model_config
    )
    return model


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def init_trainer(train_dataset, dev_dataset, task_name):
    training_args = TFTrainingArguments(
        output_dir=os.path.join(configs.OUTPUT_PATH, task_name),
        do_eval=True,
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(
            configs.OUTPUT_PATH, task_name, "logs"
        ),
        save_steps=5000,
        logging_strategy="steps",
        logging_steps=50,
        eval_steps=500,
        evaluation_strategy="steps",
    )
    with training_args.strategy.scope():
        model = load_model(task_name)
    trainer = TFTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
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

    trainer = init_trainer(train_dataset, dev_dataset, task_name)
    trainer.train()
    res = trainer.evaluate()
    print(res)
