# amharic_ner_model_comparison.py
import os
import time
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer
)
from seqeval.metrics import classification_report

# ✅ Load CoNLL format Amharic NER data
def read_conll(file_path):
    data = []
    with open(file_path, encoding='utf-8') as f:
        tokens, labels = [], []
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    data.append((tokens, labels))
                    tokens, labels = [], []
            else:
                splits = line.split()
                if len(splits) == 2:
                    token, label = splits
                    tokens.append(token)
                    labels.append(label)
        if tokens:
            data.append((tokens, labels))
    return pd.DataFrame(data, columns=["tokens", "ner_tags"])

# ✅ Label encoding and tokenizer setup
def prepare_label_mappings(df):
    label_list = sorted({label for labels in df['ner_tags'] for label in labels})
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}
    return label_list, label2id, id2label

# ✅ Tokenization function
def tokenize_and_align_labels(example, tokenizer, label2id):
    tokenized_input = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        padding='max_length',
        truncation=True,
        max_length=128
    )
    word_ids = tokenized_input.word_ids()
    aligned_labels = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(-100)
        elif word_idx != previous_word_idx:
            aligned_labels.append(label2id[example["ner_tags"][word_idx]])
        else:
            aligned_labels.append(-100)
        previous_word_idx = word_idx
    tokenized_input["labels"] = aligned_labels
    return tokenized_input

# ✅ Fine-tuning and evaluation pipeline
def fine_tune_and_evaluate(model_name, dataset, label_list, label2id, id2label):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_dataset = dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer, label2id))

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=-1)
        true_labels = [[id2label[l] for l in example if l != -100] for example in labels]
        true_preds = [[id2label[p] for (p, l) in zip(pred, label) if l != -100]
                      for pred, label in zip(predictions, labels)]
        report = classification_report(true_labels, true_preds, output_dict=True)
        return {
            "precision": report["micro avg"]["precision"],
            "recall": report["micro avg"]["recall"],
            "f1": report["micro avg"]["f1-score"]
        }

    args = TrainingArguments(
        output_dir=f"./{model_name.replace('/', '_')}_NER",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        logging_steps=10,
        report_to="none",
        logging_dir=f"./logs/{model_name.replace('/', '_')}"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    return trainer.evaluate()

# ✅ Run full experiment
def run_comparison(conll_file):
    df = read_conll(conll_file)
    label_list, label2id, id2label = prepare_label_mappings(df)
    dataset = Dataset.from_pandas(df)

    models_to_test = [
        "xlm-roberta-base",
        "distilbert-base-multilingual-cased",
        "bert-base-multilingual-cased",
        "Davlan/afroxlmr-base",
        "Davlan/bert-base-amharic"
    ]

    results = {}
    for model_name in models_to_test:
        print(f"\nEvaluating: {model_name}")
        results[model_name] = fine_tune_and_evaluate(model_name, dataset, label_list, label2id, id2label)

    results_df = pd.DataFrame(results).T.sort_values("f1", ascending=False)
    print("\n\nModel Comparison Results:\n")
    print(results_df)
    results_df.to_csv("ner_model_comparison_results.csv")

# Example Usage:
# run_comparison("labeled_conll_output.txt")
