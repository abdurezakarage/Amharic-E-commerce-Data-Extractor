import pandas as pd
from datasets import Dataset, Features, Value, ClassLabel, Sequence
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score


class AmharicTrainer:
    def __init__(self, model_name="Davlan/afro-xlmr-base"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.id_to_label = None
        self.label_to_id = None
        self.train_dataset = None
        self.eval_dataset = None

    @staticmethod
    def parse_conll(file_path):
        """
        Parses a CoNLL formatted file into a list of dictionaries,
        where each dictionary represents a sentence with its tokens and NER tags.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        sentences = []
        current_tokens = []
        current_labels = []

        for line in lines:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    token, label = parts
                    current_tokens.append(token)
                    current_labels.append(label)
            else:
                if current_tokens:
                    sentences.append({"tokens": current_tokens, "ner_tags": current_labels})
                    current_tokens = []
                    current_labels = []
        if current_tokens:
            sentences.append({"tokens": current_tokens, "ner_tags": current_labels})
        return sentences

    def _prepare_data(self, file_path):
        """
        Loads and preprocesses the data:
        1. Parses the CoNLL file.
        2. Creates label mappings (id_to_label, label_to_id).
        3. Converts string labels to numerical IDs.
        4. Splits the dataset into training and evaluation sets.
        5. Casts datasets to include ClassLabel features for NER tags.
        """
        data = self.parse_conll(file_path)

        all_labels = sorted(list(set(label for sentence in data for label in sentence['ner_tags'])))
        self.label_to_id = {label: i for i, label in enumerate(all_labels)}
        self.id_to_label = {i: label for i, label in enumerate(all_labels)}

        # Convert string labels to numerical IDs
        for sentence in data:
            sentence['ner_tags'] = [self.label_to_id[label] for label in sentence['ner_tags']]

        dataset = Dataset.from_list(data)

        # Split the dataset into training and validation
        train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset_raw = train_test_split['train']
        eval_dataset_raw = train_test_split['test']

        # Define the features with ClassLabel for ner_tags
        features_with_classlabel = Features({
            "tokens": Sequence(Value(dtype="string")),
            "ner_tags": Sequence(ClassLabel(names=all_labels))
        })
        self.train_dataset = train_dataset_raw.cast(features_with_classlabel)
        self.eval_dataset = eval_dataset_raw.cast(features_with_classlabel)

    def _tokenize_and_align_labels(self, examples):
        """
        Tokenizes inputs and aligns labels with the new tokens.
        Handles special tokens and subword tokenization by assigning -100 to non-label tokens.
        """
        tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)  # Special token (padding, BOS, EOS)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(label[word_idx] if self.id_to_label[label[word_idx]].startswith("I-") else -100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def compute_metrics(self, p):
        """
        Computes and returns evaluation metrics (f1, precision, recall) for NER.
        Ignores -100 labels during metric calculation.
        """
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (where label is -100)
        true_labels = [[self.id_to_label[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [self.id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        # Use seqeval for F1, precision, recall
        f1 = f1_score(true_labels, true_predictions)
        precision = precision_score(true_labels, true_predictions)
        recall = recall_score(true_labels, true_predictions)

        return {
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    def train_and_evaluate(self, file_path, output_dir="./ner_model_afro_xlmr_base",
                           num_train_epochs=3, learning_rate=2e-5,
                           per_device_train_batch_size=8, per_device_eval_batch_size=8,
                           weight_decay=0.01, logging_steps=50):
        """
        Orchestrates the entire training and evaluation pipeline.
        """
        print("Step 1: Preparing data...")
        self._prepare_data(file_path)
        print("Data preparation complete.")

        print("Step 2: Tokenizing and aligning labels for datasets...")
        tokenized_train_dataset = self.train_dataset.map(
            self._tokenize_and_align_labels, batched=True, remove_columns=['tokens', 'ner_tags']
        )
        tokenized_eval_dataset = self.eval_dataset.map(
            self._tokenize_and_align_labels, batched=True, remove_columns=['tokens', 'ner_tags']
        )
        print("Tokenization and alignment complete.")

        print("Step 3: Loading model and setting up training arguments...")
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.id_to_label),
            id2label=self.id_to_label,
            label2id=self.label_to_id
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            push_to_hub=False,
            logging_dir='./logs',
            logging_steps=logging_steps,
            report_to="none",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
        )

        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        print("Model and Trainer setup complete.")

        print("Step 4: Starting training...")
        trainer.train()
        print("Training complete.")

        print("Step 5: Evaluating the fine-tuned model...")
        results = trainer.evaluate()
        print(f"Evaluation results: {results}")
        return results, trainer

    def save_model(self, trainer, save_path="./fine_tuned_afro_xlmr_base_ner"):
        """
        Saves the trained model and tokenizer to the specified path.
        """
        print(f"Step 6: Saving model to {save_path}...")
        trainer.save_model(save_path)
        self.tokenizer.save_pretrained(save_path)
        print("Model saved successfully.")