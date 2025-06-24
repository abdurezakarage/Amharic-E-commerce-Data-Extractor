import numpy as np
import pandas as pd
import shap
from seqeval.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from lime.lime_text import LimeTextExplainer

def load_conll_txt(filepath):
  
    sentences = []
    with open(filepath, 'r', encoding='utf-8') as f:
        tokens = []
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append(" ".join(tokens))
                    tokens = []
            else:
                splits = line.split()
                token = splits[0]
                tokens.append(token)
        if tokens:
            sentences.append(" ".join(tokens))
    return sentences

# Path to your sample CoNLL txt file
conll_file_path = "/content/drive/MyDrive/labeled_conll_output.txt"
sentences = load_conll_txt(conll_file_path)

print(f"Loaded {len(sentences)} sentences")
print(f"Sample sentence:\n{sentences[0]}")
model_path = "/content/drive/MyDrive/xlm-roberta-base_amharic_ner"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
model.eval()

label_map = model.config.id2label
print(f"Loaded model with labels: {list(label_map.values())}")

class NERPredictor:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    def __call__(self, texts):
        all_probs = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # Average probabilities across tokens for interpretability (simplification)
            avg_probs = probs.squeeze(0).mean(dim=0).cpu().numpy()
            all_probs.append(avg_probs)
        return np.array(all_probs)

predictor = NERPredictor(model, tokenizer)

explainer = shap.Explainer(predictor, tokenizer)

# Choose a sample sentence (e.g. 5th sentence)
sample_text = sentences[5]

shap_values = explainer([sample_text])
shap.plots.text(shap_values[0])

explainer_lime = LimeTextExplainer(class_names=list(label_map.values()))

def lime_predict(texts):
    results = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1).squeeze(0).mean(dim=0)
        results.append(probs.cpu().numpy())
    return np.array(results)

exp = explainer_lime.explain_instance(
    sample_text,
    lime_predict,
    num_features=10,
    top_labels=1
)

exp.show_in_notebook()


def export_shap_report_fixed(text, shap_value_obj, filename="shap_report.csv"):
    # Flatten data
    tokens = shap_value_obj.data
    # shap_value_obj.values shape = (num_tokens, num_classes)
    contributions = shap_value_obj.values

    if isinstance(contributions[0], np.ndarray):
        # Multi-class: take max contributing class or average
        contributions = [np.mean(c) for c in contributions]  # or use np.max(c)

    df = pd.DataFrame({
        'token': tokens,
        'contribution': contributions
    })
    df.to_csv(filename, index=False)
    print(f"✅ SHAP report saved to {filename}")

export_shap_report_fixed(sample_text, shap_values[0])
def get_predictions(sentences):
    preds = []
    for sent in sentences:
        inputs = tokenizer(sent, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1).squeeze(0).tolist()
        pred_labels = [label_map.get(i, "O") for i in pred_ids][1:-1]  # ignore special tokens
        preds.append(pred_labels)
    return preds

load true labels from CoNLL file
true_labels = "/content/drive/MyDrive/labeled_conll_output.txt"

predicted_labels = get_predictions(sentences)
print(classification_report(true_labels, predicted_labels))