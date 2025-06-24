
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

df = pd.read_csv('/content/drive/MyDrive/processed_telegram_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

print(f"Loaded {len(df)} posts")
df.head()



model_path = "/content/drive/MyDrive/xlm-roberta-base_amharic_ner"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
model.eval()

ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
    # device=0  # uncomment if GPU is available
)

def predict_entities(text):
    if not isinstance(text, str) or text.strip() == "":
        return []
    try:
        return ner_pipeline(text)
    except Exception as e:
        print(f"Error on text: {text[:30]}..., {e}")
        return []

df['predicted_entities'] = df['cleaned_message'].apply(predict_entities)



def extract_prices(entities):
    prices = []
    for ent in entities:
        if ent.get('entity_group') == 'PRICE':
            # Extract digits from entity word (e.g. "1200 ብር" -> 1200)
            price_str = ent.get('word', '')
            digits = ''.join(filter(str.isdigit, price_str))
            if digits:
                try:
                    prices.append(float(digits))
                except:
                    continue
    return prices

def vendor_metrics(df_vendor):
    min_date = df_vendor['Date'].min()
    max_date = df_vendor['Date'].max()
    weeks_active = max((max_date - min_date).days / 7, 1)

    posts_per_week = len(df_vendor) / weeks_active
    avg_views = df_vendor['views'].mean() if 'views' in df_vendor.columns else 0

    top_post_idx = df_vendor['views'].idxmax() if 'views' in df_vendor.columns else None
    if top_post_idx is not None:
        top_post = df_vendor.loc[top_post_idx]
        top_entities = top_post['predicted_entities']
        top_product = next((e['word'] for e in top_entities if e.get('entity_group') == 'PRODUCT'), "N/A")
        top_price = next((e['word'] for e in top_entities if e.get('entity_group') == 'PRICE'), "N/A")
    else:
        top_product, top_price = "N/A", "N/A"

    all_prices = df_vendor['predicted_entities'].apply(extract_prices)
    flat_prices = [price for sublist in all_prices for price in sublist]
    avg_price = np.mean(flat_prices) if flat_prices else 0

    return {
        'Posts/Week': round(posts_per_week, 2),
        'Avg Views/Post': round(avg_views, 2),
        'Top Product': top_product,
        'Top Price': top_price,
        'Avg Price (ETB)': round(avg_price, 2)
    }

vendors = df['Channel Title'].unique()
results = []

for vendor in vendors:
    vendor_df = df[df['Channel Title'] == vendor]
    metrics = vendor_metrics(vendor_df)
    metrics['Vendor'] = vendor
    results.append(metrics)

scorecard_df = pd.DataFrame(results)

# Lending Score: weighted sum of Avg Views and Posts/Week
scorecard_df['Lending Score'] = (
    scorecard_df['Avg Views/Post'] * 0.5 +
    scorecard_df['Posts/Week'] * 0.5
).round(2)

scorecard_df = scorecard_df[
    ['Vendor', 'Avg Views/Post', 'Posts/Week', 'Avg Price (ETB)', 'Lending Score', 'Top Product', 'Top Price']
].sort_values(by='Lending Score', ascending=False)

scorecard_df.head()

scorecard_df.to_csv('/content/drive/MyDrive/vendor_scorecard.csv', index=False)
print("Vendor scorecard saved to vendor_scorecard.csv")