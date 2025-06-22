# scripts/label_data_conll.py

import os

class CoNLLLabeler:
    def __init__(self, df, output_path, num_messages=50):
        """
        Args:
            df (pd.DataFrame): Must include 'cleaned_message' column with Amharic text.
            output_path (str): File path to save CoNLL-formatted labeled data.
            num_messages (int): Number of messages to label.
        """
        self.df = df
        self.output_path = output_path
        self.num_messages = num_messages
        self.valid_labels = ['b-product', 'i-product', 'b-loc', 'i-loc', 'b-price', 'i-price', 'o', '']
        self.labeled_data = []

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def _prompt_label(self, token, index, total):
        while True:
            label_input = input(f"Token '{token}' [{index + 1}/{total}]: ").strip().lower()
            if label_input == 'quit':
                return 'quit'
            if label_input == 'skip':
                return 'skip'
            if label_input in self.valid_labels:
                return label_input if label_input else 'o'
            print(f"Invalid label. Use: {', '.join(self.valid_labels[:-1])} or 'o' (Enter), 'skip', 'quit'.")

    def _save_data(self):
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for sentence in self.labeled_data:
                for line in sentence:
                    f.write(line + '\n')
                f.write('\n')
        print(f"\n✅ Labeled data saved in CoNLL format to: {self.output_path}")

    def label_messages(self):
        if self.df.empty or 'cleaned_message' not in self.df.columns:
            print("❌ DataFrame must contain a 'cleaned_message' column.")
            return

        print("\n--- Starting CoNLL NER Labeling ---")
        print("Label types: B-Product, I-Product, B-LOC, I-LOC, B-PRICE, I-PRICE, O")
        print("Instructions:")
        print("- Press Enter for 'O' (outside).")
        print("- Type 'skip' to skip a message.")
        print("- Type 'quit' to save and exit.")
        print("-" * 50)

        labeled_count = 0

        for _, row in self.df.iterrows():
            if labeled_count >= self.num_messages:
                break

            message_text = str(row.get('cleaned_message', '')).strip()
            if not message_text:
                continue

            tokens = message_text.split()  # Basic whitespace tokenization
            if not tokens:
                continue

            print(f"\n--- Message {labeled_count + 1} ---")
            print(f"Cleaned Message: {message_text}")
            print(f"Tokens: {tokens}")
            print("-" * 30)

            current_labels = []
            skip = False

            for i, token in enumerate(tokens):
                label = self._prompt_label(token, i, len(tokens))
                if label == 'quit':
                    self._save_data()
                    return
                if label == 'skip':
                    skip = True
                    print("Message skipped.")
                    break
                current_labels.append(f"{token}\t{label.upper()}")

            if not skip:
                self.labeled_data.append(current_labels)
                labeled_count += 1
                print(f"✅ Message labeled. Total labeled so far: {labeled_count}")

        self._save_data()
        print(f"\n🎉 Finished! You labeled a total of {labeled_count} messages.")
