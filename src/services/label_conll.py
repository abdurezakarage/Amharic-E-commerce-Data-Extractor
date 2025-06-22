# scripts/label_data_conll.py

import os

class CoNLLLabeler:
    def __init__(self, df, output_path, num_messages=50):
        """
        Args:
            df (pd.DataFrame): Preprocessed messages with 'id', 'tokens', and 'cleaned_text'.
            output_path (str): Full path to the output CoNLL file.
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
            print(f"Invalid label. Use: {', '.join(self.valid_labels[:-1])} or 'o', 'skip', 'quit'.")

    def _save_data(self):
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for sentence in self.labeled_data:
                for line in sentence:
                    f.write(line + '\n')
                f.write('\n')
        print(f"\n✅ Labeled data saved to: {self.output_path}")

    def label_messages(self):
        if self.df.empty:
            print("❌ No data available for labeling.")
            return

        print("\n--- Starting Interactive CoNLL Labeling ---")
        print("Instructions: Label tokens as B-*, I-* or O. Use 'skip' to skip, 'quit' to stop.")
        print("-" * 50)

        labeled_count = 0
        for _, row in self.df.iterrows():
            if labeled_count >= self.num_messages:
                break

            message_id = row.get('id')
            tokens = row.get('tokens', [])
            cleaned_text = row.get('cleaned_text', '')

            if not tokens or not isinstance(tokens, list):
                continue

            print(f"\n--- Message {labeled_count + 1} (ID: {message_id}) ---")
            print(f"Cleaned Text: {cleaned_text}")
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
                print(f"✅ Message {message_id} labeled. Total labeled: {labeled_count}")

        self._save_data()
        print(f"🎉 You labeled a total of {labeled_count} messages.")
