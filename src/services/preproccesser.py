# amharic_nlp_processor.py

import pandas as pd
import re
import emoji
import os
import glob
from etnltk import Amharic 

class TextProcessor:
    def __init__(self, raw_data_dir="../data/raw", processed_data_dir="../data/processed"):
        self.raw_data_dir = os.path.abspath(raw_data_dir) 
        self.processed_data_dir = os.path.abspath(processed_data_dir) 
        self._ensure_directories_exist()

    def _ensure_directories_exist(self):
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)

    def _get_csv_files_from_directory(self, directory=None):
        """Get all CSV files from the specified directory"""
        if directory is None:
            directory = self.raw_data_dir
        
        csv_pattern = os.path.join(directory, "*.csv")
        csv_files = glob.glob(csv_pattern)
        return csv_files

    def _clean_and_normalize_amharic_text(self, text):
        if not isinstance(text, str):
            return ""

        # 1. Remove URLs, Telegram-specific elements (mentions, hashtags, channel links)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r't\.me/\S+', '', text)

        # 2. Handle emojis: convert to text descriptions and then remove
        text = emoji.demojize(text)
        text = re.sub(r':\S+:', '', text)

        try:
            amharic_doc = Amharic(text)
            cleaned_text = amharic_doc.clean
        except Exception as e:
            # print(f"Warning: etnltk processing failed for text: '{text[:50]}...'. Error: {e}. Falling back to basic regex cleaning.")
            amharic_and_latin_punctuation = r'[!"#$%&\'()*+,\-./:;<=>?@[\\\]^_`{|}~፡።፣፤፥፧፨]'
            cleaned_text = re.sub(amharic_and_latin_punctuation, '', text)
            ethiopic_numbers = r'[\u1369-\u137C]'
            cleaned_text = re.sub(r'\d+', '', cleaned_text)
            cleaned_text = re.sub(ethiopic_numbers, '', cleaned_text)

        # 4. Remove extra whitespaces after all cleaning
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        return cleaned_text

    def _amharic_word_tokenizer_etnltk(self, text):
        if not isinstance(text, str) or not text.strip():
            return []
        try:
            amharic_doc = Amharic(text)
            return amharic_doc.tokens
        except Exception as e:
            return [token for token in re.split(r'[፡\s]+', text) if token]

    def preprocess_data(self, input_csv_filename=None, output_csv_filename=None, message_column='message', process_directory=False):
        if process_directory:
            return self._preprocess_directory(message_column)
        else:
            # Validate required parameters for single file processing
            if input_csv_filename is None or output_csv_filename is None:
                raise ValueError("input_csv_filename and output_csv_filename are required when process_directory=False")
            return self._preprocess_single_file(input_csv_filename, output_csv_filename, message_column)

    def _preprocess_single_file(self, input_csv_filename, output_csv_filename, message_column='Message'):
        """Process a single CSV file"""
        input_path = os.path.join(self.raw_data_dir, input_csv_filename)
        output_path = os.path.join(self.processed_data_dir, output_csv_filename)

        try:
            df = pd.read_csv(input_path)
        except FileNotFoundError:
            print(f"Error: Input CSV file not found at {input_path}")
            return False
        except Exception as e:
            print(f"Error reading input CSV: {e}")
            return False

        # Check if message column exists (case-insensitive)
        available_columns = df.columns.tolist()
        message_column_found = None
        
        # First try exact match
        if message_column in available_columns:
            message_column_found = message_column
        else:
            # Try case-insensitive match
            for col in available_columns:
                if col.lower() == message_column.lower():
                    message_column_found = col
                    break
        
        if message_column_found is None:
            print(f"Error: Message column '{message_column}' not found in the CSV.")
            print(f"Available columns: {available_columns}")
            print("Trying to find similar column names...")
            
            # Suggest similar column names
            similar_columns = [col for col in available_columns if 'message' in col.lower() or 'text' in col.lower() or 'content' in col.lower()]
            if similar_columns:
                print(f"Similar columns found: {similar_columns}")
                print(f"Please use one of these column names or update your message_column parameter.")
            return False

        print(f"Using message column: '{message_column_found}'")

        # Separate metadata from message content
        metadata_columns = [col for col in df.columns if col != message_column_found]
        metadata_df = df[metadata_columns].copy()

        # Preprocess the message content
        print("Cleaning and normalizing Amharic text...")
        df['cleaned_message'] = df[message_column_found].apply(self._clean_and_normalize_amharic_text)

        print("Tokenizing Amharic text...")
        df['tokens'] = df['cleaned_message'].apply(self._amharic_word_tokenizer_etnltk)

        # Combine metadata and processed text into a new DataFrame
        processed_df = pd.concat([metadata_df, df[['cleaned_message', 'tokens']]], axis=1)

        # Save the preprocessed data
        try:
            processed_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"Preprocessed data successfully saved to: {output_path}")
            return True
        except Exception as e:
            print(f"Error saving preprocessed data to CSV: {e}")
            return False

    def _preprocess_directory(self, message_column='message'):
        """Process all CSV files in the raw data directory"""
        csv_files = self._get_csv_files_from_directory()
        
        if not csv_files:
            print(f"No CSV files found in directory: {self.raw_data_dir}")
            return False
        
        print(f"Found {len(csv_files)} CSV files to process:")
        for file in csv_files:
            print(f"  - {os.path.basename(file)}")
        
        successful_files = 0
        failed_files = 0
        
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            output_filename = f"processed_{filename}"
            
            print(f"\nProcessing: {filename}")
            
            # Create a temporary processor for this file
            temp_processor = TextProcessor(
                raw_data_dir=os.path.dirname(csv_file),
                processed_data_dir=self.processed_data_dir
            )
            
            success = temp_processor._preprocess_single_file(
                filename, 
                output_filename, 
                message_column
            )
            
            if success:
                successful_files += 1
            else:
                failed_files += 1
        
        print(f"\nProcessing complete!")
        print(f"Successfully processed: {successful_files} files")
        print(f"Failed to process: {failed_files} files")
        
        return successful_files > 0

    def get_processing_summary(self, directory=None):
        """Get a summary of files in the raw data directory"""
        if directory is None:
            directory = self.raw_data_dir
        
        csv_files = self._get_csv_files_from_directory(directory)
        
        summary = {
            'directory': directory,
            'total_csv_files': len(csv_files),
            'files': []
        }
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                file_info = {
                    'filename': os.path.basename(csv_file),
                    'rows': len(df),
                    'columns': list(df.columns),
                    'size_mb': os.path.getsize(csv_file) / (1024 * 1024)
                }
                summary['files'].append(file_info)
            except Exception as e:
                file_info = {
                    'filename': os.path.basename(csv_file),
                    'error': str(e)
                }
                summary['files'].append(file_info)
        
        return summary