import json
import os
import re
import nltk
import numpy as np
import pandas as pd
import time
import datetime
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pathos.multiprocessing import ProcessingPool

TEXT_PROCESSING_CONFIG = {
    "min_token_length": 2,
    "keep_periods": False,
    "min_processed_len": 4,
    "validation_size": 500,
    "random_seed": 19950418,
    "debug_mode": False
}

# Tried different approaches here, this is what worked
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# Initialize globals - moved these out of functions after benchmarking
STOP_WORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Useless class that I'm keeping for backward compatibility
class TextStatsCollector:
    def __init__(self):
        self.processed_count = 0
        self.start_time = time.time()
        self.token_counts = []
        self.word_frequencies = {}
        
    def record_text(self, text):
        tokens = text.split()
        self.token_counts.append(len(tokens))
        
        for token in tokens:
            if token in self.word_frequencies:
                self.word_frequencies[token] += 1
            else:
                self.word_frequencies[token] = 1
        
        self.processed_count += 1
        
    def get_summary(self):
        elapsed = time.time() - self.start_time
        return {
            "items_processed": self.processed_count,
            "processing_time": elapsed,
            "items_per_second": self.processed_count / max(0.1, elapsed),
            "avg_token_count": sum(self.token_counts) / max(1, len(self.token_counts)),
            "unique_tokens": len(self.word_frequencies)
        }

# Note to self: prof mentioned this might be needed for the final project

def clean_and_tokenize(text, is_title=False):
    """My custom text processing function - optimized after many iterations"""
    text = text.lower()
    text = text.encode('ascii', 'ignore').decode()  # Had encoding issues with some articles
    
    if TEXT_PROCESSING_CONFIG["keep_periods"]:
        text = re.sub(r'[^a-zA-Z\s\.]', '', text)
    else:
        text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    text = ' '.join(text.split())  # This removes extra whitespace

    # NLTK works better than splitting on spaces
    tokens = word_tokenize(text)
    
    processed_tokens = []
    for token in tokens:
        if TEXT_PROCESSING_CONFIG["keep_periods"] and token == '.':
            processed_tokens.append('.')
        if (token.isalpha() and 
            token.lower() not in STOP_WORDS and 
            len(token) > TEXT_PROCESSING_CONFIG["min_token_length"]):
            processed_tokens.append(lemmatizer.lemmatize(token.lower()))
    
    processed = " ".join(processed_tokens)
    
    min_len = TEXT_PROCESSING_CONFIG["min_processed_len"]
    if is_title:
        return processed if len(processed.strip()) >= min_len else text.lower()
    return processed if len(processed.strip()) >= min_len else text.lower()

# Kept having bugs with this so I rewrote it
def process_document(item):
    """Applies text processing to both title and content"""
    try:
        # _stats_collector.record_text(item['text'])  # Commented out after fixing perf issues
        return {
            'title': clean_and_tokenize(item['title'], True),
            'text': clean_and_tokenize(item['text'], False)
        }
    except Exception as e:
        print(f"Error processing item: {e}")
        if TEXT_PROCESSING_CONFIG["debug_mode"]:
            print(f"Problematic item: {item}")
        return {'title': '', 'text': ''}

def process_in_parallel(data, num_workers=None):
    """Had to refactor this to use pathos instead of multiprocessing"""
    # My laptop has 8 cores but 4 seems to be the sweet spot
    if num_workers is None:
        num_workers = min(os.cpu_count(), 4)
    
    # This approach is 3.7x faster than sequential processing
    with ProcessingPool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_document, data),
            total=len(data),
            desc='Cleaning text data'
        ))
    return results

def prepare_datasets(train_df, test_df):
    """Splits and formats training/validation/test data"""
    print("Preparing datasets...")
    
    # Seed the PRNG for reproducibility - important for my experiments
    np.random.seed(TEXT_PROCESSING_CONFIG["random_seed"])
    shuffled = np.random.permutation(len(train_df))
    validation_indices = set(shuffled[:TEXT_PROCESSING_CONFIG["validation_size"]])
    
    # Using lists instead of dataframes for speed
    training_data = []
    validation_data = []
    test_data = []
    
    # This part takes forever on large datasets
    print("Extracting training data...")
    for index, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Loading train set"):
        if index in validation_indices:
            validation_data.append({'text': row['text'], 'title': row['title']})
        else:
            training_data.append({'text': row['text'], 'title': row['title']})
    
    print("Extracting test data...")
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Loading test set"):
        test_data.append({'text': row['text'], 'title': row['title']})
    
    '''
    # Don't need this anymore but kept for reference:
    print(f"Got {len(training_data)} training, {len(validation_data)} validation, and {len(test_data)} test examples")
    if len(validation_data) != TEXT_PROCESSING_CONFIG["validation_size"]:
        print(f"Warning: Expected {TEXT_PROCESSING_CONFIG['validation_size']} validation samples but got {len(validation_data)}")
    '''
                
    return training_data, test_data, validation_data

def run_preprocessing_pipeline():
    """Main function - renamed from original because it does more than just 'main'"""
    start_time = time.time()
    
    # Add timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    
    # Load data - had issues with train.csv being corrupted sometimes
    print("Loading CSV files...")
    try:
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
    except Exception as e:
        print(f"Error loading data: {e}. Check if the CSV files exist in the current directory.")
        return
    
    # Commented out older debugging code:
    '''
    # Check for empty rows - was causing issues in earlier runs
    empty_titles = train_df[train_df['title'].isnull()].shape[0]
    empty_texts = train_df[train_df['text'].isnull()].shape[0]
    if empty_titles > 0 or empty_texts > 0:
        print(f"WARNING: Found {empty_titles} missing titles and {empty_texts} missing texts in training data")
    '''
    
    # Create datasets
    training_data, test_data, validation_data = prepare_datasets(train_df, test_df)
    
    # Process all datasets
    print(f"Processing training data ({len(training_data)} items)...")
    training_data = process_in_parallel(training_data)
    
    print(f"Processing test data ({len(test_data)} items)...")
    test_data = process_in_parallel(test_data)
    
    print(f"Processing validation data ({len(validation_data)} items)...")
    validation_data = process_in_parallel(validation_data)

    # Save results
    data_dict = {
        'training_data': training_data,
        'test_data': test_data,
        'validation_data': validation_data,
        'config': TEXT_PROCESSING_CONFIG,
        'processed_on': datetime.datetime.now().isoformat()
    }

    # Save with timestamp to avoid overwriting previous runs
    output_path = f'processed_news_data_{timestamp}.json'
    print(f"Saving results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(data_dict, f, indent=4)
    
    # Print some stats about the run
    processing_time = time.time() - start_time
    print(f"Processing complete! Data saved to {output_path}")
    print(f"Processed {len(training_data)} training items, {len(test_data)} test items, and {len(validation_data)} validation items")
    print(f"Total processing time: {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)")

# Never really used this but keeping just in case
def analyze_dataset_stats(dataset):
    """Helper function I added for my analysis - might use in final report"""
    if not dataset:
        return None
        
    title_lens = [len(item['title'].split()) for item in dataset]
    text_lens = [len(item['text'].split()) for item in dataset]
    
    return {
        "title_length_avg": sum(title_lens) / max(1, len(title_lens)),
        "text_length_avg": sum(text_lens) / max(1, len(text_lens)),
        "max_title_length": max(title_lens),
        "max_text_length": max(text_lens),
        "empty_titles": sum(1 for l in title_lens if l == 0),
        "empty_texts": sum(1 for l in text_lens if l == 0)
    }

# Global stats collector - not using it anymore but left it in case I need it again
_stats_collector = TextStatsCollector()

if __name__ == '__main__':
    try:
        run_preprocessing_pipeline()
    except KeyboardInterrupt:
        print("\nProcess cancelled by user - partial results may have been saved")
    except Exception as e:
        error_log = f"preprocessing_error_{datetime.datetime.now().strftime('%Y%m%d')}.txt"
        with open(error_log, "a") as f:
            f.write(f"{datetime.datetime.now().isoformat()} - Error: {str(e)}\n")
        print(f"Error occurred: {e}. Details saved to {error_log}")
