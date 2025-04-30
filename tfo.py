import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from google.colab import drive
drive.mount('/content/drive')

def _safe_read_csv(path):
    """Safely read a CSV file and return a DataFrame"""
    try:
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("CSV is empty")
        return df
    except FileNotFoundError:
        print(f"File not found: {path}")
        raise
    except Exception as e:
        print(f"Error reading {path}: {e}")
        raise

def _convert_row(row):
    """Convert a DataFrame row to dictionary"""
    return {
        'text': row['text'] if 'text' in row else '',
        'title': row['title'] if 'title' in row else ''
    }

def _generate_validation_indices(size, val_size=500, seed=42):
    """Return a set of validation indices"""
    np.random.seed(seed)
    return set(np.random.permutation(size)[:val_size])

def _split_train_valid(df, val_indices):
    """Split dataframe into training and validation sets"""
    train_data, val_data = [], []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Splitting train/valid"):
        item = _convert_row(row)
        (val_data if idx in val_indices else train_data).append(item)
    return train_data, val_data

def _extract_test_data(df):
    """Extract test data"""
    test_data = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading test data"):
        test_data.append(_convert_row(row))
    return test_data

def preprocess_data(train_df, test_df, seed=42):
    """Preprocess the data and return splits"""
    print("Initializing data preprocessing...")
    val_indices = _generate_validation_indices(len(train_df), seed=seed)
    train_data, valid_data = _split_train_valid(train_df, val_indices)
    test_data = _extract_test_data(test_df)
    return train_data, test_data, valid_data

def _write_json(obj, path):
    """Write object to JSON file"""
    try:
        with open(path, 'w') as f:
            json.dump(obj, f, indent=4)
    except Exception as e:
        print(f"Failed to write JSON to {path}: {e}")

def run_pipeline():
    """Main function to run the data pipeline"""
    train_path = '/content/drive/My Drive/train.csv'
    test_path = '/content/drive/My Drive/test.csv'

    print("Reading input files...")
    try:
        train_df = _safe_read_csv(train_path)
        test_df = _safe_read_csv(test_path)
    except Exception:
        print("Aborting due to data loading error.")
        return

    print("Processing dataset...")
    train_data, test_data, valid_data = preprocess_data(train_df, test_df)

    output = {
        'training_data': train_data,
        'test_data': test_data,
        'validation_data': valid_data
    }

    output_path = 'processed_data.json'
    print(f"Saving processed data to {output_path}...")
    _write_json(output, output_path)

    print("Done.")
    print(f"Train: {len(train_data)}, Test: {len(test_data)}, Validation: {len(valid_data)}")

if __name__ == '__main__':
    run_pipeline()
    
    
    
import torch
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from rouge_score import rouge_scorer
from tqdm import tqdm
import json

model_path = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

with open("processed_data.json") as f:
    data_bundle = json.load(f)

def _make_hf_dataset(data_list, tokenizer, max_input_len=512, max_target_len=64):
    source_texts = ["summarize: " + item['text'] for item in data_list]
    target_texts = [item['title'] for item in data_list]

    inputs = tokenizer(source_texts, max_length=max_input_len, truncation=True, padding="max_length")
    targets = tokenizer(target_texts, max_length=max_target_len, truncation=True, padding="max_length")

    data_dict = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "labels": [
            [(token if token != tokenizer.pad_token_id else -100) for token in label]
            for label in targets.input_ids
        ]
    }

    return Dataset.from_dict(data_dict)

train_set = _make_hf_dataset(data_bundle["training_data"], tokenizer)
valid_set = _make_hf_dataset(data_bundle["validation_data"], tokenizer)
test_set = _make_hf_dataset(data_bundle["test_data"], tokenizer)

collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

def _evaluate_rouge(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    score_sum = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    for pred, true in zip(decoded_preds, decoded_labels):
        result = rouge.score(true, pred)
        score_sum['rouge1'] += result['rouge1'].fmeasure
        score_sum['rouge2'] += result['rouge2'].fmeasure
        score_sum['rougeL'] += result['rougeL'].fmeasure

    return {k: v / len(decoded_preds) for k, v in score_sum.items()}

args = Seq2SeqTrainingArguments(
    output_dir="./t5-headline-generator",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=5,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    logging_steps=100,
    push_to_hub=False
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_set,
    eval_dataset=valid_set,
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=_evaluate_rouge
)

trainer.train()

def _predict_and_score(model, tokenizer, data_list, use_beam=False, beam_n=4):
    model.eval()
    predictions = []
    ground_truths = [item['title'] for item in data_list]
    batch_size = 16

    for i in tqdm(range(0, len(data_list), batch_size), desc="Generating"):
        batch = data_list[i:i+batch_size]
        texts = ["summarize: " + item['text'] for item in batch]
        tokens = tokenizer(texts, max_length=512, truncation=True, padding=True, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **tokens,
            max_length=64,
            num_beams=beam_n if use_beam else 1,
            early_stopping=True
        )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)

    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    totals = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    for pred, ref in zip(predictions, ground_truths):
        result = rouge.score(ref, pred)
        totals['rouge1'] += result['rouge1'].fmeasure
        totals['rouge2'] += result['rouge2'].fmeasure
        totals['rougeL'] += result['rougeL'].fmeasure

    average = {k: v / len(predictions) for k, v in totals.items()}

    search_type = "Beam Search" if use_beam else "Greedy"
    print(f"\n{search_type} Results:")
    for key, value in average.items():
        print(f"{key}: {value:.4f}")

    return predictions, average

if __name__ == "__main__":
    greedy_output, greedy_scores = _predict_and_score(model, tokenizer, data_bundle["test_data"], use_beam=False)

    beam_output, beam_scores = _predict_and_score(model, tokenizer, data_bundle["test_data"], use_beam=True, beam_n=4)

    print("\nResults Comparison:")
    print(f"{'Metric':<10} {'Greedy':<10} {'Beam Search':<12}")
    print("-" * 34)
    for metric in ['rouge1', 'rouge2', 'rougeL']:
        print(f"{metric:<10} {greedy_scores[metric]:.4f}      {beam_scores[metric]:.4f}")
        
    from itertools import product



if __name__ == "__main__":
    learning_rates = [3e-5, 5e-5]
    weight_decays = [0.0, 0.01]
    num_epochs = [3, 5]

    results = []

    for lr, wd, ep in product(learning_rates, weight_decays, num_epochs):
        print(f"\nTraining with LR={lr}, WD={wd}, Epochs={ep}")

        training_args = Seq2SeqTrainingArguments(
            output_dir=f"./t5-gridsearch-lr{lr}-wd{wd}-ep{ep}",
            evaluation_strategy="epoch",
            learning_rate=lr,
            weight_decay=wd,
            num_train_epochs=ep,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            save_total_limit=1,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
            logging_strategy="no",
            logging_dir=None,
            push_to_hub=False
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_set,
            eval_dataset=valid_set,
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=_evaluate_rouge
        )

        trainer.train()

        eval_metrics = trainer.evaluate()
        rouge_l = eval_metrics.get("eval_rougeL", 0.0)
        results.append({
            "learning_rate": lr,
            "weight_decay": wd,
            "num_epochs": ep,
            "rougeL": rouge_l
        })

    results = sorted(results, key=lambda x: x["rougeL"], reverse=True)
    print("\nBest Hyperparameter Set:")
    print(json.dumps(results[0], indent=4))
    

import pandas as pd
import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from rouge_score import rouge_scorer
import numpy as np
import time
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

def get_timestamp():
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def convert_csv_to_json(csv_path, json_path):
    print(f"Starting conversion at {get_timestamp()}...")
    
    data = pd.read_csv(csv_path)
    data.to_json(json_path, orient='records', lines=True)
    print(f"Converted {csv_path} to {json_path}")
    
    row_count = len(data)
    
    return json_path

def load_articles_from_json(json_path):
    start_time = time.time()
    
    with open(json_path, 'r') as file:
        json_data = [json.loads(line) for line in file]

    possible_article_fields = ['article', 'body', 'content', 'text']
    possible_title_fields = ['title', 'headline', 'header']

    article_field = next((f for f in possible_article_fields if f in json_data[0]), None)
    title_field = next((f for f in possible_title_fields if f in json_data[0]), None)

    if not article_field or not title_field:
        raise ValueError(f"Could not find article or title fields. Available: {list(json_data[0].keys())}")

    articles = [item[article_field] for item in json_data if article_field in item]
    reference_titles = [item[title_field] for item in json_data if title_field in item]

    load_time = time.time() - start_time
    if random.random() > 0.7:
        print(f"Loaded data in {load_time:.2f} seconds")

    return articles, reference_titles

def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text))

def generate_titles(articles, model, tokenizer, prompt_prefix, device):
    generated_titles = []
    start_time = time.time()
    
    total_chars = 0

    for i, article in enumerate(articles):
        if i % 10 == 0 and i > 0:
            print(f"  Processed {i}/{len(articles)} articles...")
        
        total_chars += len(article)
            
        prompt = prompt_prefix + article
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model.generate(**inputs, max_length=50, num_beams=5, early_stopping=True)
        generated_title = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_titles.append(generated_title)

    elapsed_time = time.time() - start_time
    print(f"  Generation completed in {elapsed_time:.2f} seconds for {len(articles)} articles")
    print(f"  Average time per article: {elapsed_time/len(articles):.2f} seconds")
    
    if random.random() > 0.8:
        avg_chars = total_chars / len(articles) if articles else 0
        print(f"  Debug: Avg article length: {avg_chars:.0f} chars")

    return generated_titles

def calculate_rouge_scores(reference_titles, generated_titles):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    word_match_count = 0

    for ref_title, gen_title in zip(reference_titles, generated_titles):
        scores = scorer.score(ref_title, gen_title)

        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
        
        ref_words = set(ref_title.lower().split())
        gen_words = set(gen_title.lower().split())
        word_match_count += len(ref_words.intersection(gen_words))

    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0

    return {
        'rouge1': avg_rouge1,
        'rouge2': avg_rouge2,
        'rougeL': avg_rougeL,
        'average': (avg_rouge1 + avg_rouge2 + avg_rougeL) / 3
    }

def print_sample_titles(references, generated, count=3):
    if len(references) < count or len(generated) < count:
        return
    
    for i in range(count):
        idx = random.randint(0, min(len(references), len(generated)) - 1)
        print(f"Example {i+1}:")
        print(f"  Reference: {references[idx]}")
        print(f"  Generated: {generated[idx]}")

def main():
    random_seed = random.randint(1, 100)
    random.seed(random_seed)
    
    csv_path = '/content/test.csv'
    json_path = '/content/test.json'

    model_names = ['google/flan-t5-base', 'google/flan-t5-large']

    prompt_variations = [
        "Generate a concise title for the following article: ",
        "Create an appropriate title based on this text: "
    ]
    
    max_article_length = 5000
    max_title_length = 50

    try:
        json_path = convert_csv_to_json(csv_path, json_path)

        articles, reference_titles = load_articles_from_json(json_path)
        print(f"Loaded {len(articles)} articles with reference titles")
        
        if articles and reference_titles:
            first_article_sample = articles[0][:100] + "..." if len(articles[0]) > 100 else articles[0]
            print(f"First article sample: {first_article_sample}")

        results = {}

        for model_name in model_names:
            print(f"\nProcessing model: {model_name}")
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            model = model.to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            if device.type == 'cuda':
                print(f"  GPU memory allocated: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
                print(f"  GPU memory reserved: {torch.cuda.memory_reserved() / (1024**2):.2f} MB")

            model_results = {}

            for prompt in prompt_variations:
                print(f"  Using prompt: \"{prompt}\"")

                generated_titles = generate_titles(articles, model, tokenizer, prompt, device)

                scores = calculate_rouge_scores(reference_titles, generated_titles)
                model_results[prompt] = scores

                print(f"    ROUGE-1: {scores['rouge1']:.4f}")
                print(f"    ROUGE-2: {scores['rouge2']:.4f}")
                print(f"    ROUGE-L: {scores['rougeL']:.4f}")
                print(f"    Average: {scores['average']:.4f}")
                
                if random.random() > 0.9:
                    print_sample_titles(reference_titles, generated_titles, 2)

            results[model_name] = model_results

            if device.type == 'cuda':
                del model
                torch.cuda.empty_cache()
                print(f"  GPU memory freed up")

        print("\n===== FINAL ROUGE SCORES BY PROMPT =====")

        for prompt in prompt_variations:
            print(f"\nPrompt: \"{prompt}\"")

            rouge1_total = sum(results[model][prompt]['rouge1'] for model in model_names)
            rouge2_total = sum(results[model][prompt]['rouge2'] for model in model_names)
            rougeL_total = sum(results[model][prompt]['rougeL'] for model in model_names)
            avg_total = sum(results[model][prompt]['average'] for model in model_names)

            for model in model_names:
                scores = results[model][prompt]
                print(f"  {model}:")
                print(f"    ROUGE-1: {scores['rouge1']:.4f}")
                print(f"    ROUGE-2: {scores['rouge2']:.4f}")
                print(f"    ROUGE-L: {scores['rougeL']:.4f}")
                print(f"    Average: {scores['average']:.4f}")

            model_count = len(model_names)
            print(f"  AVERAGE ACROSS MODELS:")
            print(f"    ROUGE-1: {rouge1_total/model_count:.4f}")
            print(f"    ROUGE-2: {rouge2_total/model_count:.4f}")
            print(f"    ROUGE-L: {rougeL_total/model_count:.4f}")
            print(f"    Average: {avg_total/model_count:.4f}")

    except Exception as e:
        print(f"Error: {str(e)}")
        
        import traceback
        if random.random() > 0.8:
            print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    start_time = time.time()
    print(f"Script started at: {get_timestamp()}")
    
    main()
    
    end_time = time.time()
    if random.random() > 0.5:
        print(f"Total execution time: {end_time - start_time:.2f} seconds")
