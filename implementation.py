import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
import json
from tqdm import tqdm
import os


# === GLOBAL SETTINGS === #

MODEL_NAME = "facebook/nllb-200-distilled-600M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LANG_PT = "por_Latn"
LANG_TUPI = "por_Latn"

MAX_LENGTH = 128
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
NUM_EPOCHS = 10
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01


# === DATA READING FUNCTIONS === #

def format_str(text):
    """Remove external quotes that may have been added during export"""
    text = text.strip()
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    return text


def read_tsv(file_path):
    """Reads a TSV file and returns lists of sentences in Portuguese and Tupi"""
    strings_pt = []
    strings_tupi = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        next(f)  # Skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split("\t")
            if len(parts) >= 2:
                pt, tupi = parts[0], parts[1]
                strings_pt.append(format_str(pt))
                strings_tupi.append(format_str(tupi))
    
    return strings_pt, strings_tupi


def load_all_data():
    """Loads all datasets: train, valid, test"""
    test_pt, test_tupi = read_tsv("data/test.tsv")
    train_pt, train_tupi = read_tsv("data/train.tsv")
    valid_pt, valid_tupi = read_tsv("data/valid.tsv")
    
    print(f"Train: {len(train_pt)} pairs")
    print(f"Valid: {len(valid_pt)} pairs")
    print(f"Test: {len(test_pt)} pairs")
    
    return {
        'train': (train_pt, train_tupi),
        'valid': (valid_pt, valid_tupi),
        'test': (test_pt, test_tupi)
    }


# === EVALUATION METRICS === #

def count_ngrams(text, n):
    ngrams = {}
    for i in range(len(text) - n + 1):
        ng = text[i:i+n]
        ngrams[ng] = ngrams.get(ng, 0) + 1
    return ngrams


class BleuScorer:    
    def __init__(self, weights=[0.25, 0.25, 0.25, 0.25], eps=1e-12):
        self.weights = weights
        self.eps = eps
    
    def get_prob_bleu(self, reference, hypothesis, n):
        """Calculate p_n for a specific n-gram"""
        if len(hypothesis) < n:
            return 0.0
        
        ngrams_hyp = count_ngrams(hypothesis, n)
        ngrams_ref = count_ngrams(reference, n)
        
        clipped_count = 0
        total_count = 0
        
        for ngram, count in ngrams_hyp.items():
            clipped_count += min(count, ngrams_ref.get(ngram, 0))
            total_count += count
        
        return clipped_count / total_count if total_count > 0 else 0.0
    
    def brevity_penalty(self, reference, hypothesis):
        """Calculates brevity penalty"""
        r = len(reference)
        c = len(hypothesis)
        
        if c == 0:
            return 0.0
        
        return np.exp(-max(0.0, r/c - 1))
    
    def bleu_sentence(self, reference, hypothesis):
        """Calculates BLEU for a sentence pair"""
        n = len(self.weights)
        log_probs = []
        
        for i in range(1, n + 1):
            p_n = self.get_prob_bleu(reference, hypothesis, i)
            log_probs.append(np.log(max(p_n, self.eps)))
        
        bp = self.brevity_penalty(reference, hypothesis)
        weighted_log_prob = np.dot(log_probs, self.weights)
        
        return bp * np.exp(weighted_log_prob)
    
    def bleu_corpus(self, references, hypotheses):
        """Calculates average BLEU for a corpus"""
        scores = []
        for ref, hyp in zip(references, hypotheses):
            scores.append(self.bleu_sentence(ref, hyp))
        return np.mean(scores)


class ChrFScorer:    
    def __init__(self, beta=1.0, n=6):
        self.beta = beta
        self.n = n
    
    def chrf_sentence(self, reference, hypothesis):
        """Calculates chrF for a sentence pair"""
        total_match = 0
        total_hyp = 0
        total_ref = 0
        
        for k in range(1, self.n + 1):
            ngrams_ref = count_ngrams(reference, k)
            ngrams_hyp = count_ngrams(hypothesis, k)
            
            for ng, count_hyp in ngrams_hyp.items():
                total_match += min(count_hyp, ngrams_ref.get(ng, 0))
                total_hyp += count_hyp
            
            total_ref += sum(ngrams_ref.values())
        
        if total_hyp == 0 or total_ref == 0:
            return 0.0
        
        precision = total_match / total_hyp
        recall = total_match / total_ref
        
        beta_sq = self.beta ** 2
        
        if precision + recall == 0:
            return 0.0
        
        f_score = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
        return f_score
    
    def chrf_corpus(self, references, hypotheses):
        """Calculates average chrF for a corpus"""
        scores = []
        for ref, hyp in zip(references, hypotheses):
            scores.append(self.chrf_sentence(ref, hyp))
        return np.mean(scores)


def evaluate_translations(references, hypotheses):
    """Evaluates translations using multiple metrics"""
    bleu_scorer = BleuScorer()
    chrf1_scorer = ChrFScorer(beta=1.0, n=6)
    chrf3_scorer = ChrFScorer(beta=3.0, n=6)
    
    return {
        'BLEU': bleu_scorer.bleu_corpus(references, hypotheses),
        'chrF1': chrf1_scorer.chrf_corpus(references, hypotheses),
        'chrF3': chrf3_scorer.chrf_corpus(references, hypotheses),
    }


# === ZERO-SHOT TRANSLATION === #

class ZeroShotTranslator:    
    def __init__(self):
        print(f"Loading model {MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        self.model.to(DEVICE)
        self.model.eval()
        print(f"Model loaded on {DEVICE}")
    
    def translate(self, texts, src_lang, tgt_lang, batch_size=8):
        """Translates a list of texts"""
        translations = []
        
        # Set source and target languages
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
            batch = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH
            ).to(DEVICE)
            
            # Generate translations
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_lang),
                    max_length=MAX_LENGTH,
                    num_beams=5,
                    early_stopping=True
                )
            
            # Decode translations
            batch_translations = self.tokenizer.batch_decode(
                generated,
                skip_special_tokens=True
            )
            
            translations.extend(batch_translations)
        
        return translations


def run_zero_shot_experiments(data):
    """Runs zero-shot experiments in both directions"""
    print("Zero-shot translation:")
    
    translator = ZeroShotTranslator()
    results = {}
    
    test_pt, test_tupi = data['test']
    
    print("Portuguese -> Tupi")
    translations_pt_tupi = translator.translate(test_pt, LANG_PT, LANG_TUPI)
    metrics_pt_tupi = evaluate_translations(test_tupi, translations_pt_tupi)
    results['PT->Tupi'] = {
        'translations': translations_pt_tupi,
        'metrics': metrics_pt_tupi
    }
    
    print(f"BLEU: {metrics_pt_tupi['BLEU']:.4f}")
    print(f"chrF1: {metrics_pt_tupi['chrF1']:.4f}")
    print(f"chrF3: {metrics_pt_tupi['chrF3']:.4f}")
    
    print("Tupi -> Portuguese")
    translations_tupi_pt = translator.translate(test_tupi, LANG_TUPI, LANG_PT)
    metrics_tupi_pt = evaluate_translations(test_pt, translations_tupi_pt)
    results['Tupi->PT'] = {
        'translations': translations_tupi_pt,
        'metrics': metrics_tupi_pt
    }
    
    print(f"BLEU: {metrics_tupi_pt['BLEU']:.4f}")
    print(f"chrF1: {metrics_tupi_pt['chrF1']:.4f}")
    print(f"chrF3: {metrics_tupi_pt['chrF3']:.4f}")
    
    save_results(results, "results_zero_shot.json")
    save_examples(test_pt, test_tupi, translations_pt_tupi, translations_tupi_pt,
                  "examples_zero_shot.txt")
    
    return results


# === FEW-SHOT (FINE-TUNING) === #

def prepare_dataset_for_training(pt_texts, tupi_texts, tokenizer, src_lang, tgt_lang):    
    def preprocess_function(examples):

        # Tokenize inputs
        tokenizer.src_lang = src_lang
        inputs = tokenizer(
            examples['source'],
            max_length=MAX_LENGTH,
            truncation=True,
            padding='max_length'
        )
        
        # Tokenize targets
        tokenizer.src_lang = tgt_lang
        targets = tokenizer(
            examples['target'],
            max_length=MAX_LENGTH,
            truncation=True,
            padding='max_length'
        )
        
        inputs['labels'] = targets['input_ids']
        return inputs
    
    # Create dataset
    dataset_dict = {
        'source': pt_texts,
        'target': tupi_texts
    }
    dataset = Dataset.from_dict(dataset_dict)
    dataset = dataset.map(preprocess_function, batched=True)
    
    return dataset


class FewShotTranslator:    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = None
    
    def train(self, train_data, valid_data, src_lang, tgt_lang, output_dir):
        """Trains model with fine-tuning"""
        print(f"Starting fine-tuning: {src_lang} -> {tgt_lang}")
        
        # Load fresh model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        self.model.to(DEVICE)
        
        train_pt, train_tupi = train_data
        valid_pt, valid_tupi = valid_data
        
        # If direction is Tupi->PT, swap datasets
        if src_lang == LANG_TUPI:
            train_pt, train_tupi = train_tupi, train_pt
            valid_pt, valid_tupi = valid_tupi, valid_pt
        
        train_dataset = prepare_dataset_for_training(
            train_pt, train_tupi, self.tokenizer, src_lang, tgt_lang
        )
        valid_dataset = prepare_dataset_for_training(
            valid_pt, valid_tupi, self.tokenizer, src_lang, tgt_lang
        )
        
        # Configure training
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="steps",
            eval_steps=100,
            save_steps=200,
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=NUM_EPOCHS,
            weight_decay=WEIGHT_DECAY,
            warmup_steps=WARMUP_STEPS,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
            logging_steps=50,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            push_to_hub=False,
        )
        
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            padding=True
        )
        
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        print("Starting training")
        trainer.train()
        
        trainer.save_model(output_dir)
        print(f"Model saved in {output_dir}")
    
    def translate(self, texts, src_lang, tgt_lang, batch_size=8):
        """Translate using fine-tuned model"""
        
        self.model.eval()
        translations = []
        
        self.tokenizer.src_lang = src_lang
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
            batch = texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH
            ).to(DEVICE)
            
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_lang),
                    max_length=MAX_LENGTH,
                    num_beams=5,
                    early_stopping=True
                )
            
            batch_translations = self.tokenizer.batch_decode(
                generated,
                skip_special_tokens=True
            )
            
            translations.extend(batch_translations)
        
        return translations


def run_few_shot_experiments(data):
    """Runs few-shot experiments in both directions"""
    print("Few-shot translation (fine-tuning):")
    
    results = {}
    
    train_data = data['train']
    valid_data = data['valid']
    test_pt, test_tupi = data['test']
    
    print("Fine-tuning Portuguese -> Tupi")
    translator_pt_tupi = FewShotTranslator()
    translator_pt_tupi.train(
        train_data, valid_data,
        LANG_PT, LANG_TUPI,
        "models/nllb_pt_tupi"
    )
    
    print("Evaluating on test set")
    translations_pt_tupi = translator_pt_tupi.translate(test_pt, LANG_PT, LANG_TUPI)
    metrics_pt_tupi = evaluate_translations(test_tupi, translations_pt_tupi)
    results['PT->Tupi'] = {
        'translations': translations_pt_tupi,
        'metrics': metrics_pt_tupi
    }
    
    print(f"BLEU: {metrics_pt_tupi['BLEU']:.4f}")
    print(f"chrF1: {metrics_pt_tupi['chrF1']:.4f}")
    print(f"chrF3: {metrics_pt_tupi['chrF3']:.4f}")
    
    # Clear memory
    del translator_pt_tupi
    torch.cuda.empty_cache()
    
    print("Fine-tuning Tupi -> Portuguese")
    translator_tupi_pt = FewShotTranslator()
    translator_tupi_pt.train(
        train_data, valid_data,
        LANG_TUPI, LANG_PT,
        "models/nllb_tupi_pt"
    )
    
    print("Evaluating on test set")
    translations_tupi_pt = translator_tupi_pt.translate(test_tupi, LANG_TUPI, LANG_PT)
    metrics_tupi_pt = evaluate_translations(test_pt, translations_tupi_pt)
    results['Tupi->PT'] = {
        'translations': translations_tupi_pt,
        'metrics': metrics_tupi_pt
    }
    
    print(f"BLEU: {metrics_tupi_pt['BLEU']:.4f}")
    print(f"chrF1: {metrics_tupi_pt['chrF1']:.4f}")
    print(f"chrF3: {metrics_tupi_pt['chrF3']:.4f}")
    
    save_results(results, "results_few_shot.json")
    save_examples(test_pt, test_tupi, translations_pt_tupi, translations_tupi_pt,
                  "examples_few_shot.txt")
    
    return results


# === UTILITIES === #

def save_results(results, filename):
    """Saves metrics in JSON"""
    output = {}
    for direction, data in results.items():
        output[direction] = {
            'BLEU': float(data['metrics']['BLEU']),
            'chrF1': float(data['metrics']['chrF1']),
            'chrF3': float(data['metrics']['chrF3']),
        }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved in {filename}")


def save_examples(test_pt, test_tupi, trans_pt_tupi, trans_tupi_pt, filename):
    """Saves translation examples"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("Translation examples\n")
        
        # Select 10 random examples
        indices = np.random.choice(len(test_pt), min(10, len(test_pt)), replace=False)
        
        for idx in indices:
            f.write(f"Example {idx + 1}:\n")
            f.write(f"PT:    {test_pt[idx]}\n")
            f.write(f"Tupi:  {test_tupi[idx]}\n")
            f.write(f"PT->Tupi (pred): {trans_pt_tupi[idx]}\n")
            f.write(f"Tupi->PT (pred): {trans_tupi_pt[idx]}\n")
            f.write("\n")
    
    print(f"Examples saved in {filename}")


def create_comparison_table(zero_shot_results, few_shot_results):
    """Creates final comparison table"""
    print("Comparison table: zero-shot vs few-shot")
    
    table = []
    table.append(["Direction", "Method", "BLEU", "chrF1", "chrF3"])
    table.append(["-" * 15, "-" * 10, "-" * 8, "-" * 8, "-" * 8])
    
    for direction in ['PT->Tupi', 'Tupi->PT']:
        zs_metrics = zero_shot_results[direction]['metrics']
        fs_metrics = few_shot_results[direction]['metrics']
        
        table.append([
            direction,
            "Zero-shot",
            f"{zs_metrics['BLEU']:.4f}",
            f"{zs_metrics['chrF1']:.4f}",
            f"{zs_metrics['chrF3']:.4f}"
        ])
        
        table.append([
            "",
            "Few-shot",
            f"{fs_metrics['BLEU']:.4f}",
            f"{fs_metrics['chrF1']:.4f}",
            f"{fs_metrics['chrF3']:.4f}"
        ])
        
        # Calculate improvement
        bleu_imp = ((fs_metrics['BLEU'] - zs_metrics['BLEU']) / zs_metrics['BLEU'] * 100)
        chrf1_imp = ((fs_metrics['chrF1'] - zs_metrics['chrF1']) / zs_metrics['chrF1'] * 100)
        chrf3_imp = ((fs_metrics['chrF3'] - zs_metrics['chrF3']) / zs_metrics['chrF3'] * 100)
        
        table.append([
            "",
            "Melhoria %",
            f"{bleu_imp:+.1f}%",
            f"{chrf1_imp:+.1f}%",
            f"{chrf3_imp:+.1f}%"
        ])
        
        table.append(["", "", "", "", ""])
    
    for row in table:
        print(f"{row[0]:15} {row[1]:10} {row[2]:8} {row[3]:8} {row[4]:8}")


# === MAIN FUNCTION === #

def main():    
    data = load_all_data()
    
    # Create output directories
    os.makedirs("models", exist_ok=True)
    
    zero_shot_results = run_zero_shot_experiments(data)
    few_shot_results = run_few_shot_experiments(data)
    create_comparison_table(zero_shot_results, few_shot_results)
    
    print("Execution completed")
    print("Generated files:")
    print("1. results_zero_shot.json: Zero-shot metrics")
    print("2. results_few_shot.json: Few-shot metrics")
    print("3. examples_zero_shot.txt: Zero-shot examples")
    print("4. examples_few_shot.txt: Few-shot examples")
    print("5. models/nllb_pt_tupi/: Fine-tuned model PT->Tupi")
    print("6. models/nllb_tupi_pt/: Fine-tuned model Tupi->PT")


if __name__ == "__main__":
    main()