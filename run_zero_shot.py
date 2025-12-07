"""
Simplified script to test zero-shot translation between Portuguese and Old Tupi
using the NLLB-200-distilled-600M model.
Runs quickly to validate the pipeline before full fine-tuning
"""

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import json

# === SETTINGS === #

MODEL_NAME = "facebook/nllb-200-distilled-600M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LANG_PT = "por_Latn"
LANG_TUPI = "por_Latn"

MAX_LENGTH = 128


# === AUXILIARY FUNCTIONS === #

def format_str(text):
    text = text.strip()
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    return text


def read_tsv(file_path):
    frases_pt = []
    frases_tupi = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        next(f)  # Skip header
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            
            partes = linha.split("\t")
            if len(partes) >= 2:
                pt, tupi = partes[0], partes[1]
                frases_pt.append(format_str(pt))
                frases_tupi.append(format_str(tupi))
    
    return frases_pt, frases_tupi


def count_ngrams(text, n):
    ngrams = {}
    for i in range(len(text) - n + 1):
        ng = text[i:i+n]
        ngrams[ng] = ngrams.get(ng, 0) + 1
    return ngrams


# === METRICS === #

class BleuScorer:    
    def __init__(self, weights=[0.25, 0.25, 0.25, 0.25], eps=1e-12):
        self.weights = weights
        self.eps = eps
    
    def get_prob_bleu(self, reference, hypothesis, n):
        if len(hypothesis) < n:
            return 0.0
        
        ngrams_hyp = count_ngrams(hypothesis, n)
        ngrams_ref = count_ngrams(reference, n)
        
        clipped = sum(min(count, ngrams_ref.get(ng, 0)) 
                     for ng, count in ngrams_hyp.items())
        total = sum(ngrams_hyp.values())
        
        return clipped / total if total > 0 else 0.0
    
    def brevity_penalty(self, reference, hypothesis):
        r, c = len(reference), len(hypothesis)
        return np.exp(-max(0.0, r/c - 1)) if c > 0 else 0.0
    
    def bleu_sentence(self, reference, hypothesis):
        log_probs = [np.log(max(self.get_prob_bleu(reference, hypothesis, i+1), self.eps))
                     for i in range(len(self.weights))]
        bp = self.brevity_penalty(reference, hypothesis)
        return bp * np.exp(np.dot(log_probs, self.weights))
    
    def bleu_corpus(self, references, hypotheses):
        return np.mean([self.bleu_sentence(ref, hyp) 
                       for ref, hyp in zip(references, hypotheses)])


class ChrFScorer:    
    def __init__(self, beta=1.0, n=6):
        self.beta = beta
        self.n = n
    
    def chrf_sentence(self, reference, hypothesis):
        total_match = total_hyp = total_ref = 0
        
        for k in range(1, self.n + 1):
            ngrams_ref = count_ngrams(reference, k)
            ngrams_hyp = count_ngrams(hypothesis, k)
            
            for ng, count_hyp in ngrams_hyp.items():
                total_match += min(count_hyp, ngrams_ref.get(ng, 0))
                total_hyp += count_hyp
            
            total_ref += sum(ngrams_ref.values())
        
        if total_hyp == 0 or total_ref == 0:
            return 0.0
        
        P = total_match / total_hyp
        R = total_match / total_ref
        beta_sq = self.beta ** 2
        
        return (1 + beta_sq) * P * R / (beta_sq * P + R) if (P + R) > 0 else 0.0
    
    def chrf_corpus(self, references, hypotheses):
        return np.mean([self.chrf_sentence(ref, hyp) 
                       for ref, hyp in zip(references, hypotheses)])


def evaluate_translations(references, hypotheses):
    return {
        'BLEU': BleuScorer().bleu_corpus(references, hypotheses),
        'chrF1': ChrFScorer(beta=1.0).chrf_corpus(references, hypotheses),
        'chrF3': ChrFScorer(beta=3.0).chrf_corpus(references, hypotheses),
    }


# === ZERO-SHOT TRANSLATOR === #

def translate_batch(model, tokenizer, texts, src_lang, tgt_lang, batch_size=8):
    translations = []
    tokenizer.src_lang = src_lang
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
        batch = texts[i:i+batch_size]
        
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        ).to(DEVICE)
        
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
                max_length=MAX_LENGTH,
                num_beams=5,
                early_stopping=True
            )
        
        batch_trans = tokenizer.batch_decode(generated, skip_special_tokens=True)
        translations.extend(batch_trans)
    
    return translations


def print_examples(pt_texts, tupi_texts, trans_pt_tupi, trans_tupi_pt, n=5):
    print("Translation examples")
    
    indices = np.random.choice(len(pt_texts), min(n, len(pt_texts)), replace=False)
    
    for i, idx in enumerate(indices, 1):
        print(f"Example {i}:")
        print(f"PT:    {pt_texts[idx]}")
        print(f"Tupi:  {tupi_texts[idx]}")
        print(f"PT->Tupi (pred): {trans_pt_tupi[idx]}")
        print(f"Tupi->PT (pred): {trans_tupi_pt[idx]}")
        print()


# === MAIN === #

def main():    
    test_pt, test_tupi = read_tsv("data/test.tsv")
    print(f"Total of {len(test_pt)} test pairs")
        
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    
    
    # PT -> Tupi
    trans_pt_tupi = translate_batch(model, tokenizer, test_pt, LANG_PT, LANG_TUPI)
    metrics_pt_tupi = evaluate_translations(test_tupi, trans_pt_tupi)
    
    print("Metrics for PT->Tupi:")
    print(f"  BLEU:  {metrics_pt_tupi['BLEU']:.4f}")
    print(f"  chrF1: {metrics_pt_tupi['chrF1']:.4f}")
    print(f"  chrF3: {metrics_pt_tupi['chrF3']:.4f}")
    
    # Tupi -> PT
    trans_tupi_pt = translate_batch(model, tokenizer, test_tupi, LANG_TUPI, LANG_PT)
    metrics_tupi_pt = evaluate_translations(test_pt, trans_tupi_pt)
    
    print("Metrics for Tupi->PT:")
    print(f"  BLEU:  {metrics_tupi_pt['BLEU']:.4f}")
    print(f"  chrF1: {metrics_tupi_pt['chrF1']:.4f}")
    print(f"  chrF3: {metrics_tupi_pt['chrF3']:.4f}")
    
    results = {
        'PT->Tupi': {
            'BLEU': float(metrics_pt_tupi['BLEU']),
            'chrF1': float(metrics_pt_tupi['chrF1']),
            'chrF3': float(metrics_pt_tupi['chrF3']),
        },
        'Tupi->PT': {
            'BLEU': float(metrics_tupi_pt['BLEU']),
            'chrF1': float(metrics_tupi_pt['chrF1']),
            'chrF3': float(metrics_tupi_pt['chrF3']),
        }
    }
    
    with open('results_zero_shot.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print_examples(test_pt, test_tupi, trans_pt_tupi, trans_tupi_pt, n=5)
    
    with open('examples_zero_shot.txt', 'w', encoding='utf-8') as f:
        f.write("Zero-shot translation examples:\n")
        
        indices = np.random.choice(len(test_pt), min(10, len(test_pt)), replace=False)
        for i, idx in enumerate(indices, 1):
            f.write(f"Example {i}:\n")
            f.write("-" * 80 + "\n")
            f.write(f"PT:    {test_pt[idx]}\n")
            f.write(f"Tupi:  {test_tupi[idx]}\n")
            f.write(f"PT->Tupi (pred): {trans_pt_tupi[idx]}\n")
            f.write(f"Tupi->PT (pred): {trans_tupi_pt[idx]}\n\n")
    
    
    print("FINAL SUMMARY")
    print(f"\n{'Direction':<15} {'BLEU':>8} {'chrF1':>8} {'chrF3':>8}")
    print("-" * 42)
    print(f"{'PT->Tupi':<15} {metrics_pt_tupi['BLEU']:>8.4f} "
          f"{metrics_pt_tupi['chrF1']:>8.4f} {metrics_pt_tupi['chrF3']:>8.4f}")
    print(f"{'Tupi->PT':<15} {metrics_tupi_pt['BLEU']:>8.4f} "
          f"{metrics_tupi_pt['chrF1']:>8.4f} {metrics_tupi_pt['chrF3']:>8.4f}")

if __name__ == "__main__":
    main()
