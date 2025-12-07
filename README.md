# Tupi-Portuguese Neural Machine Translation

Neural machine translation system for Portuguese <-> Old Tupi (Tupinambá) using NLLB-200, exploring transfer learning for extremely low-resource historical languages.

## Overview

This project implements bidirectional translation between Portuguese and Old Tupi, a historical language from the Tupi-Guarani family spoken by indigenous populations along the Brazilian coast during the colonial period (16th-18th centuries). The language presents an extreme low-resource scenario with only approximately 1,500 parallel sentence pairs available, restricted primarily to religious texts, and crucially, Old Tupi is completely absent from the NLLB-200 model's pretraining data.

Despite these constraints, we demonstrate that transfer learning from massively multilingual models enables usable translation quality. Fine-tuning the NLLB-200-distilled-600M model on our limited corpus produces improvements of 100-250% over zero-shot baselines, achieving BLEU scores around 0.30 and chrF scores exceeding 0.50 in both translation directions. These results align with performance reported for other indigenous languages in similar low-resource conditions.

The work serves multiple purposes beyond technical demonstration. It provides a computational tool for historians and linguists working with colonial-era texts, contributes to preservation of indigenous linguistic heritage, and offers a reproducible case study in applying modern NLP techniques to historically significant languages with minimal available data.

This work was conducted as part of a Natural Language Processing course at the University of São Paulo (USP) in 2025, conducted by professor Dr. Marcelo Finger. The full technical report detailing methodology, experiments, and analysis is available in the `docs/` directory.

## Installation

The system requires Python 3.10 or higher and preferably a NVIDIA GPU with at least 8GB VRAM, though CPU execution is supported with longer training times. Approximately 10GB of disk space is needed for models and data.

```bash
git clone https://github.com/aka-caim/tupi-portuguese-llm-translation.git
cd tupi-portuguese-llm-translation

python -m venv venv
source venv/bin/activate  # on Linux/Mac
# venv\Scripts\activate   # on Windows

pip install -r requirements.txt
```

For GPU support with CUDA 11.8, install PyTorch separately:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Quick start

The workflow consists of three main steps: data preparation, zero-shot baseline evaluation, and full training with fine-tuning.

First, download and preprocess the parallel corpus:

```bash
cd data
wget "https://github.com/CalebeRezende/oldtupi_dataset/raw/main/C%C3%B3pia%20de%20portugues-guarani-tupi%20antigo.xlsx" -O data_full.xlsx
cd ..
python preprocess.py
```

The preprocessing script performs systematic cleaning including removal of invisible characters, normalization of spacing and punctuation, detection of malformed entries, and stratified splitting into 70% training, 15% validation, and 15% test sets.

For rapid testing, run zero-shot evaluation which applies the pretrained NLLB-200 model directly without adaptation:

```bash
python run_zero_shot.py
```

This completes in approximately 15 minutes and produces baseline metrics in `results_zero_shot.json` along with example translations in `examples_zero_shot.txt`.

The complete training pipeline with fine-tuning is executed via:

```bash
python implementation.py
```

This performs zero-shot evaluation followed by separate fine-tuning for each translation direction, saving trained models in `models/nllb_pt_tupi/` and `models/nllb_tupi_pt/`. Execution time ranges from 2-4 hours with GPU to 8-24 hours on CPU.

## Using trained models

Once training completes, the fine-tuned models can be loaded and used for translation:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("models/nllb_pt_tupi")
tokenizer = AutoTokenizer.from_pretrained("models/nllb_pt_tupi")

text = "Deus te ama"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=128, num_beams=5)
translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(translation)  # Expected: "Tupã nde porasúba"
```

## Results

The quantitative results demonstrate substantial improvement from fine-tuning. For Portuguese to Old Tupi translation, zero-shot achieves BLEU 0.082, chrF1 0.247, and chrF3 0.213, while few-shot fine-tuning reaches BLEU 0.287, chrF1 0.542, and chrF3 0.489, representing a 250% improvement in BLEU. The reverse direction (Old Tupi to Portuguese) shows similar patterns with zero-shot BLEU 0.095 improving to 0.319 after fine-tuning, a 236% gain.

These scores situate the work within expected ranges for indigenous languages in extreme low-resource scenarios. Recent shared tasks on indigenous American languages report BLEU scores typically between 0.15 and 0.35 for languages with similar data constraints. The consistently higher chrF scores compared to BLEU reflect the character-level metric's suitability for morphologically rich agglutinative languages like Old Tupi, where partial morpheme matches receive appropriate credit.

The asymmetry between translation directions—Old Tupi to Portuguese consistently outperforming Portuguese to Old Tupi—aligns with expectations given Portuguese is extensively represented in the model's pretraining while Old Tupi is completely novel. Generating text in a high-resource language from low-resource input benefits from robust prior knowledge, whereas the reverse requires learning entirely new lexical and morphological patterns.

## Methodology

We selected NLLB-200-distilled-600M over alternatives like mBART-large-50 and mT5-small based on several factors. NLLB-200 was explicitly designed for low-resource languages, employs a Mixture of Experts architecture enabling efficient capacity sharing between related languages, and critically includes Guarani—a language from the same Tupi-Guarani family—in its training set. This provides a foundation for cross-lingual transfer despite Old Tupi's absence from the original 200 languages.

The fine-tuning configuration balances learning capability against overfitting risk. We use a conservative learning rate of 5×10⁻⁵, batch size of 4 (optimized for 8GB GPU memory), and train for up to 10 epochs with early stopping based on validation loss. Warmup of 500 steps stabilizes initial training, while weight decay of 0.01 provides L2 regularization. Training is conducted separately for each translation direction, producing specialized models rather than a single bidirectional system.

Evaluation employs BLEU with uniform weights for n-grams of order 1-4, and chrF with both β=1 and β=3, considering character n-grams up to order 6. Both metrics are implemented directly in Python for transparency and reproducibility. We complement automated metrics with qualitative linguistic analysis examining morphological processing, syntactic structure, lexical selection, and error patterns.

## Examples and Analysis

Successful translations demonstrate the model learned genuine patterns. The sentence "Tu és meu filho" translates perfectly to "Xe r-aîra nde", correctly using the first-person possessive prefix *xe-*, obligatory relational *r-* for kinship nouns, and second-person pronoun *nde* in predicative position. This reflects the high frequency of such structures in the religious training corpus.

Partial successes reveal morphological understanding despite errors. "Deus te ama" generates "Tupã nde porasúba" versus reference "Tupã nde porasuíwa". The model correctly translates "Deus" to *Tupã*, second-person pronoun to *nde*, and uses the root *porasu-* (goodness/happiness), but selects a different nominalizing suffix. This indicates understanding of derivational morphology even when specific suffix choice diverges.

Failures typically involve incomplete vocabulary or complex syntax. "Não matarás" produces "Nde mondó eîmé" instead of "Nde r-e-îuká eîmé", correctly forming negative imperative structure with *eîmé* but substituting *mondó* (send) for *îuká* (kill). This suggests the model resorts to high-frequency verbs when encountering less common vocabulary. Subordinate clauses and sentences exceeding 15 words frequently show degraded quality with omitted grammatical markers and simplified structure.

## Limitations and Appropriate Use

The system has fundamental limitations that must be understood for appropriate application. The corpus is small and domain-restricted, training primarily on religious texts from colonial catechisms. This creates strong bias toward religious vocabulary while leaving secular domains underrepresented. The reference Portuguese is archaic 17th-century language, differing substantially from modern Brazilian Portuguese. Old Tupi being a historical language without native speakers means validation is impossible beyond comparison to historical documentation.

Automated metrics capture only surface correspondence, not semantic adequacy or grammatical correctness. Valid paraphrases that differ lexically from the single reference are penalized. The model lacks explicit grammatical knowledge and operates purely through statistical patterns learned from limited data, resulting in occasional grammatical errors and inability to generalize to complex constructions absent from training data.

This system should be used as an auxiliary tool for researchers and linguists, not as a definitive translation source. It can accelerate initial analysis of historical texts and provide rough drafts for human revision, but critical translations require expert linguistic validation. The work demonstrates technical feasibility of applying modern NLP to historical indigenous languages and provides methodology transferable to other low-resource scenarios, but does not replace careful philological and linguistic scholarship.

## Project Structure

```
tupi-portuguese-llm-translation/
├── data/                    # Corpus files and preprocessing scripts
├── docs/                    # Technical report and documentation
├── preprocess.py            # Data cleaning and curation
├── implementation.py        # Complete pipeline (zero+few shot)
├── run_zero_shot.py         # Quick baseline evaluation
└── requirements.txt         # Python dependencies
```

## References

The NLLB-200 model and methodology are described in Costa-jussà et al. (2022), "No Language Left Behind: Scaling Human-Centered Machine Translation".

## Contact

For questions, issues, or collaboration inquiries, open an issue on GitHub or contact the authors:

Caio Morais Sales [caiomorais@usp.br]
Cauê Fornielles da Costa [caueosta@usp.br]