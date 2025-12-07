import pandas as pd
import re
from sklearn.model_selection import train_test_split

# Filter 0: manual adaptation in the spreadsheet before exporting
df = pd.read_excel("data/data_full.xlsx")

# Filter 1: remove invisible characters
def clear_invisible_chars(text):
    text = re.sub(r'[\x00-\x1F\x7F\u200B\uFEFF]', '', text)
    return text.strip()

# Filter 2: remove duplicate spaces
def normalize_spaces(texto):
    return re.sub(r'\s+', ' ', texto).strip()

# Filter 3: remove external quotes, may have been caused by export
def remove_external_quotes(text):
    text = text.strip()
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith('“') and text.endswith('”')) or \
       (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    return text

# Filter 4: remove spaces before punctuation
def fix_space_punct(text):
    text = re.sub(r'\s+([!?.,;:])', r'\1', text)
    return text

def clean_text(text):
    text = str(text)
    text = clear_invisible_chars(text)
    text = normalize_spaces(text)
    text = remove_external_quotes(text)
    text = fix_space_punct(text)
    return text

df["Portuguese"] = df["Portuguese"].apply(clean_text)
df["Old Tupi"] = df["Old Tupi"].apply(clean_text)

# Filter 5: remove empty lines
df = df[(df["Portuguese"] != "") & (df["Old Tupi"] != "")]
df = df.dropna()

# Filter 6: remove duplicates (identical in Portuguese and Tupi, DO NOT REMOVE two distinct translations for the same sentence)
df = df.drop_duplicates()

# Filter 7: heuristic for suspect cases, detected 6 cases, among these 5 were poorly formatted. They were removed and cataloged.
tupi_patt = r"[îûỹẽ]"
df["suspect_chars"] = df["Portuguese"].str.count(tupi_patt)
suspect = df[df["suspect_chars"] >= 2]

legitimate = suspect[
    suspect["Portuguese"].str.contains("Gûaîxará", na=False) |
    suspect["Old Tupi"].str.contains("Gûaîxará", na=False)
]

noise = suspect.drop(index=legitimate.index)
noise.to_csv(
    "data/noise_removed.tsv",
    sep="\t",
    index=False,
    encoding="utf-8"
)

df = df.drop(index=noise.index)
df = df.drop("suspect_chars", axis=1)

train, temp = train_test_split(df, test_size=0.30, random_state=42)
valid, test = train_test_split(temp, test_size=0.50, random_state=42)

train.to_csv("data/train.tsv", index=False, encoding="utf-8", sep="\t")
valid.to_csv("data/valid.tsv", index=False, encoding="utf-8", sep="\t")
test.to_csv("data/test.tsv", index=False, encoding="utf-8", sep="\t")