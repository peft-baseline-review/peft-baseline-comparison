import pandas as pd
import numpy as np
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

# Dataset 1: Tigrinya Sentiment Dataset
DS1_TRAIN_CSV = "data/processed/sentiment/train.csv"
DS1_VAL_CSV   = "data/processed/sentiment/dev.csv"
DS1_TEST_CSV  = "data/processed/sentiment/test.csv"

# Dataset 2: TiDG
DS2_TRAIN_CSV = "data/processed/tidg/train.csv"
DS2_VAL_CSV   = "data/processed/tidg/val.csv"
DS2_TEST_CSV  = "data/processed/tidg/test.csv"

# ============================================================
# STATISTICS FUNCTIONS
# ============================================================

def count_words(text: str) -> int:
    """Return number of whitespace-separated tokens."""
    return len(str(text).split())

def compute_word_stats(csv_path, dstype="sentiment"):
    """Load CSV and return list of word lengths for all texts."""
    if dstype == "tidg":
        df = pd.read_csv(csv_path)
        texts = df['cleaned_text'].astype(str)
    else:
        df = pd.read_csv(csv_path, sep=";")
        texts = df['text'].astype(str)
    word_lengths = texts.apply(count_words).tolist()
    return word_lengths

def compute_vocab_size(csv_path, col_name="text"):
    """Compute vocabulary size (unique word types) from training CSV only."""
    if col_name == "cleaned_text":
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(csv_path, sep=";")
    vocab = set()
    for text in df[col_name].astype(str):
        vocab.update(text.split())
    return len(vocab)

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Processing Tigrinya Sentiment Dataset...")
    train_len1 = compute_word_stats(DS1_TRAIN_CSV)
    val_len1   = compute_word_stats(DS1_VAL_CSV)
    test_len1  = compute_word_stats(DS1_TEST_CSV)
    all_lengths1 = np.array(train_len1 + val_len1 + test_len1)
    avg_len1 = np.mean(all_lengths1)
    vocab_size1 = compute_vocab_size(DS1_TRAIN_CSV)

    print("Processing TiDG Dataset...")
    train_len2 = compute_word_stats(DS2_TRAIN_CSV, "tidg")
    val_len2   = compute_word_stats(DS2_VAL_CSV, "tidg")
    test_len2  = compute_word_stats(DS2_TEST_CSV, "tidg")
    all_lengths2 = np.array(train_len2 + val_len2 + test_len2)
    avg_len2 = np.mean(all_lengths2)
    vocab_size2 = compute_vocab_size(DS2_TRAIN_CSV, "cleaned_text")

    print("\n% ----- Copy these numbers into your table -----")
    print(f"Tigrinya Sentiment Dataset & 2 & "
          f"{len(train_len1):,} & {len(val_len1):,} & {len(test_len1):,} & "
          f"{avg_len1:.1f} & {vocab_size1:,} \\\\")
    print(f"TiDG (Tigrinya Diverse Genre Corpus) & 7 & "
          f"{len(train_len2):,} & {len(val_len2):,} & {len(test_len2):,} & "
          f"{avg_len2:.1f} & {vocab_size2:,} \\\\")