import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
import re

# ============================================================
# CONFIGURATION
# ============================================================

DS1_TRAIN_CSV = "data/processed/sentiment/train.csv"
DS1_VAL_CSV   = "data/processed/sentiment/dev.csv"
DS1_TEST_CSV  = "data/processed/sentiment/test.csv"

DS2_TRAIN_CSV = "data/processed/tidg/train.csv"
DS2_VAL_CSV   = "data/processed/tidg/val.csv"
DS2_TEST_CSV  = "data/processed/tidg/test.csv"

# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def identity(x):
    return x

# ============================================================
# STATISTICS
# ============================================================

def count_words(text: str) -> int:
    return len(text.split())

def count_chars(text: str) -> int:
    return len(text.replace(' ', ''))

def compute_stats_for_split(csv_path, preprocess_fn,
                            text_column="text",
                            delimiter=",",
                            desc="Processing"):
    """Load CSV, preprocess, compute word and character lengths."""
    df = pd.read_csv(csv_path, sep=delimiter)
    texts = df[text_column].astype(str).tolist()

    word_lengths = []
    char_lengths = []

    for text in tqdm(texts, desc=desc):
        proc_text = preprocess_fn(text)
        word_lengths.append(count_words(proc_text))
        char_lengths.append(count_chars(proc_text))

    df['word_len'] = word_lengths
    df['char_len'] = char_lengths
    return df

def overall_stats(df_train, df_val, df_test, preprocess_fn, text_column):
    n_classes = df_train['label'].nunique()
    train_n = len(df_train)
    val_n = len(df_val)
    test_n = len(df_test)
    total = train_n + val_n + test_n

    all_word_lens = np.concatenate(
        [df_train['word_len'], df_val['word_len'], df_test['word_len']])
    avg_words = float(np.mean(all_word_lens))
    max_words = float(np.max(all_word_lens))

    all_char_lens = np.concatenate(
        [df_train['char_len'], df_val['char_len'], df_test['char_len']])
    avg_chars = float(np.mean(all_char_lens))
    max_chars = float(np.max(all_char_lens))

    vocab_counter = Counter()
    for text in df_train[text_column].astype(str):
        proc_text = preprocess_fn(text)
        vocab_counter.update(proc_text.split())

    return {
        'n_classes': n_classes,
        'train': train_n,
        'val': val_n,
        'test': test_n,
        'total': total,
        'max_words': max_words,
        'avg_words': avg_words,
        'max_chars': max_chars,
        'avg_chars': avg_chars,
        'vocab_size_words': len(vocab_counter),
    }

def print_latex_row(ds_name, stats, use_chars=False):
    if use_chars:
        print(f"{ds_name} & {stats['n_classes']} & {stats['train']} & "
              f"{stats['val']} & {stats['test']} & {stats['total']} & "
              f"{stats['max_chars']:.0f} & {stats['avg_chars']:.1f} & "
              f"{stats['vocab_size_words']} \\\\")
    else:
        print(f"{ds_name} & {stats['n_classes']} & {stats['train']} & "
              f"{stats['val']} & {stats['test']} & {stats['total']} & "
              f"{stats['max_words']:.0f} & {stats['avg_words']:.1f} & "
              f"{stats['vocab_size_words']} \\\\")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=== Sentiment (with preprocessing) ===")
    df1_train = compute_stats_for_split(
        DS1_TRAIN_CSV, preprocess_text, text_column="text",
        delimiter=";", desc="DS1 Train")
    df1_val = compute_stats_for_split(
        DS1_VAL_CSV, preprocess_text, text_column="text",
        delimiter=";", desc="DS1 Val")
    df1_test = compute_stats_for_split(
        DS1_TEST_CSV, preprocess_text, text_column="text",
        delimiter=";", desc="DS1 Test")
    stats1 = overall_stats(df1_train, df1_val, df1_test,
                           preprocess_text, "text")

    print("\n=== TiDG (already preprocessed) ===")
    df2_train = compute_stats_for_split(
        DS2_TRAIN_CSV, identity, text_column="cleaned_text",
        delimiter=",", desc="DS2 Train")
    df2_val = compute_stats_for_split(
        DS2_VAL_CSV, identity, text_column="cleaned_text",
        delimiter=",", desc="DS2 Val")
    df2_test = compute_stats_for_split(
        DS2_TEST_CSV, identity, text_column="cleaned_text",
        delimiter=",", desc="DS2 Test")
    stats2 = overall_stats(df2_train, df2_val, df2_test,
                           identity, "cleaned_text")

    print("\n% ----- LaTeX rows for table (average words) -----")
    print_latex_row("Tigrinya Sentiment Dataset", stats1, use_chars=False)
    print_latex_row("TiDG (Tigrinya Diverse Genre Corpus)", stats2, use_chars=False)