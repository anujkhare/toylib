import datasets as hf_datasets
import functools
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np
import re


def load_glove_embeddings(filepath):
    embeddings = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vector
    return embeddings


def preprocess_text(text: str, stopwords=None, cls_token=None):
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and replace with space
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords if provided
    if stopwords:
        tokens = [word for word in tokens if word not in stopwords]

    if cls_token:
        tokens = [cls_token] + tokens

    return tokens


def embed_tokens(tokens, glove, embedding_dim):
    vectors = []
    emb_missing = []
    for token in tokens:
        if token in glove:
            vectors.append(glove[token])
            emb_missing.append(False)
        else:
            vectors.append(np.zeros(embedding_dim))
            emb_missing.append(True)
            print(f"Token not found in glove: {token}")
    return np.stack(vectors), np.array(emb_missing)


class EmbeddedTextDataset(Dataset):
    def __init__(self, hf_dataset, glove, embedding_dim, max_tokens=50, cls_token=None):
        self.data = hf_dataset
        self.glove = glove
        self.embedding_dim = embedding_dim
        self.max_tokens = max_tokens
        self.cls_token = cls_token

    def __len__(self):
        return len(self.data)

    @property
    def stopwords(self) -> list[str]:
        stopwords = [
            "a",
            "about",
            "above",
            "after",
            "again",
            "against",
            "all",
            "am",
            "an",
            "and",
            "any",
            "are",
            "as",
            "at",
            "be",
            "because",
            "been",
            "before",
            "being",
            "below",
            "between",
            "both",
            "but",
            "by",
            "could",
            "did",
            "do",
            "does",
            "doing",
            "down",
            "during",
            "each",
            "few",
            "for",
            "from",
            "further",
            "had",
            "has",
            "have",
            "having",
            "he",
            "he'd",
            "he'll",
            "he's",
            "her",
            "here",
            "here's",
            "hers",
            "herself",
            "him",
            "himself",
            "his",
            "how",
            "how's",
            "i",
            "i'd",
            "i'll",
            "i'm",
            "i've",
            "if",
            "in",
            "into",
            "is",
            "it",
            "it's",
            "its",
            "itself",
            "let's",
            "me",
            "more",
            "most",
            "my",
            "myself",
            "nor",
            "of",
            "on",
            "once",
            "only",
            "or",
            "other",
            "ought",
            "our",
            "ours",
            "ourselves",
            "out",
            "over",
            "own",
            "same",
            "she",
            "she'd",
            "she'll",
            "she's",
            "should",
            "so",
            "some",
            "such",
            "than",
            "that",
            "that's",
            "the",
            "their",
            "theirs",
            "them",
            "themselves",
            "then",
            "there",
            "there's",
            "these",
            "they",
            "they'd",
            "they'll",
            "they're",
            "they've",
            "this",
            "those",
            "through",
            "to",
            "too",
            "under",
            "until",
            "up",
            "very",
            "was",
            "we",
            "we'd",
            "we'll",
            "we're",
            "we've",
            "were",
            "what",
            "what's",
            "when",
            "when's",
            "where",
            "where's",
            "which",
            "while",
            "who",
            "who's",
            "whom",
            "why",
            "why's",
            "with",
            "would",
            "you",
            "you'd",
            "you'll",
            "you're",
            "you've",
            "your",
            "yours",
            "yourself",
            "yourselves",
        ]

        # specific stopwords
        specific_sw = ["br", "movie", "film"]

        # all stopwords
        stopwords = stopwords + specific_sw
        return stopwords

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = preprocess_text(
            item["text"], stopwords=self.stopwords, cls_token=self.cls_token
        )[: self.max_tokens]
        emb, emb_missing = embed_tokens(tokens, self.glove, self.embedding_dim)

        # Pad or truncate
        num_pad = 0
        if emb.shape[0] < self.max_tokens:
            num_pad = self.max_tokens - emb.shape[0]

            # pad embeddings
            pad = np.zeros((num_pad, self.embedding_dim))
            emb = np.vstack([emb, pad])
            # pad mask
            emb_missing = np.concatenate(
                [emb_missing, np.ones(num_pad).astype(np.bool)]
            )
        else:
            emb = emb[: self.max_tokens]
            emb_missing = emb_missing[: self.max_tokens]

        return {
            "embedding": emb.astype(np.float32),
            "embedding_missing": emb_missing.astype(np.bool),
            "num_tokens": len(tokens),
            "raw_text": item["text"],
            "label": item["label"],
            "num_pad": num_pad,
        }


def numpy_collate(batch):
    return {
        "embedding": np.stack([item["embedding"] for item in batch]),  # (B, T, D),
        "embedding_missing": np.stack(
            [item["embedding_missing"] for item in batch]
        ),  # (B, T),
        "label": np.array([item["label"] for item in batch], dtype=np.float32),  # (B,)
        "num_pad": np.array([item["num_pad"] for item in batch]),  # (B,)
        "num_tokens": np.array([item["num_tokens"] for item in batch]),  # (B,)
        "raw_text": np.array([item["raw_text"] for item in batch]),  # (B,)
    }


def load_dataset(glove, batch_size, embedding_dim, max_tokens, cls_token=None):
    # Load the Huggingface dataset
    imdb_dataset = hf_datasets.load_dataset("imdb")

    # Split train into train/val
    train_val = imdb_dataset["train"].train_test_split(test_size=0.05, seed=42)

    # Create Dataset classes
    dataset_fn = functools.partial(
        EmbeddedTextDataset,
        glove=glove,
        embedding_dim=embedding_dim,
        max_tokens=max_tokens,
        cls_token=cls_token,
    )
    train_dataset = dataset_fn(train_val["train"])
    val_dataset = dataset_fn(train_val["test"])
    test_dataset = dataset_fn(imdb_dataset["test"])

    # Set up dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=numpy_collate,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=numpy_collate,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=numpy_collate,
    )
    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_dataloader,
        val_dataloader,
        test_dataloader,
    )
