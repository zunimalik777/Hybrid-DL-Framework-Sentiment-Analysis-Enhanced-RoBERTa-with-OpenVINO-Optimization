"""
Objective 2: Two-Layer ("Two-State") GRU

A GRU classifier stacked with two recurrent layers (two hidden "states"
propagated through the sequence), initialized with the Word2Vec
embedding matrix from Objective 4.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from . import config


class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class TwoLayerGRUClassifier(nn.Module):
    """Embedding -> 2-layer (bi)GRU -> dense classification head."""

    def __init__(self, embedding_matrix: np.ndarray, hidden_size=config.GRU_HIDDEN_SIZE,
                 num_layers=config.GRU_NUM_LAYERS, bidirectional=config.GRU_BIDIRECTIONAL,
                 dropout=config.GRU_DROPOUT, num_classes=config.GRU_NUM_CLASSES,
                 freeze_embeddings=False):
        super().__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            freeze=freeze_embeddings,
            padding_idx=0,
        )
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,      # "two-state" = 2 stacked GRU layers
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        directions = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * directions, num_classes)

    def forward(self, input_ids):
        # input_ids: (batch, seq_len)
        embedded = self.embedding(input_ids)                    # (batch, seq_len, embed_dim)
        outputs, hidden = self.gru(embedded)                     # hidden: (num_layers*dir, batch, hidden)
        # Concatenate the final layer's forward/backward hidden states (the two GRU "states")
        if self.gru.bidirectional:
            last_forward = hidden[-2]
            last_backward = hidden[-1]
            final_state = torch.cat([last_forward, last_backward], dim=1)
        else:
            final_state = hidden[-1]
        final_state = self.dropout(final_state)
        logits = self.classifier(final_state)
        return logits


def train_gru(model, train_loader, val_loader, epochs=config.GRU_EPOCHS, lr=config.GRU_LR,
              device=config.DEVICE):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for sequences, labels in tqdm(train_loader, desc=f"GRU epoch {epoch + 1}/{epochs}"):
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(sequences)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * sequences.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        val_loss, val_acc = _evaluate_loss(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    return model, history


def _evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for sequences, labels in loader:
            sequences, labels = sequences.to(device), labels.to(device)
            logits = model(sequences)
            loss = criterion(logits, labels)
            total_loss += loss.item() * sequences.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def predict_gru(model, loader, device=config.DEVICE):
    model.eval()
    model.to(device)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device)
            logits = model(sequences)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
    return np.array(all_preds), np.array(all_labels)


def make_dataloaders(train_seq, train_labels, val_seq, val_labels, test_seq, test_labels,
                      batch_size=config.GRU_BATCH_SIZE):
    train_loader = DataLoader(SequenceDataset(train_seq, train_labels), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SequenceDataset(val_seq, val_labels), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(SequenceDataset(test_seq, test_labels), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
