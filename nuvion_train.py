# nuvion_train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast, GradScaler
import math
import os
from datasets import load_dataset, concatenate_datasets, Dataset as HFDataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm

# ========================
# CONFIG
# ========================
DEV_MODE = False
DEV_SAMPLE_PER_DS = 500
MAX_SEQ_LEN = 384
BATCH_SIZE = 16
NUM_EPOCHS = 1 if DEV_MODE else 10
LEARNING_RATE = 1e-4
D_MODEL = 512
NHEAD = 8
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1
VOCAB_SIZE = 32000

UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEV_MODE={DEV_MODE} | Device={device}")

# ========================
# DATASETS
# ========================
def safe_load_dataset(name, split="train", dev_sample=None):
    try:
        ds = load_dataset(name, split=split)
        if dev_sample:
            ds = ds.select(range(min(len(ds), dev_sample)))
        return ds
    except Exception as e:
        print(f"Could not load {name}; using empty dataset fallback. Error: {e}")
        return HFDataset.from_dict({'instruction': [], 'output': [], 'input': [], 'conversations': []})

def unify_and_format(example):
    if "instruction" in example and "output" in example:
        return {"source_text": str(example["instruction"]), "target_text": str(example["output"])}
    if "conversations" in example and len(example["conversations"]) >= 2:
        conv = example["conversations"]
        return {"source_text": str(conv[0]["value"]), "target_text": str(conv[1]["value"])}
    if "input" in example and "output" in example:
        return {"source_text": str(example["input"]), "target_text": str(example["output"])}
    return {"source_text": "", "target_text": ""}

def load_all_datasets():
    print("Loading datasets (OASST1, Alpaca, ShareGPT)...")
    ds_list = [
        safe_load_dataset("OpenAssistant/oasst1", dev_sample=DEV_SAMPLE_PER_DS if DEV_MODE else None),
        safe_load_dataset("tatsu-lab/alpaca", dev_sample=DEV_SAMPLE_PER_DS if DEV_MODE else None),
        safe_load_dataset("openai/ShareGPT", dev_sample=DEV_SAMPLE_PER_DS if DEV_MODE else None)
    ]
    ds_list = [ds for ds in ds_list if len(ds) > 0]
    return concatenate_datasets(ds_list)

# ========================
# TOKENIZER
# ========================
def get_or_train_tokenizer(dataset, path="tokenizer_unified_nuvion.json"):
    if os.path.exists(path):
        print(f"Loading existing tokenizer from {path}")
        return Tokenizer.from_file(path)

    print("Training new tokenizer...")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)

    def iterator():
        for item in dataset:
            yield item["source_text"]
            yield item["target_text"]

    tokenizer.train_from_iterator(iterator(), trainer=trainer)
    tokenizer.save(path)
    print(f"Tokenizer saved to {path}")
    return tokenizer

# ========================
# DATASET CLASS
# ========================
class UnifiedDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_len = MAX_SEQ_LEN

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        src, tgt = item["source_text"], item["target_text"]
        if not src or not tgt:
            return None
        src_ids = [SOS_IDX] + self.tokenizer.encode(src).ids[:self.max_len - 2] + [EOS_IDX]
        tgt_ids = [SOS_IDX] + self.tokenizer.encode(tgt).ids[:self.max_len - 2] + [EOS_IDX]
        return torch.tensor(src_ids), torch.tensor(tgt_ids)

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None
    srcs, tgts = zip(*batch)
    src_padded = pad_sequence(srcs, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgts, padding_value=PAD_IDX)
    return src_padded, tgt_padded

# ========================
# MODEL
# ========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, 1, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0, 0::2] = torch.sin(pos * div_term)
        pe[:, 0, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:x.size(0)])

class NuvionTransformer(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.d_model = D_MODEL
        self.embedding = nn.Embedding(vocab_size, D_MODEL)
        self.pos_encoder = PositionalEncoding(D_MODEL, DROPOUT)
        self.transformer = nn.Transformer(
            d_model=D_MODEL,
            nhead=NHEAD,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_DECODER_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            batch_first=False
        )
        self.out_linear = nn.Linear(D_MODEL, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                src_padding_mask=None, tgt_padding_mask=None, memory_key_padding_mask=None):
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))
        if tgt_mask is None:
            L = tgt.size(0)
            tgt_mask = torch.triu(torch.ones(L, L)) == 1
            tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, 0.0).to(src.device)
        out = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask,
                               src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.out_linear(out)

# ========================
# TRAINING LOOP
# ========================
def train():
    dataset_raw = load_all_datasets()
    dataset_raw = dataset_raw.map(unify_and_format)
    dataset_raw = dataset_raw.filter(lambda x: x["source_text"] and x["target_text"])
    tokenizer = get_or_train_tokenizer(dataset_raw)

    train_dataset = UnifiedDataset(dataset_raw, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              collate_fn=collate_fn, shuffle=True,
                              num_workers=2, pin_memory=True)

    model = NuvionTransformer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    scaler = GradScaler()

    for epoch in range(NUM_EPOCHS):
        print(f"=== Epoch {epoch + 1}/{NUM_EPOCHS} ===")
        model.train()
        loop = tqdm(train_loader)
        for i, (src, tgt) in enumerate(loop):
            if src is None: continue
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            with (torch.cuda.amp.autocast() if device.type == "cuda" else torch.amp.autocast("cpu")):
                output = model(src, tgt[:-1])
                loss = criterion(output.view(-1, VOCAB_SIZE), tgt[1:].reshape(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loop.set_description(f"Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "Nuvion.pth")
    print("Training complete. Model saved to Nuvion.pth")

# ========================
# ENTRY POINT
# ========================
if __name__ == "__main__":
    train()
