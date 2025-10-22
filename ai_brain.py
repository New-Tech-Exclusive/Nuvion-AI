import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.amp import GradScaler, autocast

import math
import os
import gc
from datasets import load_dataset, concatenate_datasets
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


# ------------------- 1. MODEL ARCHITECTURE -------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MyTransformerModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_encoder_layers: int,
                 num_decoder_layers: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.out_linear = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.out_linear.bias.data.zero_()
        self.out_linear.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))
        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(0)).to(src.device)
        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return self.out_linear(output)


# ------------------- 2. DATA AND TOKENIZER -------------------

UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
special_tokens = ['[UNK]', '[PAD]', '[SOS]', '[EOS]']
MAX_SEQ_LEN = 384


def get_or_train_unified_tokenizer(dataset, name, vocab_size=32000):
    tokenizer_path = f'tokenizer_unified_{name}.json'
    if os.path.exists(tokenizer_path):
        print(f"Loading existing tokenizer from {tokenizer_path}...")
        return Tokenizer.from_file(tokenizer_path)
    print(f"Training new tokenizer (vocab_size={vocab_size})...")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)

    def text_iterator(batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            sources = [x for x in batch['source_text'] if isinstance(x, str) and x]
            targets = [x for x in batch['target_text'] if isinstance(x, str) and x]
            if sources: yield sources
            if targets: yield targets

    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer trained and saved to {tokenizer_path}")
    return tokenizer


def normalize_translation_batch(batch):
    return {
        'source_text': [pair.get('en', '') for pair in batch['translation']],
        'target_text': [pair.get('fr', '') for pair in batch['translation']]
    }


class UnifiedDataset(Dataset):
    def __init__(self, raw_dataset, tokenizer, max_len=MAX_SEQ_LEN):
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        item = self.raw_dataset[idx]
        src_text = item.get('source_text', '')
        tgt_text = item.get('target_text', '')
        if not src_text or not tgt_text:
            return None
        src_ids = self.tokenizer.encode(src_text).ids
        tgt_ids = self.tokenizer.encode(tgt_text).ids
        if len(src_ids) > self.max_len - 2 or len(tgt_ids) > self.max_len - 2:
            return None
        src_tensor = torch.tensor([SOS_IDX] + src_ids + [EOS_IDX])
        tgt_tensor = torch.tensor([SOS_IDX] + tgt_ids + [EOS_IDX])
        return src_tensor, tgt_tensor


def collate_batch(batch):
    batch = [x for x in batch if x is not None]
    if not batch: return None, None
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_padded, tgt_padded


# ------------------- 3. TRAINING -------------------

if __name__ == '__main__':

    D_MODEL = 512
    NHEAD = 8
    NUM_ENCODER_LAYERS = 4
    NUM_DECODER_LAYERS = 4
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.1
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 16

    # Load datasets
    print("Loading datasets...")
    try:
        ds1 = load_dataset('opus_books', 'en-fr', split='train', trust_remote_code=True)
        ds2 = load_dataset('un_pc', 'en-fr', split='train', trust_remote_code=True)
    except Exception:
        ds1 = load_dataset('opus_books', 'en-fr', split='train', ignore_verifications=True, trust_remote_code=True)
        ds2 = load_dataset('un_pc', 'en-fr', split='train', ignore_verifications=True, trust_remote_code=True)

    combined_ds = concatenate_datasets([ds1, ds2])
    raw_dataset = combined_ds.map(normalize_translation_batch, batched=True,
                                  num_proc=4, remove_columns=combined_ds.column_names)
    del ds1, ds2, combined_ds
    gc.collect()

    raw_dataset = raw_dataset.shuffle(seed=42)
    print(f"Total examples: {len(raw_dataset)}")

    tokenizer_name = f"trans_only_{len(raw_dataset)}"
    tokenizer = get_or_train_unified_tokenizer(raw_dataset, tokenizer_name)
    VOCAB_SIZE = tokenizer.get_vocab_size()
    print(f"Vocabulary size: {VOCAB_SIZE}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = MyTransformerModel(
        VOCAB_SIZE, D_MODEL, NHEAD,
        NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, DROPOUT
    ).to(device)

    # ---- SAVE MODEL AS NUVION ----
    model_path = 'Nuvion.pth'
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            print("Model loaded successfully from Nuvion.pth")
        except Exception as e:
            print(f"Could not load model weights from Nuvion.pth: {e}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    train_dataset = UnifiedDataset(raw_dataset, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_batch,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )


    # ------------------- HELPER FUNCTIONS -------------------

    def make_padding_mask(seq):
        return (seq.transpose(0, 1) == PAD_IDX)


    def train_step(src, tgt):
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:-1, :]
        tgt_expected = tgt[1:, :]

        src_mask = None
        tgt_mask = model._generate_square_subsequent_mask(tgt_input.size(0)).to(device)
        src_pad_mask = make_padding_mask(src)
        tgt_pad_mask = make_padding_mask(tgt_input)
        memory_pad_mask = src_pad_mask

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            out = model(
                src, tgt_input, src_mask, tgt_mask,
                src_pad_mask, tgt_pad_mask, memory_pad_mask
            )
            loss = criterion(out.view(-1, VOCAB_SIZE), tgt_expected.reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        return loss.item()


    # ------------------- TRAINING LOOP -------------------

    global_step = 0
    save_every = 1000
    print_every = 50

    try:
        for epoch in range(1, NUM_EPOCHS + 1):
            print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} ===")
            try:
                from tqdm import tqdm

                loader = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
            except ImportError:
                loader = train_loader

            running_loss = 0.0
            count = 0

            for batch in loader:
                if batch is None:
                    continue
                src, tgt = batch
                if src is None or tgt is None:
                    continue
                loss_val = train_step(src, tgt)
                running_loss += loss_val
                count += 1
                global_step += 1

                if global_step % print_every == 0:
                    avg = running_loss / max(1, count)
                    print(f"[Step {global_step}] avg loss: {avg:.4f}")
                    running_loss = 0.0
                    count = 0

                if global_step % save_every == 0:
                    torch.save(model.state_dict(), model_path)
                    print(f"Saved checkpoint at step {global_step} to Nuvion.pth")

            torch.save(model.state_dict(), model_path)
            print(f"End of epoch {epoch} checkpoint saved to Nuvion.pth")

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint to Nuvion.pth...")
        torch.save(model.state_dict(), model_path)
    except Exception as e:
        print(f"Error during training: {e}")
        torch.save(model.state_dict(), model_path)
