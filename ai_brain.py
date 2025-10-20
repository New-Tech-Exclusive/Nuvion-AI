import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.amp import GradScaler, autocast

import math
import os
from datasets import load_dataset, concatenate_datasets
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import gc

# --- 1. The Model Architecture ---

class PositionalEncoding(nn.Module):
    """
    Injects positional information into the input embeddings.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1) # Shape: [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) # Shape: [d_model / 2]
        
        pe = torch.zeros(max_len, 1, d_model) # Shape: [max_len, 1, d_model]
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Adds positional encoding to the input embeddings. """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class MyTransformerModel(nn.Module):
    """
    The complete Transformer model for sequence-to-sequence tasks (e.g., translation).
    """
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int, 
                 nhead: int, 
                 num_encoder_layers: int, 
                 num_decoder_layers: int, 
                 dim_feedforward: int, 
                 dropout: float = 0.1):
        super().__init__()
        self.model_type = 'Transformer'
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

    def _init_weights(self) -> None:
        """ Initializes the weights of the embedding and output linear layers. """
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.out_linear.bias.data.zero_()
        self.out_linear.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """ Generates a square causal mask for the decoder's self-attention. """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, 
                src: torch.Tensor, 
                tgt: torch.Tensor, 
                src_mask: torch.Tensor = None,
                tgt_mask: torch.Tensor = None,
                src_padding_mask: torch.Tensor = None,
                tgt_padding_mask: torch.Tensor = None,
                memory_key_padding_mask: torch.Tensor = None
                ) -> torch.Tensor:
        """ Performs the forward pass of the Transformer model. """
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))

        if tgt_mask is None:
            tgt_seq_len = tgt.size(0)
            tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len).to(src.device)

        output = self.transformer(
            src=src_emb, 
            tgt=tgt_emb, 
            src_mask=src_mask, 
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        return self.out_linear(output)

# --- 2. Data Loading and Tokenization (Definitions) ---

# Define special token indices used by the tokenizer and model
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
special_tokens = ['[UNK]', '[PAD]', '[SOS]', '[EOS]']

# Maximum sequence length (tokens) allowed.
MAX_SEQ_LEN = 384 

def get_or_train_unified_tokenizer(dataset, name, vocab_size=32000) -> Tokenizer:
    """ 
    Loads a pre-trained BPE tokenizer if it exists, otherwise trains a new one
    on the provided dataset (expects 'source_text' and 'target_text' columns).
    """
    tokenizer_path = f'tokenizer_unified_{name}.json'
    if os.path.exists(tokenizer_path):
        print(f"Loading existing unified tokenizer from {tokenizer_path}...")
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        print(f"Training new unified tokenizer (Vocab size: {vocab_size})...")
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
        
        def text_iterator(batch_size=1000):
            if 'source_text' not in dataset.column_names or 'target_text' not in dataset.column_names:
                print("Warning: 'source_text' or 'target_text' columns missing for tokenizer training.")
                return 
            
            for i in range(0, len(dataset), batch_size):
                try:
                    batch = dataset[i : i + batch_size]
                    valid_source = [text for text in batch['source_text'] if isinstance(text, str) and text]
                    valid_target = [text for text in batch['target_text'] if isinstance(text, str) and text]
                    if valid_source: yield valid_source
                    if valid_target: yield valid_target
                except Exception as e:
                    print(f"Warning: Error in tokenizer text_iterator batch {i // batch_size}: {e}")
                    continue 

        tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
        tokenizer.save(tokenizer_path)
        print(f"New unified tokenizer trained and saved to {tokenizer_path}")
    return tokenizer

def normalize_translation_batch(batch) -> dict:
    """
    Takes a batch from a translation dataset and reformats it into the 
    unified {'source_text': ..., 'target_text': ...} structure (e.g., 'en' to 'fr').
    """
    return {
        'source_text': [pair.get('en', '') for pair in batch['translation']],
        'target_text': [pair.get('fr', '') for pair in batch['translation']]
    }

class UnifiedDataset(Dataset):
    """
    PyTorch Dataset class that handles tokenization, adding special tokens (SOS, EOS), 
    and filtering sequences that are longer than MAX_SEQ_LEN.
    """
    def __init__(self, raw_dataset, tokenizer, max_len=MAX_SEQ_LEN):
        super().__init__()
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        """ Returns the total number of examples in the dataset. """
        return len(self.raw_dataset)

    def __getitem__(self, idx: int) -> tuple | None:
        """ Retrieves, tokenizes, and formats a single example. """
        try:
            item = self.raw_dataset[idx]
            src_text = item.get('source_text', '') 
            tgt_text = item.get('target_text', '')
            
            if not src_text or not tgt_text: 
                return None

            src_tokens_raw = self.tokenizer.encode(src_text).ids
            tgt_tokens_raw = self.tokenizer.encode(tgt_text).ids
            
            # Filter sequences that are too long
            if len(src_tokens_raw) > self.max_len - 2 or len(tgt_tokens_raw) > self.max_len - 2:
                return None
            
            # Add Start-of-Sentence and End-of-Sentence tokens
            src_tokens = [SOS_IDX] + src_tokens_raw + [EOS_IDX]
            tgt_tokens = [SOS_IDX] + tgt_tokens_raw + [EOS_IDX]
            
            return torch.tensor(src_tokens), torch.tensor(tgt_tokens)
        except Exception as e:
            print(f"Warning: Error processing item {idx}: {e}") 
            return None

def collate_batch(batch) -> tuple | tuple[None, None]:
    """
    Collates a list of examples into a padded batch.
    """
    # Filter out None items (skipped examples from __getitem__)
    batch = [item for item in batch if item is not None]
    
    # If the entire batch was filtered out, return None
    if not batch: 
        return None, None
        
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)
        
    # Pad sequences to the length of the longest sequence in that batch
    src_padded = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=False)
    tgt_padded = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=False)
    
    return src_padded, tgt_padded


# --- SCRIPT EXECUTION GUARD ---
if __name__ == '__main__':

    # --- 3. Hyperparameters & Initialization ---
    D_MODEL = 512
    NHEAD = 8
    NUM_ENCODER_LAYERS = 4
    NUM_DECODER_LAYERS = 4
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.1

    NUM_EPOCHS = 10
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 16

    # --- Data Loading and Preprocessing ---
    
    print("Loading translation datasets...")
    # Load translation datasets (English-French books and UN documents)
    try:
        # Load datasets
        dataset1 = load_dataset('opus_books', 'en-fr', split='train', trust_remote_code=True) 
        dataset2 = load_dataset('un_pc', 'en-fr', split='train', trust_remote_code=True) 
    except Exception as e:
        print(f"Error loading translation datasets: {e}")
        print("Attempting to load with ignore_verifications=True. Use this cautiously.")
        dataset1 = load_dataset('opus_books', 'en-fr', split='train', ignore_verifications=True, trust_remote_code=True)
        dataset2 = load_dataset('un_pc', 'en-fr', split='train', ignore_verifications=True, trust_remote_code=True)
        
    translation_ds = concatenate_datasets([dataset1, dataset2]) # Combine the two translation datasets
    
    print("Normalizing translation data...")
    # Apply the normalization function to standardize the format
    raw_dataset = translation_ds.map( # Use raw_dataset directly for the combined translation data
        normalize_translation_batch, 
        batched=True, 
        num_proc=4, 
        remove_columns=translation_ds.column_names, 
        desc="Normalizing translation data"
    )
    # Free up memory
    del dataset1, dataset2, translation_ds
    gc.collect()

    
    print("Shuffling combined dataset...")
    # Shuffle the combined dataset rows randomly (which is now only the translation data)
    raw_dataset = raw_dataset.shuffle(seed=42)
    print(f"Total combined examples (only translation): {len(raw_dataset)}")

    # --- Tokenizer Training/Loading ---
    tokenizer_name = f"trans_only_{len(raw_dataset)}" # Changed name for the translation-only tokenizer
    tokenizer = get_or_train_unified_tokenizer(raw_dataset, tokenizer_name)
    VOCAB_SIZE = tokenizer.get_vocab_size()
    print(f"Unified Vocab Size: {VOCAB_SIZE}")

    # --- Model Initialization ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = MyTransformerModel(
        VOCAB_SIZE, D_MODEL, NHEAD,
        NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, DROPOUT
    ).to(device)

    print("Skipping torch.compile() on Windows due to Triton incompatibility.")

    # --- Load Checkpoint (Resume Training) ---
    model_path = 'my_transformer_brain_trans_only.pth' # Changed checkpoint name
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path} to continue training...")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load model weights. Error: {e}")
            print("Starting fresh.")
    else:
        print(f"No existing model found at '{model_path}'. Starting a new training session...")

    # --- Loss Function, Optimizer, and Scaler ---
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX) 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler(device='cuda', enabled=(device.type == 'cuda'))

    # --- 4. The Training Loop ---
    
    print("Creating UnifiedDataset object...")
    train_dataset = UnifiedDataset(raw_dataset, tokenizer)
    
    print("Creating DataLoader...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        collate_fn=collate_batch,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    print("Starting training...")
    try:
        try:
            from tqdm import tqdm
            train_iterator = tqdm(train_loader, desc=f"Epoch 1/{NUM_EPOCHS}", unit="batch", leave=True) 
        except ImportError:
            print("tqdm not installed, using basic iterator. Run 'pip install tqdm' for progress bar.")
            train_iterator = train_loader

        # Loop over batches provided by the DataLoader
        for i, batch_data in enumerate(train_iterator): 
            # Placeholder for the training steps from the original script
            if i % 1000 == 0:
                 torch.save(model.state_dict(), model_path)
                 print(f"Saving checkpoint to {model_path}")
            pass
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving model checkpoint...")
        torch.save(model.state_dict(), model_path)
        print(f"Final model checkpoint saved to {model_path}")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        torch.save(model.state_dict(), model_path)
        print(f"Final model checkpoint saved to {model_path} after error.")
