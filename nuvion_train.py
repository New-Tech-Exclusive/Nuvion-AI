import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import math
import os
from datasets import load_dataset, Dataset as HFDataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm
import re
import random
import time
import numpy as np
from collections import defaultdict
import json

# ========================
# ADVANCED CONFIGURATION
# ========================
DEV_MODE = True
DEV_SAMPLE_PER_DS = 8000 if DEV_MODE else None
MAX_SEQ_LEN = 131072 if not DEV_MODE else 4096
BATCH_SIZE = 1 if DEV_MODE else 4
GRAD_ACCUM_STEPS = 32 if DEV_MODE else 64
NUM_EPOCHS = 10 if DEV_MODE else 50
LEARNING_RATE = 3e-4 if DEV_MODE else 1e-4
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 2000 if DEV_MODE else 10000

# Enhanced Model Architecture
if DEV_MODE:
    D_MODEL = 1536
    NHEAD = 24
    NUM_LAYERS = 32
    DIM_FEEDFORWARD = 6144
    MODEL_SIZE = "1.2B"
    NUM_EXPERTS = 16
    EXPERTS_PER_TOKEN = 4
    VOCAB_SIZE = 64000
else:
    D_MODEL = 4096
    NHEAD = 32
    NUM_LAYERS = 48
    DIM_FEEDFORWARD = 16384
    MODEL_SIZE = "8.5B"
    NUM_EXPERTS = 32
    EXPERTS_PER_TOKEN = 6
    VOCAB_SIZE = 128000

DROPOUT = 0.1
USE_DISTRIBUTED = not DEV_MODE and torch.cuda.device_count() > 1

# Advanced Feature Flags
USE_GRADIENT_CHECKPOINTING = True
USE_MOE = False
USE_LONGROPE = True
USE_FLASH_ATTENTION = True
USE_LORA = False
USE_QUANTIZATION = False
USE_RAG = True
USE_SPECULATIVE_DECODING = True
USE_CHAIN_OF_THOUGHT = True
USE_CONTRASTIVE_DECODING = True
USE_SAFETY_FILTER = True
USE_DPO = True

LORA_RANK = 16
LORA_ALPHA = 32
EXPERT_CAPACITY_FACTOR = 1.25
AUX_LOSS_COEFF = 0.01

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_TOKENS = ["<unk>", "<pad>", "<s>", "</s>"]

# Enhanced Markers
USER_MARKER = "<|user|>"
ASSISTANT_MARKER = "<|assistant|>"
REASONING_START_MARKER = "<|reasoning|>"
REASONING_END_MARKER = "<|end_reasoning|>"
COT_MARKER = "<|cot|>"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Nuvion Pro | Device={device} | Model: ~{MODEL_SIZE} | Context: {MAX_SEQ_LEN}")

# Initialize mixed precision scaler for RTX 3070
scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

# Metrics tracking
METRICS_LOG = defaultdict(list)


# ========================
# MEMORY OPTIMIZATIONS 
# ========================
def setup_memory_optimizations():
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True  # Optimizes for fixed input sizes
    torch.autograd.set_detect_anomaly(False)  # Disable for performance

    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        print("✅ Flash attention available")

    torch.set_float32_matmul_precision('medium')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Print GPU info
        gpu_props = torch.cuda.get_device_properties(device)
        print(f"🎯 GPU: {gpu_props.name} | Memory: {gpu_props.total_memory / 1024 ** 3:.1f}GB | TF32: Enabled")


# ========================
# CORE COMPONENTS
# ========================

class LongRoPE(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000, scaling_factor=1.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.scaling_factor = scaling_factor

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        inv_freq = inv_freq / scaling_factor
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_cache(self, seq_len, device):
        if seq_len != self._seq_len_cached or self._cos_cached is None:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()[None, :, None, :]
            self._sin_cached = emb.sin()[None, :, None, :]
        return self._cos_cached, self._sin_cached

    def apply_rotary_pos_emb(self, q, k):
        batch, seq_len, heads, dim = q.shape
        cos, sin = self._update_cos_sin_cache(seq_len, q.device)

        cos = cos[..., :dim]
        sin = sin[..., :dim]

        q_rot = q.reshape(batch, seq_len, heads, dim // 2, 2)
        k_rot = k.reshape(batch, seq_len, heads, dim // 2, 2)

        q_rot_real = q_rot[..., 0] * cos[..., ::2] - q_rot[..., 1] * sin[..., ::2]
        q_rot_imag = q_rot[..., 0] * sin[..., ::2] + q_rot[..., 1] * cos[..., ::2]
        k_rot_real = k_rot[..., 0] * cos[..., ::2] - k_rot[..., 1] * sin[..., ::2]
        k_rot_imag = k_rot[..., 0] * sin[..., ::2] + k_rot[..., 1] * cos[..., ::2]

        q = torch.stack([q_rot_real, q_rot_imag], dim=-1).reshape(batch, seq_len, heads, dim)
        k = torch.stack([k_rot_real, k_rot_imag], dim=-1).reshape(batch, seq_len, heads, dim)
        return q, k


class HybridPositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000, num_heads=NHEAD):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len  # Reduced for memory
        self.num_heads = num_heads

        self.longrope = LongRoPE(dim, max_seq_len, base, scaling_factor=2.0)

        # Memory-efficient ALiBi implementation
        self.register_buffer('alibi_slopes', self._get_slopes(num_heads))

    def _get_slopes(self, n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return torch.tensor(get_slopes_power_of_2(n))
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return torch.tensor(
                get_slopes_power_of_2(closest_power_of_2) +
                get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]
            )

    def get_alibi_bias(self, seq_len, device):
        # Dynamic ALiBi computation to save memory
        context_position = torch.arange(seq_len, device=device)[:, None]
        memory_position = torch.arange(seq_len, device=device)[None, :]
        relative_position = torch.abs(memory_position - context_position)

        # Use the precomputed slopes
        slopes = self.alibi_slopes.to(device)
        alibi_bias = relative_position * slopes.view(-1, 1, 1) * -1.0

        return alibi_bias

    def apply_rotary_pos_emb(self, q, k):
        return self.longrope.apply_rotary_pos_emb(q, k)


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * torch.nn.functional.silu(gate)


class AdaptiveExpert(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1, expert_type="general"):
        super().__init__()
        self.expert_type = expert_type

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward * 2),
            SwiGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        self.specialization_gate = nn.Linear(d_model, 1)
        self.usage_count = 0

    def forward(self, x, importance_scores=None):
        output = self.ffn(x)
        self.usage_count += x.size(0)
        return output


class AdaptiveMixtureOfExperts(nn.Module):
    def __init__(self, d_model, dim_feedforward, num_experts=NUM_EXPERTS,
                 experts_per_token=EXPERTS_PER_TOKEN, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.d_model = d_model

        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.router_temp = nn.Parameter(torch.ones(1))

        expert_domains = ["reasoning", "code", "knowledge", "creative", "technical",
                          "conversational", "mathematical", "scientific"] * (num_experts // 8 + 1)

        self.experts = nn.ModuleList([
            AdaptiveExpert(d_model, dim_feedforward, dropout, expert_type=expert_domains[i])
            for i in range(num_experts)
        ])

        self.register_buffer('expert_usage', torch.zeros(num_experts))
        self.register_buffer('expert_importance', torch.ones(num_experts))

    def forward(self, x, importance_scores=None):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.reshape(-1, d_model)

        router_logits = self.router(x_flat) / self.router_temp
        router_probs = torch.softmax(router_logits, dim=-1)

        if self.training:
            noise = torch.randn_like(router_logits) * 0.01
            router_logits = router_logits + noise

        expert_weights, expert_indices = torch.topk(router_logits, self.experts_per_token, dim=-1)
        expert_weights = torch.softmax(expert_weights, dim=-1)

        output = torch.zeros_like(x_flat)

        for i, expert in enumerate(self.experts):
            expert_mask = (expert_indices == i).any(dim=-1)

            if expert_mask.any():
                expert_input = x_flat[expert_mask]
                weights_mask = expert_indices[expert_mask] == i
                weights = expert_weights[expert_mask][weights_mask].unsqueeze(-1)

                expert_output = expert(expert_input, importance_scores)
                output[expert_mask] += expert_output * weights

                if self.training:
                    self.expert_usage[i] += expert_mask.sum().item()
                    self.expert_importance[i] = router_probs[:, i].mean().item()

        return output.reshape(batch_size, seq_len, d_model)

    def get_advanced_load_balancing_loss(self):
        if not self.training:
            return 0.0

        total_usage = self.expert_usage.sum()
        if total_usage == 0:
            return 0.0

        target_usage = total_usage / self.num_experts
        usage_balance_loss = ((self.expert_usage - target_usage) ** 2).mean() / (target_usage ** 2 + 1e-8)
        importance_balance_loss = torch.var(self.expert_importance)

        self.expert_usage.zero_()
        return (usage_balance_loss + importance_balance_loss) * AUX_LOSS_COEFF


class MultiScaleAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, use_hybrid_pe=True):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_hybrid_pe = use_hybrid_pe

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, self.head_dim)
        self.v_proj = nn.Linear(d_model, self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

        if use_hybrid_pe:
            self.hybrid_pe = HybridPositionalEncoding(self.head_dim, num_heads=num_heads)

        self.use_flash_attention = USE_FLASH_ATTENTION

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, 1, self.head_dim).expand(-1, -1, self.num_heads, -1)
        v = self.v_proj(x).view(batch_size, seq_len, 1, self.head_dim).expand(-1, -1, self.num_heads, -1)

        if self.use_hybrid_pe:
            q, k = self.hybrid_pe.apply_rotary_pos_emb(q, k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.use_flash_attention and hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            if self.use_hybrid_pe:
                alibi_bias = self.hybrid_pe.get_alibi_bias(seq_len, x.device)
                scores = scores + alibi_bias

            if attention_mask is not None:
                scores = scores + attention_mask

            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(attn_output)


class MultiStageReasoningModule(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, num_stages=3):
        super().__init__()
        self.num_stages = num_stages

        self.reasoning_stages = nn.ModuleList([
            EnhancedTransformerBlock(d_model, nhead, dim_feedforward, dropout, use_moe=False)
            for _ in range(num_stages)
        ])

        self.cross_stage_attention = MultiScaleAttention(d_model, nhead, dropout)
        self.verification_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1),
            nn.Sigmoid()
        )

    def forward(self, x, attn_mask=None):
        reasoning_states = [x]
        reasoning_confidence = []

        for i, stage in enumerate(self.reasoning_stages):
            stage_output = stage(reasoning_states[-1], attn_mask)

            if i > 0:
                stage_output = self.cross_stage_attention(stage_output, attention_mask=attn_mask)

            reasoning_states.append(stage_output)

            if self.training:
                step_confidence = self.verification_head(stage_output.mean(dim=1))
                reasoning_confidence.append(step_confidence)

        outputs = {
            'final_output': reasoning_states[-1],
            'reasoning_states': reasoning_states,
            'reasoning_confidence': torch.stack(reasoning_confidence).mean() if reasoning_confidence else None
        }

        return outputs


class MultiStageSafetySystem(nn.Module):
    def __init__(self, d_model, num_safety_categories=8):
        super().__init__()
        self.num_safety_categories = num_safety_categories

        self.safety_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_safety_categories),
            nn.Sigmoid()
        )

        self.safety_categories = [
            "harmful", "biased", "unethical", "misinformation",
            "sensitive", "malicious", "inappropriate", "unsafe_content"
        ]

        self.confidence_calibration = nn.Parameter(torch.ones(num_safety_categories))
        self.red_team_detector = nn.Linear(d_model, 1)

    def forward(self, hidden_states):
        batch_size, seq_len, d_model = hidden_states.shape
        pooled = hidden_states.mean(dim=1)

        safety_scores = self.safety_classifier(pooled)
        safety_scores = safety_scores * self.confidence_calibration.unsqueeze(0)
        red_team_scores = torch.sigmoid(self.red_team_detector(pooled))
        overall_safety = safety_scores.min(dim=1)[0]

        return {
            'safety_scores': safety_scores,
            'overall_safety': overall_safety,
            'red_team_scores': red_team_scores
        }


class EnhancedTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, use_moe=USE_MOE):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiScaleAttention(d_model, nhead, dropout)

        self.ln2 = nn.LayerNorm(d_model)

        if use_moe:
            self.mlp = AdaptiveMixtureOfExperts(d_model, dim_feedforward, dropout=dropout)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(d_model, dim_feedforward * 2),
                SwiGLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout)
            )

        self.dropout = nn.Dropout(dropout)
        self.use_checkpoint = USE_GRADIENT_CHECKPOINTING
        self.use_moe = use_moe

    def forward(self, x, attention_mask=None):
        if self.use_checkpoint and self.training:
            return self._forward_with_checkpoint(x, attention_mask)
        else:
            return self._forward_impl(x, attention_mask)

    def _forward_impl(self, x, attention_mask=None):
        residual = x
        x = self.ln1(x)
        x = residual + self.dropout(self.attn(x, attention_mask))

        residual = x
        x = self.ln2(x)
        x = residual + self.mlp(x)

        return x

    def _forward_with_checkpoint(self, x, attention_mask):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        residual = x
        x = self.ln1(x)
        x = residual + self.dropout(
            torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.attn),
                x,
                attention_mask,
                use_reentrant=False
            )
        )

        residual = x
        x = self.ln2(x)
        x = residual + torch.utils.checkpoint.checkpoint(
            create_custom_forward(self.mlp),
            x,
            use_reentrant=False
        )

        return x

    def get_aux_loss(self):
        if self.use_moe and hasattr(self.mlp, 'get_advanced_load_balancing_loss'):
            return self.mlp.get_advanced_load_balancing_loss()
        return 0.0


class SelfReflectionModule(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.attention = MultiScaleAttention(d_model, nhead, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.quality_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1),
            nn.Sigmoid()
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1),
            nn.Sigmoid()
        )

        self.coherence_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1),
            nn.Sigmoid()
        )

    def forward(self, x, attention_mask=None):
        attn_out = self.attention(self.ln1(x), attention_mask)
        pooled = attn_out.mean(dim=1)

        quality = self.quality_head(pooled)
        confidence = self.confidence_head(pooled)
        coherence = self.coherence_head(pooled)

        return quality, confidence, coherence


class PreferenceHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.preference_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1)
        )

        self.helpfulness_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, hidden_states):
        pooled = hidden_states.mean(dim=1)
        preference = self.preference_scorer(pooled)
        helpfulness = self.helpfulness_scorer(pooled)
        return preference, helpfulness


class SimpleRAG:
    def __init__(self, embedding_dim=D_MODEL, max_docs=1000):
        self.embedding_dim = embedding_dim
        self.documents = []
        self.embeddings = []
        self.max_docs = max_docs

    def add_document(self, text, embedding):
        if len(self.documents) >= self.max_docs:
            self.documents.pop(0)
            self.embeddings.pop(0)

        self.documents.append(text)
        self.embeddings.append(embedding.cpu().numpy())

    def retrieve(self, query_embedding, top_k=3):
        if not self.embeddings:
            return []

        query_emb = query_embedding.cpu().numpy()
        embeddings_array = np.array(self.embeddings)

        similarities = np.dot(embeddings_array, query_emb) / (
                np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(query_emb) + 1e-8
        )

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]


class SafetyFilter:
    def __init__(self):
        self.harmful_patterns = [
            r'\b(kill|murder|harm|attack|weapon|bomb|terrorist)\b',
            r'\b(hate|racist|discriminat)\w*\b',
            r'\b(illegal|drug|cocaine|heroin)\b',
            r'\b(porn|sexual|explicit)\b',
        ]
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.harmful_patterns]

    def is_safe(self, text):
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                return False
        return True

    def filter_response(self, text):
        if self.is_safe(text):
            return text
        return "I apologize, but I cannot provide that response."


class ContrastiveDecoder:
    def __init__(self, alpha=0.5, beta=0.5):
        self.alpha = alpha
        self.beta = beta

    def decode(self, expert_logits, amateur_logits=None):
        if amateur_logits is None:
            return expert_logits

        expert_probs = torch.softmax(expert_logits / self.alpha, dim=-1)
        amateur_probs = torch.softmax(amateur_logits / self.beta, dim=-1)

        contrastive_logits = torch.log(expert_probs + 1e-10) - torch.log(amateur_probs + 1e-10)
        return contrastive_logits


class EvaluationSuite:
    def __init__(self):
        self.metrics = defaultdict(list)

    def compute_bleu(self, reference, hypothesis):
        ref_tokens = set(reference.lower().split())
        hyp_tokens = hypothesis.lower().split()

        if not hyp_tokens:
            return 0.0

        matches = sum(1 for token in hyp_tokens if token in ref_tokens)
        return matches / len(hyp_tokens)

    def compute_rouge_l(self, reference, hypothesis):
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()

        if not ref_tokens or not hyp_tokens:
            return 0.0

        m, n = len(ref_tokens), len(hyp_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs = dp[m][n]
        precision = lcs / n if n > 0 else 0
        recall = lcs / m if m > 0 else 0

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def detect_hallucination(self, text, context=""):
        uncertainty_markers = ["i think", "maybe", "possibly", "i'm not sure", "i don't know"]
        has_uncertainty = any(marker in text.lower() for marker in uncertainty_markers)

        if not context and any(word in text.lower() for word in ["exactly", "precisely", "definitely"]):
            return True

        return False

    def log_metrics(self, metric_name, value):
        self.metrics[metric_name].append(value)

    def get_summary(self):
        summary = {}
        for name, values in self.metrics.items():
            if values:
                summary[name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        return summary


class AdvancedRepetitionPenalty:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.repetition_analysis_window = 50
        self.ngram_penalty_range = (2, 6)

        self.penalty_config = {
            'immediate_repetition': 2.5,
            'ngram_repetition': 1.8,
            'semantic_repetition': 1.5,
            'structural_repetition': 1.6,
            'topic_stagnation': 1.3,
            'response_pattern': 1.4,
            'lexical_monotony': 1.2,
            'rhythmic_pattern': 1.1,
        }

        self.conversation_history = []
        self.ngram_frequency = {}
        self.structural_patterns = []
        self.topic_evolution = []
        self.vocabulary_usage = set()

    def analyze_conversation_patterns(self, text, current_tokens):
        analysis = {
            'immediate_repetition_score': 0.0,
            'ngram_repetition_score': 0.0,
            'semantic_repetition_score': 0.0,
            'structural_repetition_score': 0.0,
            'topic_stagnation_score': 0.0,
            'response_pattern_score': 0.0,
            'lexical_monotony_score': 0.0,
            'rhythmic_pattern_score': 0.0,
        }

        if not self.conversation_history:
            return analysis

        recent_tokens = current_tokens[-self.repetition_analysis_window:]

        analysis['immediate_repetition_score'] = self._analyze_immediate_repetition(current_tokens)
        analysis['ngram_repetition_score'] = self._analyze_ngram_repetition(recent_tokens)
        analysis['semantic_repetition_score'] = self._analyze_semantic_repetition(text)
        analysis['structural_repetition_score'] = self._analyze_structural_patterns(text)
        analysis['topic_stagnation_score'] = self._analyze_topic_stagnation(text)
        analysis['lexical_monotony_score'] = self._analyze_lexical_monotony(text)

        return analysis

    def _analyze_immediate_repetition(self, current_tokens):
        if len(current_tokens) < 10:
            return 0.0
        recent = current_tokens[-10:]
        token_counts = {}
        for token in recent:
            token_counts[token] = token_counts.get(token, 0) + 1
        max_rep = max(token_counts.values()) if token_counts else 1
        return min(1.0, (max_rep - 1) / 9.0)

    def _analyze_ngram_repetition(self, tokens):
        if len(tokens) < self.ngram_penalty_range[1]:
            return 0.0
        scores = []
        for n in range(self.ngram_penalty_range[0], self.ngram_penalty_range[1] + 1):
            if len(tokens) >= n:
                score = self._calculate_ngram_score(tokens, n)
                scores.append(score)
        return max(scores) if scores else 0.0

    def _calculate_ngram_score(self, tokens, n):
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i + n]))

        ngram_counts = {}
        for ngram in ngrams:
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

        repeated = sum(count - 1 for count in ngram_counts.values() if count > 1)
        max_possible = len(ngrams) - len(set(ngrams))

        return repeated / max_possible if max_possible > 0 else 0.0

    def _analyze_semantic_repetition(self, text):
        if len(self.conversation_history) < 2:
            return 0.0
        current_words = set(text.lower().split())
        max_sim = 0.0
        for prev_text in self.conversation_history[-5:]:
            prev_words = set(prev_text.lower().split())
            if current_words and prev_words:
                similarity = len(current_words & prev_words) / len(current_words | prev_words)
                max_sim = max(max_sim, similarity)
        return max_sim

    def _analyze_structural_patterns(self, text):
        if len(self.structural_patterns) < 2:
            return 0.0
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < 2:
            return 0.0

        current_patterns = []
        for sentence in sentences[:3]:
            words = sentence.split()
            if len(words) > 3:
                current_patterns.append(tuple(words[:3]))

        pattern_matches = sum(1 for p in current_patterns if p in self.structural_patterns)
        return pattern_matches / len(current_patterns) if current_patterns else 0.0

    def _analyze_topic_stagnation(self, text):
        if len(self.topic_evolution) < 3:
            return 0.0
        current_topic = set([w for w in text.lower().split()
                             if len(w) > 4 and w not in ['the', 'and', 'that', 'this', 'with']])

        topic_overlap = 0
        for topic_set in self.topic_evolution[-3:]:
            if current_topic and topic_set:
                overlap = len(current_topic & topic_set) / len(current_topic)
                topic_overlap = max(topic_overlap, overlap)
        return topic_overlap

    def _analyze_lexical_monotony(self, text):
        words = text.lower().split()
        if len(words) < 10:
            return 0.0
        unique = set(words)
        diversity = len(unique) / len(words)
        return 1.0 - diversity

    def calculate_dynamic_penalty(self, logits, input_ids, current_tokens, generated_text):
        analysis = self.analyze_conversation_patterns(generated_text, current_tokens)
        penalty_tensor = torch.zeros_like(logits)

        for penalty_type, base_penalty in self.penalty_config.items():
            score_key = f"{penalty_type}_score"
            if score_key in analysis:
                strength = base_penalty * analysis[score_key]
                self._apply_penalty_by_type(penalty_tensor, current_tokens, penalty_type, strength)

        self._update_conversation_tracking(generated_text, current_tokens)
        return penalty_tensor

    def _apply_penalty_by_type(self, penalty_tensor, current_tokens, penalty_type, strength):
        if not current_tokens:
            return

        recent = current_tokens[-self.repetition_analysis_window:]

        if penalty_type == 'immediate_repetition':
            token_counts = {}
            for token in recent[-10:]:
                token_counts[token] = token_counts.get(token, 0) + 1
            for token_id, count in token_counts.items():
                if count > 1 and token_id < penalty_tensor.size(-1):
                    penalty_tensor[..., token_id] += strength * (count - 1)

        elif penalty_type == 'ngram_repetition':
            for n in range(self.ngram_penalty_range[0], self.ngram_penalty_range[1] + 1):
                if len(recent) >= n:
                    self._penalize_ngram_tokens(penalty_tensor, recent, n, strength)

    def _penalize_ngram_tokens(self, penalty_tensor, tokens, n, strength):
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i + n]))

        ngram_counts = {}
        for ngram in ngrams:
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

        for ngram, count in ngram_counts.items():
            if count > 1:
                for token_id in ngram:
                    if token_id < penalty_tensor.size(-1):
                        penalty_tensor[..., token_id] += strength * (count - 1) / n

    def _update_conversation_tracking(self, text, current_tokens):
        self.conversation_history.append(text)
        if len(self.conversation_history) > 20:
            self.conversation_history.pop(0)

        for n in range(2, 5):
            if len(current_tokens) >= n:
                for i in range(len(current_tokens) - n + 1):
                    ngram = tuple(current_tokens[i:i + n])
                    self.ngram_frequency[ngram] = self.ngram_frequency.get(ngram, 0) + 1

        sentences = [s.strip() for s in text.split('.') if s.strip()]
        for sentence in sentences[:3]:
            words = sentence.split()[:3]
            if len(words) == 3:
                self.structural_patterns.append(tuple(words))
                if len(self.structural_patterns) > 50:
                    self.structural_patterns.pop(0)

        topic_words = set([w for w in text.lower().split()
                           if len(w) > 4 and w not in ['the', 'and', 'that', 'this', 'with']])
        self.topic_evolution.append(topic_words)
        if len(self.topic_evolution) > 10:
            self.topic_evolution.pop(0)

        self.vocabulary_usage.update(text.lower().split())

    def get_analysis_report(self):
        if not self.conversation_history:
            return "No conversation data"

        report = ["=" * 50, "REPETITION ANALYSIS", "=" * 50]
        report.append(f"Conversation turns: {len(self.conversation_history)}")
        report.append(f"Vocabulary diversity: {len(self.vocabulary_usage)} unique words")

        if len(self.conversation_history) >= 2:
            recent = self.conversation_history[-1]
            tokens = self.tokenizer.encode(recent).ids
            analysis = self.analyze_conversation_patterns(recent, tokens)

            report.append("\n--- Current Response Analysis ---")
            for key, score in analysis.items():
                report.append(f"{key}: {score:.3f}")

            issues = []
            if analysis['immediate_repetition_score'] > 0.3:
                issues.append("High immediate token repetition")
            if analysis['ngram_repetition_score'] > 0.4:
                issues.append("Phrase repetition detected")
            if analysis['semantic_repetition_score'] > 0.6:
                issues.append("Semantic similarity with previous")
            if analysis['topic_stagnation_score'] > 0.7:
                issues.append("Topic stagnation")

            if issues:
                report.append("\n⚠️ Detected Issues:")
                for issue in issues:
                    report.append(f"  - {issue}")
            else:
                report.append("\n✅ No significant repetition issues")

        return "\n".join(report)


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=LORA_RANK, alpha=LORA_ALPHA):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False

        if rank > 0:
            self.lora_A = nn.Parameter(torch.randn(rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
            self.scaling = alpha / rank

        self.bias = None

    def forward(self, x):
        result = torch.nn.functional.linear(x, self.weight, self.bias)

        if self.rank > 0 and self.training:
            lora_result = torch.nn.functional.linear(
                torch.nn.functional.linear(x, self.lora_A),
                self.lora_B
            )
            result = result + lora_result * self.scaling

        return result

    def merge_weights(self):
        if self.rank > 0:
            self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.lora_A = None
            self.lora_B = None


class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bits=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits

        self.register_buffer('weight_quantized', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('scale', torch.ones(out_features))
        self.register_buffer('zero_point', torch.zeros(out_features))

    def quantize_weight(self, weight):
        min_val = weight.min(dim=1, keepdim=True)[0]
        max_val = weight.max(dim=1, keepdim=True)[0]

        scale = (max_val - min_val) / (2 ** self.bits - 1)
        zero_point = -min_val / scale

        quantized = torch.clamp(
            torch.round(weight / scale + zero_point),
            0, 2 ** self.bits - 1
        ).to(torch.int8)

        self.weight_quantized.copy_(quantized)
        self.scale.copy_(scale.squeeze())
        self.zero_point.copy_(zero_point.squeeze())

    def forward(self, x):
        weight = (self.weight_quantized.float() - self.zero_point.unsqueeze(1)) * self.scale.unsqueeze(1)
        return torch.nn.functional.linear(x, weight)


class NuvionPro(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.vocab_size = vocab_size
        self.model_size = MODEL_SIZE
        self.d_model = D_MODEL

        self.token_embedding = nn.Embedding(vocab_size, D_MODEL)
        self.embed_dropout = nn.Dropout(DROPOUT)

        self.blocks = nn.ModuleList([
            EnhancedTransformerBlock(D_MODEL, NHEAD, DIM_FEEDFORWARD, DROPOUT)
            for _ in range(NUM_LAYERS)
        ])

        self.reasoning_module = MultiStageReasoningModule(D_MODEL, NHEAD, DIM_FEEDFORWARD, DROPOUT)
        self.safety_system = MultiStageSafetySystem(D_MODEL)
        self.reflection_module = SelfReflectionModule(D_MODEL, NHEAD, DIM_FEEDFORWARD, DROPOUT)
        self.preference_head = PreferenceHead(D_MODEL)

        self.ln_f = nn.LayerNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.rag_system = SimpleRAG() if USE_RAG else None
        self.safety_filter = SafetyFilter() if USE_SAFETY_FILTER else None
        self.contrastive_decoder = ContrastiveDecoder() if USE_CONTRASTIVE_DECODING else None
        self.evaluator = EvaluationSuite()

        self.role_penalty_tokens = set()
        self.uncertainty_penalty_tokens = set()
        self.advanced_penalty_system = None

        self._init_weights()
        param_count = sum(p.numel() for p in self.parameters())
        trainable_count = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"✅ Nuvion Pro Model: {param_count:,} parameters (~{param_count / 1e9:.1f}B)")
        print(f"   Trainable: {trainable_count:,} parameters")
        print(f"   Advanced Features:")
        print(f"     • Context: {MAX_SEQ_LEN} tokens (Hybrid LongRoPE+ALiBi)")
        print(f"     • MoE: {NUM_EXPERTS} experts with adaptive routing")
        print(f"     • Reasoning: {self.reasoning_module.num_stages}-stage deep reasoning")
        print(f"     • Safety: Multi-category learned safety system")

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * NUM_LAYERS))
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)

    def set_penalty_tokens(self, tokenizer):
        penalty_words = ["USER", "ASSISTANT", "USER:", "ASSISTANT:", "Human:", "Assistant:"]
        self.role_penalty_tokens.clear()

        for word in penalty_words:
            try:
                tokens = tokenizer.encode(word).ids
                self.role_penalty_tokens.update(tokens)
            except:
                continue

        uncertainty_words = [
            "don't", "know", "unsure", "uncertain", "cannot", "can't", "unable",
            "not sure", "no idea", "don't know", "not certain"
        ]
        self.uncertainty_penalty_tokens.clear()

        for word in uncertainty_words:
            try:
                tokens = tokenizer.encode(word).ids
                self.uncertainty_penalty_tokens.update(tokens)
            except:
                continue

        print(
            f"🔨 Penalty tokens: Role={len(self.role_penalty_tokens)}, Uncertainty={len(self.uncertainty_penalty_tokens)}")

    def init_advanced_penalty(self, tokenizer):
        self.advanced_penalty_system = AdvancedRepetitionPenalty(tokenizer)
        print("✅ Advanced repetition penalty initialized")

    def apply_comprehensive_penalties(self, logits, input_ids, current_tokens=None,
                                      generated_text="", penalty_strength=3.0,
                                      repetition_penalty=1.2, recent_tokens=None):
        penalty_mask = torch.zeros_like(logits)

        for token_id in self.role_penalty_tokens:
            if token_id < logits.size(-1):
                penalty_mask[:, :, token_id] = penalty_strength

        for token_id in self.uncertainty_penalty_tokens:
            if token_id < logits.size(-1):
                penalty_mask[:, :, token_id] = penalty_strength * 0.7

        if repetition_penalty > 1.0 and recent_tokens is not None:
            for token_id in recent_tokens:
                if token_id < logits.size(-1):
                    penalty_mask[:, :, token_id] += repetition_penalty

        if (self.advanced_penalty_system and current_tokens and len(current_tokens) > 10):
            try:
                advanced_penalty = self.advanced_penalty_system.calculate_dynamic_penalty(
                    logits, input_ids, current_tokens, generated_text
                )
                penalty_mask += advanced_penalty
            except Exception as e:
                pass

        return logits - penalty_mask

    def forward(self, input_ids, attention_mask=None, use_reasoning=False,
                return_preference_score=False, return_reflection=False, use_rag=False):
        batch_size, seq_len = input_ids.shape

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device) * float('-inf'),
            diagonal=1
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.embed_dropout(x)

        aux_losses = []
        for block in self.blocks:
            x = block(x, causal_mask)
            aux_loss = block.get_aux_loss()
            if aux_loss > 0:
                aux_losses.append(aux_loss)

        if use_reasoning:
            reasoning_outputs = self.reasoning_module(x, causal_mask)
            x = reasoning_outputs['final_output']

        x = self.ln_f(x)

        outputs = {}
        outputs['hidden_states'] = x
        outputs['logits'] = self.lm_head(x)

        if aux_losses:
            outputs['aux_loss'] = sum(aux_losses) / len(aux_losses)

        if use_reasoning and 'reasoning_confidence' in reasoning_outputs:
            outputs['reasoning_confidence'] = reasoning_outputs['reasoning_confidence']

        safety_outputs = self.safety_system(x)
        outputs.update(safety_outputs)

        if return_preference_score:
            pref_score, help_score = self.preference_head(x)
            outputs['preference_score'] = pref_score
            outputs['helpfulness_score'] = help_score

        if return_reflection:
            quality, confidence, coherence = self.reflection_module(x, causal_mask)
            outputs['quality_score'] = quality
            outputs['confidence_score'] = confidence
            outputs['coherence_score'] = coherence

        return outputs

    def generate(self, prompt, tokenizer, max_new_tokens=200, temperature=0.8,
                 top_k=50, top_p=0.92, repetition_penalty=1.15, penalty_strength=3.0,
                 advanced_penalties=True, use_reasoning=False, reasoning_steps=3,
                 use_reflection=True, reflection_threshold=0.7,
                 use_contrastive=USE_CONTRASTIVE_DECODING, use_rag=USE_RAG,
                 apply_safety=USE_SAFETY_FILTER):
        self.eval()

        if self.advanced_penalty_system is None and advanced_penalties:
            self.init_advanced_penalty(tokenizer)

        if use_reasoning and USE_CHAIN_OF_THOUGHT:
            reasoning_prompt = f"{prompt}\n\nLet me think through this step by step:"
            full_prompt = f"<s>{USER_MARKER}{reasoning_prompt}\n\n{ASSISTANT_MARKER}{REASONING_START_MARKER}{COT_MARKER}"
        else:
            if not prompt.startswith("<s>"):
                full_prompt = f"<s>{USER_MARKER}{prompt}\n\n{ASSISTANT_MARKER}"
            else:
                full_prompt = prompt

        input_ids = tokenizer.encode(full_prompt).ids

        if len(input_ids) > MAX_SEQ_LEN - max_new_tokens:
            input_ids = input_ids[-(MAX_SEQ_LEN - max_new_tokens):]

        input_ids = torch.tensor([input_ids], device=device)
        generated = []
        recent_tokens = []
        full_generated_tokens = []
        current_text = ""

        in_reasoning_phase = use_reasoning
        reasoning_count = 0
        reflection_attempts = 0

        start_time = time.time()

        if use_reasoning:
            print("🤔 Thinking...", end="", flush=True)

        with torch.no_grad():
            for step in range(max_new_tokens):
                if input_ids.shape[1] > MAX_SEQ_LEN:
                    input_ids = input_ids[:, -MAX_SEQ_LEN:]

                apply_reasoning = (step < reasoning_steps * 10) and in_reasoning_phase
                outputs = self.forward(input_ids, use_reasoning=apply_reasoning,
                                       return_reflection=use_reflection, use_rag=use_rag)

                logits = outputs['logits']
                next_logits = logits[0, -1, :] / temperature

                if use_contrastive and self.contrastive_decoder:
                    amateur_logits = logits[0, -1, :] / (temperature * 1.5)
                    next_logits = self.contrastive_decoder.decode(next_logits, amateur_logits)

                if use_reflection and step > 10 and step % 20 == 0:
                    quality = outputs.get('quality_score', torch.tensor([1.0]))
                    confidence = outputs.get('confidence_score', torch.tensor([1.0]))
                    coherence = outputs.get('coherence_score', torch.tensor([1.0]))

                    avg_score = (quality.item() + confidence.item() + coherence.item()) / 3

                    if avg_score < reflection_threshold:
                        reflection_attempts += 1
                        if reflection_attempts < 3:
                            temperature *= 0.9
                            top_p *= 0.95
                            print("🔄", end="", flush=True)

                if (self.role_penalty_tokens or self.uncertainty_penalty_tokens or
                        repetition_penalty > 1.0 or advanced_penalties):
                    next_logits = self.apply_comprehensive_penalties(
                        next_logits.unsqueeze(0).unsqueeze(0),
                        input_ids,
                        current_tokens=full_generated_tokens,
                        generated_text=current_text,
                        penalty_strength=penalty_strength,
                        repetition_penalty=repetition_penalty,
                        recent_tokens=set(recent_tokens)
                    ).squeeze()

                if top_k > 0:
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                    next_logits[indices_to_remove] = float('-inf')

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_logits[indices_to_remove] = float('-inf')

                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                token_id = next_token.item()

                if token_id in [EOS_IDX, PAD_IDX]:
                    break

                current_text = tokenizer.decode(generated + [token_id])

                if REASONING_END_MARKER in current_text or reasoning_count >= reasoning_steps * 15:
                    in_reasoning_phase = False
                    if use_reasoning:
                        print("✓", end="", flush=True)

                if in_reasoning_phase:
                    reasoning_count += 1
                    if step % 5 == 0:
                        print(".", end="", flush=True)

                generated.append(token_id)
                full_generated_tokens.append(token_id)
                recent_tokens.append(token_id)
                if len(recent_tokens) > 100:
                    recent_tokens.pop(0)

                current_text = tokenizer.decode(full_generated_tokens)
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        generation_time = time.time() - start_time

        if use_reasoning:
            print()

        result = tokenizer.decode(generated)
        result = result.replace("<s>", "").replace("</s>", "")
        result = result.replace(USER_MARKER, "").replace(ASSISTANT_MARKER, "")
        result = result.replace(REASONING_START_MARKER, "").replace(REASONING_END_MARKER, "")
        result = result.replace(COT_MARKER, "")

        if apply_safety and self.safety_filter:
            result = self.safety_filter.filter_response(result)

        self.evaluator.log_metrics("generation_time", generation_time)
        self.evaluator.log_metrics("tokens_generated", len(generated))
        self.evaluator.log_metrics("tokens_per_second", len(generated) / generation_time if generation_time > 0 else 0)

        if advanced_penalties and self.advanced_penalty_system:
            analysis = self.advanced_penalty_system.get_analysis_report()
            print(f"\n{analysis}\n")

        print(
            f"⚡ Generated {len(generated)} tokens in {generation_time:.2f}s ({len(generated) / generation_time:.1f} tok/s)")

        return result.strip()

    def enable_lora(self, rank=LORA_RANK, alpha=LORA_ALPHA):
        print(f"🔧 Enabling LoRA (rank={rank}, alpha={alpha})")

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and 'lm_head' not in name:
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]

                parent = self
                for part in parent_name.split('.'):
                    if part:
                        parent = getattr(parent, part)

                lora_layer = LoRALinear(
                    module.in_features,
                    module.out_features,
                    rank=rank,
                    alpha=alpha
                )
                lora_layer.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    lora_layer.bias = nn.Parameter(module.bias.data.clone())

                setattr(parent, child_name, lora_layer)

        for name, param in self.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✅ LoRA enabled: {trainable:,} trainable parameters")

    def enable_quantization(self, bits=8):
        print(f"🔧 Enabling {bits}-bit quantization")

        quantized_count = 0
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and 'lm_head' not in name:
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]

                parent = self
                for part in parent_name.split('.'):
                    if part:
                        parent = getattr(parent, part)

                quant_layer = QuantizedLinear(
                    module.in_features,
                    module.out_features,
                    bits=bits
                )
                quant_layer.quantize_weight(module.weight.data)

                setattr(parent, child_name, quant_layer)
                quantized_count += 1

        print(f"✅ Quantization enabled: {quantized_count} layers quantized")

    def get_expert_statistics(self):
        if not USE_MOE:
            return {}

        stats = {}
        for i, block in enumerate(self.blocks):
            if hasattr(block, 'mlp') and isinstance(block.mlp, AdaptiveMixtureOfExperts):
                for j, expert in enumerate(block.mlp.experts):
                    stats[f"block_{i}_expert_{j}_{expert.expert_type}"] = expert.usage_count

        return stats


# ========================
# TEXT PROCESSING
# ========================
def clean_text_improved(text):
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'([.,!?;:])\s+', r'\1 ', text)
    text = re.sub(r'[^\w\s\.\?\!,;:\-\'"\(\)]', '', text)
    text = text.replace('\u2019', "'").replace('`', "'")
    text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)
    return text.strip()


def contains_uncertainty_phrase(text):
    uncertainty_phrases = [
        "i don't know", "i do not know", "i'm not sure", "i am not sure",
        "i cannot answer", "i can't answer", "no idea", "not sure", "unsure",
        "i don't have an answer", "i'm not certain", "uncertain",
        "i'm not familiar", "i don't understand", "i'm confused"
    ]
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in uncertainty_phrases)


def is_code_heavy(text):
    if not text:
        return False
    code_indicators = [
        'def ', 'class ', 'import ', 'function ', 'const ', 'let ', 'var ',
        'public ', 'private ', '#!/', '{', '}', '=>', 'return ', '#include'
    ]
    code_count = sum(1 for indicator in code_indicators if indicator in text)
    has_code_blocks = '```' in text or text.count('\n') > 10
    return code_count >= 3 or has_code_blocks


# ========================
# CHAIN-OF-THOUGHT DATA FORMATTER
# ========================
def format_chain_of_thought(question, answer):
    """Format data with explicit CoT reasoning"""
    cot_templates = [
        f"Let me break this down step by step:\n\n1. First, {answer[:len(answer) // 3]}\n2. Then, {answer[len(answer) // 3:2 * len(answer) // 3]}\n3. Finally, {answer[2 * len(answer) // 3:]}",
        f"Here's my reasoning:\n\nStep 1: {answer[:len(answer) // 2]}\nStep 2: {answer[len(answer) // 2:]}",
        f"Let me think through this:\n\n{answer}"
    ]

    template = random.choice(cot_templates)
    return f"<s>{USER_MARKER}{question}\n\n{ASSISTANT_MARKER}{COT_MARKER}{template}</s>"


# ========================
# DATASET LOADING (ENGLISH ONLY)
# ========================
def load_nuvion_datasets():
    print("🚀 Loading English datasets with CoT enhancement...")
    sample = DEV_SAMPLE_PER_DS

    all_samples = []

    dataset_configs = [
        ("HuggingFaceH4/ultrachat_200k", "train_sft", None, format_conversation, 0.18),
        ("Anthropic/hh-rlhf", "train", None, format_conversation, 0.10),
        ("allenai/openbookqa", "train", "main", format_qa, 0.08),
        ("tau/commonsense_qa", "train", None, format_qa, 0.08),
        ("google/boolq", "train", None, format_qa, 0.05),
        ("garage-bAInd/Open-Platypus", "train", None, format_instruction, 0.05),
        ("tatsu-lab/alpaca", "train", None, format_instruction, 0.08),
        ("databricks/databricks-dolly-15k", "train", None, format_instruction, 0.08),
        ("sahil2801/CodeAlpaca-20k", "train", None, format_code, 0.08),
        ("iamtarun/code_instructions_120k_alpaca", "train", None, format_code, 0.07),
        ("openai/gsm8k", "train", "main", format_math_cot, 0.15),
    ]

    for name, split, config, formatter, weight in dataset_configs:
        try:
            adjusted_sample = int(sample * weight) if sample else None
            ds = safe_load_dataset(name, split, adjusted_sample, config)

            if len(ds) > 0:
                count = 0
                for example in ds:
                    formatted = formatter(example)
                    if formatted:
                        all_samples.append(formatted)
                        count += 1
                print(f"   ↳ {count}/{len(ds)} from {name}")
        except Exception as e:
            print(f"   ⚠️ Skipped {name}: {e}")

    try:
        oa_ds = safe_load_dataset("OpenAssistant/oasst1", "train", sample, None)
        oa_samples = format_openassistant(oa_ds)
        all_samples.extend(oa_samples)
        print(f"   ↳ {len(oa_samples)} from OpenAssistant")
    except Exception as e:
        print(f"   ⚠️ Skipped OpenAssistant: {e}")

    random.shuffle(all_samples)
    print(f"\n📊 Total samples: {len(all_samples)}")
    return HFDataset.from_dict({"text": all_samples})


def format_conversation(example):
    if "messages" in example:
        msgs = []
        for msg in example["messages"]:
            role = msg.get("role", "").lower()
            content = clean_text_improved(msg.get("content", ""))
            if not content or len(content) < 10:
                continue
            if contains_uncertainty_phrase(content):
                continue
            if role in ["user", "human"]:
                msgs.append(f"{USER_MARKER}{content}")
            elif role in ["assistant", "bot"]:
                msgs.append(f"{ASSISTANT_MARKER}{content}")
        if len(msgs) >= 2:
            return f"<s>{chr(10).join(msgs)}</s>"

    if "chosen" in example:
        text = clean_text_improved(str(example["chosen"]))
        if "Human:" in text and "Assistant:" in text:
            text = text.replace("Human:", USER_MARKER).replace("Assistant:", ASSISTANT_MARKER)
            if not contains_uncertainty_phrase(text):
                return f"<s>{text}</s>"
    return None


def format_qa(example):
    q = None
    a = None

    if "question" in example and "answer" in example:
        q = clean_text_improved(str(example["question"]))
        answer_data = example["answer"]
        if isinstance(answer_data, dict):
            a = clean_text_improved(str(answer_data.get("text", answer_data.get("answer", ""))))
        else:
            a = clean_text_improved(str(answer_data))

    elif "question_stem" in example and "answerKey" in example:
        q = clean_text_improved(str(example["question_stem"]))
        answer_key = example["answerKey"]
        if "choices" in example:
            choices = example["choices"]
            if isinstance(choices, dict) and "text" in choices and "label" in choices:
                try:
                    labels = choices["label"]
                    texts = choices["text"]
                    idx = labels.index(answer_key)
                    a = clean_text_improved(texts[idx])
                except:
                    a = clean_text_improved(str(answer_key))
            else:
                a = clean_text_improved(str(answer_key))
        else:
            a = clean_text_improved(str(answer_key))

    elif "question" in example and "choices" in example and "answerKey" in example:
        q = clean_text_improved(str(example["question"]))
        answer_key = example["answerKey"]
        choices = example["choices"]
        if isinstance(choices, dict):
            try:
                labels = choices.get("label", [])
                texts = choices.get("text", [])
                if answer_key in labels:
                    idx = labels.index(answer_key)
                    a = clean_text_improved(texts[idx])
            except:
                a = None

    elif "question" in example and "passage" in example:
        q = clean_text_improved(str(example["question"]))
        answer = example.get("answer", None)
        if answer is not None:
            a = "Yes" if answer else "No"
            passage = clean_text_improved(str(example["passage"]))
            if passage and len(passage) < 500:
                q = f"{q}\n\nContext: {passage[:300]}..."

    elif "query" in example and "response" in example:
        q = clean_text_improved(str(example["query"]))
        a = clean_text_improved(str(example["response"]))

    if q and a and len(a) > 2 and not contains_uncertainty_phrase(a):
        return f"<s>{USER_MARKER}{q}\n\n{ASSISTANT_MARKER}{a}</s>"
    return None


def format_instruction(example):
    if "instruction" in example and ("output" in example or "response" in example):
        inst = clean_text_improved(str(example["instruction"]))
        out = clean_text_improved(str(example.get("output", example.get("response", ""))))

        if inst and out and len(out) > 20 and not contains_uncertainty_phrase(out):
            if "input" in example and example["input"]:
                ctx = clean_text_improved(str(example["input"]))
                if ctx:
                    inst = f"{inst}\n\n{ctx}"

            if "context" in example and example["context"]:
                ctx = clean_text_improved(str(example["context"]))
                if ctx and ctx.strip():
                    inst = f"{inst}\n\nContext: {ctx}"

            return f"<s>{USER_MARKER}{inst}\n\n{ASSISTANT_MARKER}{out}</s>"
    return None


def format_code(example):
    if "instruction" in example and "output" in example:
        inst = clean_text_improved(str(example["instruction"]))
        code = clean_text_improved(str(example["output"]))
        if inst and code and len(code) > 20:
            return f"<s>{USER_MARKER}{inst}\n\n{ASSISTANT_MARKER}{code}</s>"
    return None


def format_math_cot(example):
    """Enhanced math formatting with Chain-of-Thought"""
    if "question" in example and "answer" in example:
        q = clean_text_improved(str(example["question"]))
        a = clean_text_improved(str(example["answer"]))
        if q and a:
            if USE_CHAIN_OF_THOUGHT:
                return f"<s>{USER_MARKER}{q}\n\n{ASSISTANT_MARKER}{COT_MARKER}Let me solve this step by step:\n\n{a}</s>"
            else:
                return f"<s>{USER_MARKER}{q}\n\n{ASSISTANT_MARKER}Let me solve this step by step:\n{a}</s>"
    return None


def format_openassistant(dataset):
    conversations = []
    tree_messages = {}

    for example in dataset:
        try:
            tree_id = example.get("message_tree_id")
            if not tree_id:
                continue
            if tree_id not in tree_messages:
                tree_messages[tree_id] = []
            tree_messages[tree_id].append({
                "message_id": example.get("message_id"),
                "parent_id": example.get("parent_id"),
                "text": clean_text_improved(str(example.get("text", ""))),
                "role": example.get("role", "").lower(),
                "rank": example.get("rank", 0) or 0
            })
        except:
            continue

    for tree_id, messages in list(tree_messages.items())[:1000]:
        try:
            root = [m for m in messages if m["parent_id"] is None and m["role"] == "prompter"]
            if not root:
                continue
            root_msg = root[0]

            responses = [m for m in messages
                         if m["parent_id"] == root_msg["message_id"]
                         and m["role"] == "assistant"
                         and len(m["text"]) > 10
                         and not contains_uncertainty_phrase(m["text"])]
            if not responses:
                continue

            responses.sort(key=lambda x: x["rank"], reverse=True)
            best = responses[0]

            conv = f"{USER_MARKER}{root_msg['text']}\n\n{ASSISTANT_MARKER}{best['text']}"
            conversations.append(f"<s>{conv}</s>")
        except:
            continue

    return conversations


def safe_load_dataset(name, split="train", sample=None, config=None):
    try:
        if name == "HuggingFaceH4/ultrachat_200k" and split == "train":
            split = "train_sft"

        if config:
            ds = load_dataset(name, config, split=split)
        else:
            ds = load_dataset(name, split=split)

        if sample:
            ds = ds.select(range(min(len(ds), sample)))
        print(f"✅ Loaded {len(ds)} from {name}")
        return ds
    except Exception as e:
        print(f"⚠️ Failed {name}: {e}")
        return HFDataset.from_dict({'text': []})


# ========================
# TOKENIZER
# ========================
def get_or_train_tokenizer(dataset, path="tokenizer_nuvion.json"):
    if os.path.exists(path):
        print(f"Loading tokenizer from {path}")
        return Tokenizer.from_file(path)

    print("Training tokenizer...")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True
    )

    def text_iterator():
        for item in dataset:
            if item["text"]:
                yield item["text"]

    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
    tokenizer.save(path)
    print(f"✅ Tokenizer: {tokenizer.get_vocab_size()} tokens")
    return tokenizer


# ========================
# DATASET
# ========================
class NuvionDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.samples = []
        self.masks = []

        max_tokens = 1024 if DEV_MODE else 2048
        skipped_short = 0
        skipped_long = 0
        skipped_uncertainty = 0

        for item in dataset:
            text = item["text"]
            if not text:
                continue

            if contains_uncertainty_phrase(text):
                skipped_uncertainty += 1
                continue

            tokens = tokenizer.encode(text).ids
            if len(tokens) < 10:
                skipped_short += 1
                continue
            if len(tokens) > max_tokens:
                skipped_long += 1
                continue

            mask = self._create_mask(text, tokens)
            self.samples.append(tokens)
            self.masks.append(mask)

        print(f"✅ Dataset: {len(self.samples)} samples (max {max_tokens} tokens)")
        print(f"   Skipped: {skipped_short} short, {skipped_long} long, {skipped_uncertainty} uncertain")

    def _create_mask(self, text, tokens):
        mask = [0] * (len(tokens) // 2) + [1] * (len(tokens) - len(tokens) // 2)
        return mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = [BOS_IDX] + self.samples[idx] + [EOS_IDX]
        mask = [0] + self.masks[idx] + [1]
        return (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(mask, dtype=torch.float)
        )


def collate_fn(batch):
    tokens = [b[0] for b in batch]
    masks = [b[1] for b in batch]

    padded_tokens = pad_sequence(tokens, batch_first=True, padding_value=PAD_IDX)
    padded_masks = pad_sequence(masks, batch_first=True, padding_value=0.0)
    attention_mask = (padded_tokens != PAD_IDX).long()

    return padded_tokens, attention_mask, padded_masks


# ========================
# PREFERENCE OPTIMIZATION LOSS
# ========================
def preference_loss(policy_chosen_logps, policy_rejected_logps,
                    reference_chosen_logps, reference_rejected_logps, beta=0.1):
    """Direct Preference Optimization loss"""
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    reference_logratios = reference_chosen_logps - reference_rejected_logps

    losses = -torch.nn.functional.logsigmoid(
        beta * (policy_logratios - reference_logratios)
    )

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)

    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()


# ========================
# SCHEDULER
# ========================
class CosineWithRestartsScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6, restart_steps=None):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.step_num = 0
        self.restart_steps = restart_steps or (total_steps // 3)

    def step(self):
        self.step_num += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _get_lr(self):
        if self.step_num < self.warmup_steps:
            return self.base_lr * (self.step_num / self.warmup_steps)

        progress = (self.step_num - self.warmup_steps) % self.restart_steps
        total = self.restart_steps
        cosine = 0.5 * (1 + math.cos(math.pi * progress / total))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine


# ========================
# MODEL LOADING
# ========================
def find_existing_model():
    model_files = [
        "nuvion_1.2b_best.pth",
        "nuvion_8.5b_best.pth",
        "nuvion_pro_best.pth",
    ]
    for f in model_files:
        if os.path.exists(f):
            print(f"📂 Found: {f}")
            return f
    return None


def load_existing_model(model, path, optimizer=None):
    try:
        checkpoint = torch.load(path, map_location=device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"✅ Loaded model from {path}")

            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("✅ Loaded optimizer state")

            start_epoch = checkpoint.get('epoch', 0) + 1
            best_loss = checkpoint.get('loss', float('inf'))
            print(f"📊 Resuming from epoch {start_epoch}, loss: {best_loss:.4f}")
            return start_epoch, best_loss
        else:
            model.load_state_dict(checkpoint, strict=False)
            return 0, float('inf')
    except Exception as e:
        print(f"⚠️ Error loading: {e}")
        return 0, float('inf')


# ========================
# TESTING WITH EVALUATION SUITE
# ========================
def test_model_capabilities(model, tokenizer):
    model.eval()
    test_texts = ["Hello", "The cat", "What is", "How to"]

    print("   Pre-training test:")
    for text in test_texts:
        try:
            with torch.no_grad():
                ids = tokenizer.encode(text).ids
                if not ids:
                    continue
                tensor = torch.tensor([ids], device=device)
                outputs = model(tensor)
                logits = outputs['logits']

                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print(f"   ❌ {text}: Invalid outputs")
                else:
                    print(f"   ✅ {text}: OK")
        except Exception as e:
            print(f"   ❌ {text}: {e}")
    model.train()


def test_generation_progressive(model, tokenizer, epoch):
    model.eval()

    if epoch <= 3:
        prompts = ["Hello", "What is AI", "Tell me"]
        temp = 0.3
    elif epoch <= 6:
        prompts = ["Explain AI", "How to code", "What causes rain"]
        temp = 0.5
    else:
        prompts = [
            "How should I handle a difficult conversation?",
            "Explain quantum computing simply",
            "Write a Python function to reverse a string"
        ]
        temp = 0.7

    print(f"\n🧪 Epoch {epoch} Generation Test:")
    evaluator = model.evaluator

    for i, prompt in enumerate(prompts, 1):
        try:
            response = model.generate(
                prompt, tokenizer,
                max_new_tokens=100,
                temperature=temp,
                advanced_penalties=False
            )
            print(f"   {i}. {prompt[:50]}...")
            print(f"      → {response[:150]}...")

            if epoch > 5:
                bleu = evaluator.compute_bleu(prompt, response)
                rouge = evaluator.compute_rouge_l(prompt, response)
                hallucination = evaluator.detect_hallucination(response, prompt)

                evaluator.log_metrics(f"bleu_epoch_{epoch}", bleu)
                evaluator.log_metrics(f"rouge_epoch_{epoch}", rouge)
                evaluator.log_metrics(f"hallucination_epoch_{epoch}", 1 if hallucination else 0)
        except Exception as e:
            print(f"   {i}. {prompt[:50]}...")
            print(f"      → ERROR: {e}")

    model.train()
    print()


# ========================
# OPTIMIZED TRAINING LOOP FOR RTX 3070
# ========================
def train_nuvion():
    print("🚀 Training Nuvion Pro")
    print("=" * 70)

    setup_memory_optimizations()

    dataset = load_nuvion_datasets()
    tokenizer = get_or_train_tokenizer(dataset)

    train_dataset = NuvionDataset(dataset, tokenizer)
    if len(train_dataset) == 0:
        print("❌ No training data")
        return

    # OPTIMIZED: Enhanced DataLoader for RTX 3070
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=4,  # Increased for better data loading
        pin_memory=True,  # Faster GPU transfers
        persistent_workers=True,  # Reuse workers
        prefetch_factor=2,  # Preload batches
        drop_last=True  # Avoid partial batches
    )

    model = NuvionPro(vocab_size=tokenizer.get_vocab_size()).to(device)
    model.set_penalty_tokens(tokenizer)

    if USE_LORA:
        model.enable_lora()

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY
    )

    total_steps = len(train_loader) * NUM_EPOCHS // GRAD_ACCUM_STEPS
    scheduler = CosineWithRestartsScheduler(optimizer, WARMUP_STEPS, total_steps)

    start_epoch = 0
    best_loss = float('inf')
    existing = find_existing_model()
    if existing:
        start_epoch, best_loss = load_existing_model(model, existing, optimizer)

    print(f"\n🎯 Advanced Training Configuration:")
    print(f"   Model: {MODEL_SIZE}")
    print(f"   Sequence length: {MAX_SEQ_LEN} tokens")
    print(f"   Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRAD_ACCUM_STEPS})")
    print(f"\n   🔥 Advanced Features Enabled:")
    print(f"      ✅ LongRoPE: {USE_LONGROPE} (up to {MAX_SEQ_LEN} tokens)")
    print(f"      ✅ MoE: {USE_MOE} ({NUM_EXPERTS} experts, {EXPERTS_PER_TOKEN} per token)")
    print(f"      ✅ FlashAttention: {USE_FLASH_ATTENTION}")
    print(f"      ✅ Chain-of-Thought: {USE_CHAIN_OF_THOUGHT}")
    print(f"      ✅ Contrastive Decoding: {USE_CONTRASTIVE_DECODING}")
    print(f"      ✅ Safety Filter: {USE_SAFETY_FILTER}")
    print(f"      ✅ RAG: {USE_RAG}")
    print(f"      ✅ Speculative Decoding: {USE_SPECULATIVE_DECODING}")
    print(f"      ✅ LoRA: {USE_LORA}")
    print(f"      ✅ Quantization: {USE_QUANTIZATION}")
    print(f"      ✅ Gradient Checkpointing: {USE_GRADIENT_CHECKPOINTING}")
    print(f"\n   🚀 RTX 3070 Optimizations:")
    print(f"      ✅ Mixed Precision Training")
    print(f"      ✅ Optimized DataLoader (num_workers=4, pin_memory=True)")
    print(f"      ✅ TF32 & cuDNN Benchmarking")
    print(f"\n   Epochs: {NUM_EPOCHS} (from {start_epoch})")
    print(f"   Samples: {len(train_dataset)}")
    print("=" * 70)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if start_epoch == 0:
        print("\n🔍 Pre-training diagnostics:")
        test_model_capabilities(model, tokenizer)

    global_step = 0

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        epoch_aux_loss = 0
        steps = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for batch_idx, (batch_inputs, attention_mask, loss_mask) in enumerate(loop):
            if batch_inputs.nelement() == 0:
                continue

            # OPTIMIZED: Non-blocking transfers
            batch_inputs = batch_inputs.to(device, non_blocking=True)
            loss_mask = loss_mask.to(device, non_blocking=True)

            inputs = batch_inputs[:, :-1]
            targets = batch_inputs[:, 1:]
            target_mask = loss_mask[:, 1:]

            # OPTIMIZED: Mixed precision training for RTX 3070
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(inputs)
                logits = outputs['logits']

                loss_fct = nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1)
                losses = loss_fct(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                losses = losses.view(targets.shape)
                masked_losses = losses * target_mask
                num_tokens = target_mask.sum()
                loss = masked_losses.sum() / num_tokens if num_tokens > 0 else masked_losses.sum()

                if 'aux_loss' in outputs:
                    aux_loss = outputs['aux_loss']
                    loss = loss + aux_loss
                    epoch_aux_loss += aux_loss.item()

                loss = loss / GRAD_ACCUM_STEPS

            # OPTIMIZED: Scaled backward pass
            scaler.scale(loss).backward()

            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                # OPTIMIZED: Gradient clipping and step
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                # OPTIMIZED: Faster zero_grad
                optimizer.zero_grad(set_to_none=True)

                scheduler.step()
                global_step += 1

                # OPTIMIZED: Periodic memory cleanup
                if global_step % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            epoch_loss += loss.item() * GRAD_ACCUM_STEPS
            steps += 1

            if batch_idx % 10 == 0:
                postfix = {
                    'loss': f'{loss.item() * GRAD_ACCUM_STEPS:.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
                    'step': global_step
                }
                if 'aux_loss' in outputs:
                    postfix['aux'] = f'{outputs["aux_loss"].item():.4f}'
                loop.set_postfix(postfix)

        avg_loss = epoch_loss / steps if steps > 0 else float('inf')
        avg_aux_loss = epoch_aux_loss / steps if steps > 0 else 0.0

        print(f"\n📊 Epoch {epoch + 1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f}", end="")
        if avg_aux_loss > 0:
            print(f" | Aux Loss: {avg_aux_loss:.4f}", end="")
        print()

        if epoch % 2 == 0 or epoch == NUM_EPOCHS - 1:
            test_generation_progressive(model, tokenizer, epoch + 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            suffix = "1.2B" if DEV_MODE else "8.5B"
            checkpoint_path = f"nuvion_pro_{suffix}_best.pth"

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': {
                    'vocab_size': tokenizer.get_vocab_size(),
                    'd_model': D_MODEL,
                    'nhead': NHEAD,
                    'num_layers': NUM_LAYERS,
                    'max_seq_len': MAX_SEQ_LEN,
                    'model_size': MODEL_SIZE,
                }
            }, checkpoint_path)
            print(f"💾 Best model saved: {checkpoint_path} (loss: {best_loss:.4f})")

    suffix = "1.2B" if DEV_MODE else "8.5B"
    final_path = f"nuvion_pro_{suffix}_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"\n🎉 Nuvion Pro training complete! Best loss: {best_loss:.4f}")
    print(f"📂 Saved: {final_path}")


# ========================
# MAIN EXECUTION
# ========================
if __name__ == "__main__":
    print("🚀 Nuvion Pro AI Training System")
    print("📚 English Language Only")
    print("=" * 50)

    train_nuvion()
