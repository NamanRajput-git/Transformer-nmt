# 🔤 Transformer NMT — English → Hindi Neural Machine Translation

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)
![SentencePiece](https://img.shields.io/badge/SentencePiece-BPE-green?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-IITB%20EN--HI-orange?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Kaggle%20GPU-20BEFF?style=flat-square&logo=kaggle)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

**A from-scratch Transformer implementation for English-to-Hindi neural machine translation, trained on the IITB English-Hindi parallel corpus.**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture Diagram](#-architecture-diagram)
- [Training Pipeline Flowchart](#-training-pipeline-flowchart)
- [Project Structure](#-project-structure)
- [Classes & Modules](#-classes--modules)
- [Dataset](#-dataset)
- [Hyperparameters](#-hyperparameters)
- [Training Results](#-training-results)
- [Quick Start](#-quick-start)
- [Dependencies](#-dependencies)

---

## 🌐 Overview

This project implements the **Transformer** architecture (Vaswani et al., 2017 — *"Attention Is All You Need"*) from scratch using PyTorch for English-to-Hindi translation. Key features:

| Feature | Detail |
|---|---|
| **Architecture** | Encoder-Decoder Transformer |
| **Tokenizer** | SentencePiece BPE, 28 000 shared vocabulary |
| **Dataset** | IIT Bombay English-Hindi corpus (~1.6 M pairs) |
| **Training** | Mixed-precision (FP16), Noam LR schedule, gradient accumulation |
| **Inference** | Beam search (beam = 4) with length normalisation |
| **Evaluation** | Cross-entropy loss + SacreBLEU |
| **Platform** | Kaggle (GPU T4 / P100) |

---

## 🏗 Architecture Diagram

```mermaid
graph TD
    subgraph INPUT["Input Processing"]
        EN["English Sentence\n(raw text)"]
        SP_ENC["SentencePiece BPE\nTokeniser (28k vocab)"]
        EN --> SP_ENC
    end

    subgraph ENCODER["Transformer Encoder  ×6"]
        EMB_S["Source Embedding\n(28000 × 256)"]
        PE_S["Positional Encoding\n(sinusoidal)"]
        ENC_L["Encoder Layer\n├─ Multi-Head Self-Attention (4 heads)\n├─ Add & LayerNorm\n├─ Feed-Forward (256→1024→256)\n└─ Add & LayerNorm"]
        EMB_S --> PE_S --> ENC_L
    end

    subgraph DECODER["Transformer Decoder  ×6"]
        EMB_T["Target Embedding\n(28000 × 256)\n[weight-tied with output]"]
        PE_T["Positional Encoding\n(sinusoidal)"]
        DEC_L["Decoder Layer\n├─ Masked Multi-Head Self-Attention\n├─ Add & LayerNorm\n├─ Cross-Attention (enc memory)\n├─ Add & LayerNorm\n├─ Feed-Forward (256→1024→256)\n└─ Add & LayerNorm"]
        EMB_T --> PE_T --> DEC_L
    end

    subgraph OUTPUT["Output"]
        FC["Linear Projection\n(256 → 28000)\n[weights tied with embedding]"]
        SOFTMAX["Softmax / Log-Softmax"]
        HI["Hindi Token"]
        FC --> SOFTMAX --> HI
    end

    SP_ENC --> EMB_S
    ENC_L -->|"Memory (K, V)"| DEC_L
    DEC_L --> FC

    style INPUT   fill:#1e3a5f,color:#fff
    style ENCODER fill:#1a4731,color:#fff
    style DECODER fill:#4a1a3a,color:#fff
    style OUTPUT  fill:#4a3a1a,color:#fff
```

### Model Dimensions

```
Input Tokens  ──► Embedding(28000, 256) ──► ×√256 ──► PositionalEncoding
                                                            │
                                              ┌─────────────▼─────────────┐
                                              │  Encoder  (×6 layers)      │
                                              │  d_model=256, nhead=4      │
                                              │  dim_ff=1024, dropout=0.1  │
                                              └─────────────┬─────────────┘
                                                            │ memory
                                              ┌─────────────▼─────────────┐
                                              │  Decoder  (×6 layers)      │
                                              │  + causal mask             │
                                              └─────────────┬─────────────┘
                                                            │
                                                      Linear(256→28000)
                                                      [weight-tied]
                                                            │
                                                      Output logits
```

---

## 🔄 Training Pipeline Flowchart

```mermaid
flowchart TD
    A([Start]) --> B[Load IITB EN-HI Dataset\ncfilt/iitb-english-hindi]
    B --> C[Build DataFrames\ntrain / val / test]
    C --> D[Load SentencePiece BPE Model\nbpe.model — 28k vocab]
    D --> E[Derive BOS · EOS · PAD token IDs\nfrom tokeniser]
    E --> F[Encode & Filter\nBOS + tokens + EOS\nDrop duplicates\nDrop len > 100]
    F --> G[CustomDataset & DataLoader\nbatch=16, pad_sequence collate]

    G --> H[Instantiate Custom_Transformer\nd_model=256, nhead=4, layers=6]
    H --> I[Adam optimiser lr=1.0\nNoam LR schedule warmup=4000\nGradScaler for AMP]

    I --> J{Epoch loop\n1..10}

    J --> K[Mini-batch forward pass\nautocast FP16]
    K --> L[teacher-forced decode\ntarget_in = target,:-1\ntarget_out = target,1:]
    L --> M[CrossEntropy loss\nlabel_smoothing=0.1\ndivide by GRAD_ACCUM=4]
    M --> N[scaler.scale loss .backward]

    N --> O{Accum step?\ni mod 4 == 0}
    O -- No --> K
    O -- Yes --> P[Unscale gradients\nClip grad_norm ≤ 1.0]
    P --> Q{NaN / Inf\ngrad norm?}
    Q -- Yes --> R[Skip step\nzero_grad] --> K
    Q -- No --> S[scaler.step optimizer\nscheduler.step\nzero_grad]

    S --> T{Epoch done?}
    T -- No --> K
    T -- Yes --> U[Compute val loss\nevaluate function]
    U --> V[Save checkpoint_end_epochN.pt]
    V --> W{val_loss <\nbest_val_loss?}
    W -- Yes --> X[Save best_transformer_model.pt]
    W -- No --> Y{More epochs?}
    X --> Y
    Y -- Yes --> J
    Y -- No --> Z[Load best model\nBeam Search inference\nSacreBLEU evaluation]
    Z --> END([Done])

    style A    fill:#2d6a4f,color:#fff
    style END  fill:#2d6a4f,color:#fff
    style Q    fill:#9b2226,color:#fff
    style W    fill:#1d3557,color:#fff
```

---

## 📁 Project Structure

```
TRANSFORMER/
│
├── transformer_en_hi_merged.ipynb   # Main training + inference notebook
├── dataset.ipynb                    # Dataset exploration & preprocessing
│
├── bpe.model                        # SentencePiece BPE model (28k vocab)
├── bpe.vocab                        # BPE vocabulary file
│
├── best_transformer_model.pt        # Best checkpoint (saved by val loss)
├── training_info.json               # Training metadata (loss, hyperparams)
├── lr_schedule.png                  # Noam LR schedule visualisation
│
├── dataset.csv                      # Full parallel corpus (~452 MB)
│
└── data/
    ├── train_en.txt                 # Raw English training text
    └── train_hi.txt                 # Raw Hindi training text
```

---

## 🧩 Classes & Modules

### `PositionalEncoding` — `nn.Module`

Implements **sinusoidal positional encoding** as described in Vaswani et al. (2017).

```
PositionalEncoding
├── __init__(d_model, max_len=512, dropout=0.1)
│       Precomputes the PE buffer of shape (1, max_len, d_model).
│       Even dims → sin(pos / 10000^(2i/d_model))
│       Odd  dims → cos(pos / 10000^(2i/d_model))
│       Registers as a non-learnable buffer.
│
└── forward(x: Tensor[B, T, D]) → Tensor[B, T, D]
        x = x + pe[:, :T]       # broadcast-add positional signal
        return Dropout(x)
```

| Param | Type | Default | Purpose |
|---|---|---|---|
| `d_model` | int | — | Model embedding dimension |
| `max_len` | int | 512 | Maximum sequence length supported |
| `dropout` | float | 0.1 | Applied after adding positional signal |

---

### `Custom_Transformer` — `nn.Module`

End-to-end encoder-decoder Transformer for seq2seq translation.

```
Custom_Transformer
├── __init__(vocab_size, d_model=256, nhead=4, num_layers=6,
│            dim_ff=1024, dropout=0.1)
│       ├─ nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
│       ├─ PositionalEncoding(d_model, dropout)
│       ├─ nn.Transformer(d_model, nhead, num_encoder_layers=num_layers,
│       │                  num_decoder_layers=num_layers,
│       │                  dim_feedforward=dim_ff, batch_first=True)
│       └─ nn.Linear(d_model, vocab_size)   ← weight-tied with embedding
│
├── _init_weights()
│       Xavier uniform initialisation for all 2-D parameters.
│
└── forward(source, target) → logits[B, T_tgt, vocab_size]
        ├─ src_key_padding_mask  = (source == PAD_ID)
        ├─ tgt_key_padding_mask  = (target == PAD_ID)
        ├─ tgt_mask              = causal square subsequent mask
        ├─ source = PE(Emb(source) × √d_model)
        ├─ target = PE(Emb(target) × √d_model)
        └─ return Linear(Transformer(source, target, masks...))
```

| Param | Default | Meaning |
|---|---|---|
| `vocab_size` | 28 000 | Shared source & target vocabulary |
| `d_model` | 256 | Embedding / hidden dimension |
| `nhead` | 4 | Attention heads per layer |
| `num_layers` | 6 | Encoder AND decoder depth |
| `dim_ff` | 1 024 | Feed-forward hidden width |
| `dropout` | 0.1 | Applied throughout |

> **Weight Tying**: `Linear.weight = Embedding.weight` — shares parameters between input embedding and output projection, improving generalisation and reducing total parameters.

---

### `CustomDataset` — `torch.utils.data.Dataset`

Wraps the encoded token-ID lists into a PyTorch Dataset.

```
CustomDataset
├── __init__(df: DataFrame)
│       self.source = df['en_ids'].tolist()   # list of int lists
│       self.target = df['hi_ids'].tolist()
│
├── __len__() → int
│
└── __getitem__(index) → (LongTensor[src_len], LongTensor[tgt_len])
```

---

### `collate_fn` — Batch Collation Function

Dynamic padding of variable-length sequences within a mini-batch.

```python
def collate_fn(batch):
    source_batch, target_batch = zip(*batch)
    source_batch = rnn_utils.pad_sequence(source_batch, batch_first=True, padding_value=PAD_ID)
    target_batch = rnn_utils.pad_sequence(target_batch, batch_first=True, padding_value=PAD_ID)
    return source_batch, target_batch
```

Pads each batch to the **maximum length in that batch** (not a global max), keeping compute efficient.

---

### `evaluate` — Validation Loss

```python
def evaluate(model, val_dataLoader, loss_fun, device) → float
```

Runs the model in `eval()` mode with `torch.no_grad()` and `autocast()` over the full validation set; returns mean cross-entropy loss.

---

### `translate_sentence` — Beam Search Decoder

```
translate_sentence(sentence, model, sp, device,
                   max_len=80, beam_size=4) → str

Algorithm:
  1. Tokenise: [BOS] + sp.encode(sentence) + [EOS]
  2. Initialise beams: [(score=0.0, tokens=[BOS_ID])]
  3. For each decoding step (up to max_len):
       For each live beam:
         • Run model forward pass on current tokens
         • Compute log_softmax over last position
         • Expand to top-K candidates (log-score + lp)
       Length-normalise all candidates; keep top beam_size
       If all beams ended with EOS → stop early
  4. Select beam with highest length-normalised score
  5. Decode inner tokens (strip BOS/EOS) → sp.decode()
```

---

### `compute_bleu` — SacreBLEU Estimation

```python
def compute_bleu(model, data_df, sp, device,
                 n_samples=500, max_len=80, beam_size=4) → float
```

Translates a random subset of `n_samples` sentence pairs and computes **corpus BLEU** using `sacrebleu.corpus_bleu`.

---

### `noam_lambda` — Noam Learning Rate Schedule

```python
def noam_lambda(step):
    step = max(step, 1)
    return (D_MODEL ** -0.5) * min(step ** -0.5, step * WARMUP_STEPS ** -1.5)
```

Implements the schedule from *Attention Is All You Need*:

```
lr(step) = d_model^(-0.5) × min(step^(-0.5), step × warmup_steps^(-1.5))
```

- **Warmup phase** (step < warmup_steps): LR increases linearly
- **Decay phase** (step ≥ warmup_steps): LR decays as 1/√step

Peak LR reached at `step = warmup_steps = 4000`.

---

## 📊 Dataset

| Split | Pairs | Source |
|---|---|---|
| Train | ~1.58 M | IITB English-Hindi Corpus |
| Validation | ~521 | IITB |
| Test | ~2 507 | IITB |

**Loaded via**: `datasets.load_dataset("cfilt/iitb-english-hindi")`

**Preprocessing pipeline**:
1. Extract `translation["en"]` and `translation["hi"]` fields
2. Encode with SentencePiece BPE: `[BOS] + tokens + [EOS]`
3. Deduplicate on English side
4. Filter: drop pairs where either side > 100 tokens

---

## ⚙️ Hyperparameters

| Hyperparameter | Value | Notes |
|---|---|---|
| `vocab_size` | 28 000 | Shared BPE vocab |
| `d_model` | 256 | Embedding dimension |
| `nhead` | 4 | Attention heads |
| `num_layers` | 6 | Encoder and decoder depth |
| `dim_ff` | 1 024 | FFN hidden size |
| `dropout` | 0.1 | All sub-layers |
| `MAX_LEN` | 100 | Token sequence length filter |
| `BATCH_SIZE` | 16 | Mini-batch size |
| `GRAD_ACCUM_STEPS` | 4 | Effective batch = 64 |
| `WARMUP_STEPS` | 4 000 | Noam LR schedule |
| `NUM_EPOCHS` | 10 | Training epochs |
| `label_smoothing` | 0.1 | Cross-entropy regularisation |
| `grad_clip` | 1.0 | Max gradient norm |
| `beam_size` | 4 | Beam search width at inference |
| **Trainable params** | **~17 M** | |

---

## 📈 Training Results

| Metric | Value |
|---|---|
| Best validation loss | **4.88** |
| Perplexity | **~132** |
| Epochs trained | 10 |
| LR schedule | Noam (warmup=4000) |

### Noam Learning Rate Schedule

![LR Schedule](lr_schedule.png)

> **Note**: The current val loss of 4.88 indicates the model has learned meaningful structure but has room to improve. Scaling to `d_model=512` and using the full 1.6 M training pairs is expected to push val loss below 2.5 (perplexity ~12) with proportionally better BLEU scores.

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch sentencepiece datasets sacrebleu matplotlib pandas
```

### Running on Kaggle

1. Upload `bpe.model` and `bpe.vocab` as a Kaggle dataset input.
2. Open `transformer_en_hi_merged.ipynb` in a Kaggle notebook with GPU enabled.
3. Run all cells sequentially.

### Local Inference (CPU/GPU)

```python
import torch
import sentencepiece as spm
import torch.nn.functional as F

# Load tokeniser
sp = spm.SentencePieceProcessor()
sp.load("bpe.model")

BOS_ID = sp.bos_id()   # 2
EOS_ID = sp.eos_id()   # 3
PAD_ID = max(sp.pad_id(), 0)

# Reconstruct model (must match training config)
model = Custom_Transformer(vocab_size=28000, d_model=256,
                           nhead=4, num_layers=6, dim_ff=1024)
model.load_state_dict(torch.load("best_transformer_model.pt", map_location="cpu"))
model.eval()

# Translate
result = translate_sentence("I love machine learning.", model, sp, device="cpu")
print(result)
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `torch` | Model definition, training, autocast |
| `torch.nn` | `Transformer`, `Embedding`, `Linear`, `Dropout` |
| `torch.cuda.amp` | `autocast`, `GradScaler` — FP16 mixed precision |
| `torch.utils.data` | `Dataset`, `DataLoader` |
| `torch.nn.utils.rnn` | `pad_sequence` — dynamic batching |
| `sentencepiece` | BPE tokenisation / detokenisation |
| `datasets` | HuggingFace `load_dataset` for IITB corpus |
| `sacrebleu` | Standard BLEU evaluation |
| `pandas` | DataFrame manipulation |
| `matplotlib` | LR schedule visualisation |
| `json`, `os`, `glob`, `math`, `shutil` | Standard library utilities |

---

## 📖 References

1. Vaswani, A. et al. (2017). **Attention Is All You Need**. *NeurIPS 2017*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
2. Kunchukuttan, A. et al. (2018). **The IIT Bombay English-Hindi Parallel Corpus**. [arXiv:1710.02855](https://arxiv.org/abs/1710.02855)
3. Kudo, T. & Richardson, J. (2018). **SentencePiece: A simple and language independent subword tokenizer**. [arXiv:1808.06226](https://arxiv.org/abs/1808.06226)
4. Post, M. (2018). **A Call for Clarity in Reporting BLEU Scores** (SacreBLEU). [arXiv:1804.08771](https://arxiv.org/abs/1804.08771)

---

## 📄 License

This project is licensed under the **MIT License**.

---

<div align="center">
Made with ❤️ using PyTorch &nbsp;|&nbsp; IIT Bombay EN-HI Corpus &nbsp;|&nbsp; Kaggle GPU
</div>
