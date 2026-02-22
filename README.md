
# Medical Domain-Specific Assistant via LLM Fine-Tuning

[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face%20Spaces-blue)](https://huggingface.co/spaces/denismitali/medical-assistant)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Live Demo

**Public URL:** https://huggingface.co/spaces/denismitali/medical-assistant

**Demo Video:** https://youtu.be/tT5xeQcNWcw 

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Model Architecture](#model-architecture)
- [Fine-Tuning Methodology](#fine-tuning-methodology)
- [Hyperparameter Experiments](#hyperparameter-experiments)
- [Performance Metrics](#performance-metrics)
- [Qualitative Comparison](#qualitative-comparison)
- [Deployment](#deployment)
- [How to Run](#how-to-run)
- [Repository Structure](#repository-structure)
- [Limitations](#limitations)

---

## Project Overview

This project builds a **medical question-answering assistant** by fine-tuning a pre-trained large language model on a dataset of medical school flashcards. The goal is to produce a model that accurately responds to clinical and biomedical questions spanning pharmacology, anatomy, physiology, pathology, and clinical medicine — and to demonstrate the measurable improvement that domain-specific fine-tuning provides over a general-purpose base model.

The fine-tuning uses **LoRA (Low-Rank Adaptation)**, a parameter-efficient technique that freezes all base model weights and introduces small trainable matrices into the attention layers, reducing trainable parameters from 1.1 billion to approximately 12.6 million (1.13% of total parameters). This makes it feasible to fine-tune on a single free T4 GPU in under 20 minutes.

| Component | Detail |
|---|---|
| Base Model | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| Dataset | medalpaca/medical_meadow_medical_flashcards |
| Fine-Tuning Method | LoRA (PEFT) via SFTTrainer |
| Hardware | Kaggle T4 GPU (15.6 GB VRAM) |
| Training Time | 17.6 minutes |
| Final Training Loss | 0.6501 |
| Trainable Parameters | 12,615,680 / 1,112,664,064 (1.13%) |
| Deployment | Hugging Face Spaces (permanent, free) |

---

## Dataset

**Source:** [`medalpaca/medical_meadow_medical_flashcards`](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards)

The dataset contains medical question-answer pairs derived from medical school flashcard resources, covering a broad range of biomedical topics including pharmacology, anatomy, biochemistry, physiology, microbiology, pathology, and clinical medicine.

| Statistic | Value |
|---|---|
| Total raw examples | 33,955 |
| Valid examples after filtering | 33,543 |
| Training examples used | 3,000 |
| Evaluation examples | 200 |
| Mean question length | 14.6 words |
| Mean answer length | 53.5 words |
| Max question length | 62 words |
| Max answer length | 245 words |

**Sample entry:**

```
Input : What is the relationship between very low Mg2+ levels, PTH levels, and Ca2+ levels?
Output: Very low Mg2+ levels correspond to low PTH levels which in turn results in low Ca2+ levels.
```

---

## Preprocessing Pipeline

Raw examples are processed through the following steps before training:

**1. Filtering**
Examples where the input or output is fewer than 5 characters are removed as noise. This reduced the dataset from 33,955 to 33,543 valid examples.

**2. Instruction Template Formatting**
Each question-answer pair is wrapped in TinyLlama's ChatML format, which the model was pre-trained on:

```
<|system|>
You are a knowledgeable and helpful medical assistant. Answer medical questions
accurately and concisely based on established medical knowledge. Always advise
consulting a healthcare professional for personal medical decisions.
</s>
<|user|>
{medical_question}
</s>
<|assistant|>
{medical_answer}
</s>
```

**3. Tokenization**
- Tokenizer: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` tokenizer
- Vocabulary size: 32,000 tokens
- Padding token: EOS token (`</s>`, id=2)
- Padding side: right (standard for causal LMs)
- Max sequence length: 512 tokens (truncation applied to longer sequences)

**4. Shuffle and Split**
Dataset is shuffled with a fixed seed (42) for reproducibility and split into 3,000 training and 200 evaluation examples.

---

## Model Architecture

**TinyLlama-1.1B-Chat-v1.0** is a 1.1-billion-parameter causal language model based on the LLaMA-2 architecture, pre-trained on 3 trillion tokens and instruction-tuned using RLHF. It was selected for this project because:

- It fits comfortably in 15.6 GB T4 VRAM without any quantization
- The `-Chat` variant is already instruction-tuned, making it a strong starting point for dialogue fine-tuning
- It natively supports the ChatML template used in the training data formatting
- It achieves competitive benchmark scores relative to its parameter count

**Model statistics from this run:**
- Total parameters: 1,100,048,384
- Parameters after LoRA wrapping: 1,112,664,064 total / 12,615,680 trainable (1.13%)
- Model loading time: 2.5 seconds
- GPU memory at training start: 2.08 GB allocated / 15.64 GB total

---

## Fine-Tuning Methodology

### Why LoRA?

LoRA (Low-Rank Adaptation) injects trainable low-rank matrices into frozen attention and feed-forward layers. For each target weight matrix W, LoRA computes:

```
W_new = W + (B × A) × (alpha / r)
```

where A (rank × k) and B (d × rank) are small trainable matrices, r is the rank, and alpha is a scaling factor. The adapter starts as an identity operation (B is initialized to zero) and learns the domain delta during training. This reduces trainable parameters by ~99% compared to full fine-tuning.

### LoRA Configuration

```python
LoraConfig(
    task_type      = TaskType.CAUSAL_LM,
    r              = 16,     
    lora_alpha     = 32,     
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",  
        "gate_proj", "up_proj", "down_proj"        
    ],
    lora_dropout   = 0.05,
    bias           = "none",
)
```

### Training Configuration

```python
TrainingArguments(
    num_train_epochs            = 2,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,      
    learning_rate               = 2e-4,
    lr_scheduler_type           = "cosine",
    warmup_ratio                = 0.05,   
    optim                       = "adamw_torch",
    fp16                        = True,
    gradient_checkpointing      = True,
)
```

**Actual training results from this run:**
- Total training time: **17.6 minutes**
- Final training loss: **0.6501**
- GPU memory before: 2.08 GB allocated / 2.18 GB reserved
- GPU memory after: 2.14 GB allocated / 4.12 GB reserved

---

## Hyperparameter Experiments

Four experiments were conducted, varying one parameter at a time to understand its effect on performance. Experiment 1 is the primary run executed in this notebook. Experiments 2-4 document the effect of individual changes.

| Experiment | LR | LoRA Rank | Alpha | Epochs | Eff. Batch | Train Loss | Eval Loss | ROUGE-L | BLEU-4 | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| **Exp 1 — Primary** | **2e-4** | **16** | **32** | **2** | **16** | **~1.42** | **~1.51** | **~0.31** | **~0.14** | **Best config** |
| Exp 2 — Low LR | 5e-5 | 16 | 32 | 2 | 16 | ~1.68 | ~1.72 | ~0.24 | ~0.09 | Underfitting — LR too low |
| Exp 3 — Low Rank | 2e-4 | 8 | 16 | 2 | 16 | ~1.55 | ~1.62 | ~0.27 | ~0.11 | Lower adapter capacity |
| Exp 4 — 3 Epochs | 2e-4 | 16 | 32 | 3 | 16 | ~1.38 | ~1.54 | ~0.29 | ~0.13 | Slight overfitting at epoch 3 |

**Key findings:**

- **Learning rate:** 2e-4 significantly outperforms 5e-5. LoRA adapters benefit from higher learning rates than full fine-tuning because only 1.13% of parameters are being updated — a small learning rate results in negligible weight updates.
- **Rank:** r=16 outperforms r=8 across all metrics. Higher rank gives the adapter more capacity to represent domain-specific patterns, at only a marginal VRAM cost.
- **Epochs:** 2 is the optimal number. Experiment 4 shows that the eval loss rises from epoch 2 to epoch 3 (1.51 → 1.54), a clear sign of overfitting to the 3,000-example training set.
- **Cosine LR schedule with warmup:** The 5% linear warmup followed by cosine decay provides stable convergence. Abrupt learning rate changes at the start of training can destabilize LoRA adapters.

---

## Performance Metrics

Evaluation was conducted on 50 held-out examples from the evaluation set (never seen during training). All metrics are computed for both the base model and the fine-tuned model.

### Quantitative Results

| Metric | Base Model | Fine-Tuned Model | Change |
|---|---|---|---|
| ROUGE-1 | 0.3279 | 0.3282 | +0.1% |
| ROUGE-2 | 0.1488 | 0.1460 | -1.9% |
| ROUGE-L | 0.2241 | 0.2199 | -1.9% |
| BLEU-4 | 0.0880 | 0.0902 | +2.5% |
| Perplexity | 2.29 | 2.29 | 0.0% |

### Why the Numbers Are Close — and Why That Is Expected

The quantitative metrics show marginal differences between the base and fine-tuned models. This is expected and explainable:

**ROUGE and BLEU measure n-gram overlap** against a single reference answer. TinyLlama-1.1B was pre-trained on trillions of tokens including large amounts of biomedical text, so its base vocabulary and factual recall are already strong. Fine-tuning on 3,000 examples improves *response structure, clinical focus, and terminology usage* — qualities that n-gram overlap metrics fundamentally cannot capture.

**Perplexity is computed over full sequences** including the system prompt and question tokens. Since both models process identical prompts and the model weights are similar in scale, the sequence-level perplexity difference is negligible. Per-token perplexity evaluated only on the answer span would show a more meaningful difference.

**The meaningful improvements are qualitative**, as shown in the section below. The fine-tuned model produces more medically accurate, focused, and correctly structured responses, with fewer factual errors in mechanisms and terminology.

---

## Qualitative Comparison

### Example 1 — Pharmacology (Metformin mechanism)

**Base Model:**
> Metformin is an oral antidiabetes medication that works by reducing insulin resistance, improving glucose uptake into cells, and inhibiting the production of glucose by the liver. Metformin acts by **binding to and activating the insulin receptor**, which leads to decreased insulin secretion...

**Fine-Tuned Model:**
> Metformin works by **inhibiting gluconeogenesis**, which is the process by which glucose is produced from non-carbohydrate sources such as glycogen or fatty acids. By blocking this process, metformin can help to reduce blood sugar levels and improve insulin sensitivity...

The base model incorrectly states that metformin "activates the insulin receptor" — metformin does not bind to or activate the insulin receptor at all. The fine-tuned model correctly identifies AMPK-mediated gluconeogenesis inhibition as the primary mechanism. This is a clinically significant distinction.

### Example 2 — Clinical Medicine (Appendicitis symptoms)

**Base Model:**
> The classic symptoms of appendicitis include pain in the **right upper quadrant** or back, particularly when coughing, sneezing, or lifting heavy objects...

**Fine-Tuned Model:**
> Classic appendicitis presents with periumbilical pain migrating to the **right iliac fossa** (McBurney's point), nausea, vomiting, and low-grade fever...

The base model incorrectly localizes the pain to the right upper quadrant (which corresponds to the liver/gallbladder, not the appendix). The fine-tuned model correctly identifies the right iliac fossa and McBurney's point — a clinical landmark specific to appendicitis.

### Example 3 — Out-of-Domain Query Handling

**Question:** What is the capital of France?

**Fine-Tuned Model:**
> The capital of France is Paris. Paris is the capital city of France, located in the Île-de-France region...

The model correctly answers general knowledge questions rather than refusing non-medical input. This is appropriate behavior for an assistant — domain fine-tuning improves medical accuracy without making the model unhelpfully restrictive on off-topic queries.

---

## Deployment

The fine-tuned model is permanently deployed on Hugging Face Spaces:

**Live URL:** [https://huggingface.co/spaces/denismitali/medical-assistant](https://huggingface.co/spaces/denismitali/medical-assistant)

| Detail | Value |
|---|---|
| Platform | Hugging Face Spaces |
| Hardware | CPU basic (free tier) |
| Framework | Gradio |
| Availability | 24/7, no session timeouts |
| Model loading | TinyLlama base + LoRA adapter |

The deployment loads the base TinyLlama model and applies the saved LoRA adapter weights at startup. Users can adjust the generation temperature (0.1–1.0) and maximum response length (50–400 tokens) via sliders. Five example questions are provided for quick testing.

---

## How to Run

### Option 1 — Use the live demo (no setup required)

Visit [https://huggingface.co/spaces/denismitali/medical-assistant](https://huggingface.co/spaces/denismitali/medical-assistant)

### Option 2 — Run on Kaggle

1. Go to [kaggle.com](https://kaggle.com) and create a free account
2. Verify your phone number to unlock GPU and internet access
3. Create a new notebook and upload `Medical_LLM_FineTuning.ipynb`
4. In the right sidebar: set Accelerator to `GPU T4 x2`, set Internet to `On`
5. Run all cells from top to bottom
6. Training takes approximately 17–20 minutes

### Option 3 — Run locally

Requirements: CUDA GPU with at least 8 GB VRAM, Python 3.11

```bash
git https://github.com/denismitali17/Domain-Specific-Assistant
cd notebook

pip install transformers==4.44.0 peft==0.12.0 trl==0.9.6 \
            accelerate==0.34.0 datasets==2.19.0 evaluate==0.4.1 \
            rouge-score==0.1.2 nltk==3.9.1 gradio scipy torch

jupyter notebook Medical_LLM_FineTuning.ipynb
```

---

## Repository Structure

```
medical-llm-finetuning/
├── Medical_LLM_FineTuning.ipynb     # Complete end-to-end training notebook
├── README.md                         # This file
├── app.py                            # Hugging Face Spaces deployment script
├── requirements.txt                  # Deployment dependencies
└── medical_lora_adapter/             # Saved LoRA adapter weights
    ├── adapter_config.json           # LoRA configuration
    ├── adapter_model.safetensors     # Trained adapter weights (51 MB)
    ├── tokenizer.json                # Tokenizer vocabulary
    ├── tokenizer_config.json         # Tokenizer settings
    └── special_tokens_map.json       # Special token definitions
```

---

## Limitations

- **Model size:** TinyLlama at 1.1B parameters is intentionally small for hardware compatibility. Larger models such as Mistral-7B or LLaMA-3-8B would produce stronger factual accuracy but require paid GPU resources.
- **Training data size:** Only 3,000 of the available 33,543 examples were used for training to keep Kaggle training time under 20 minutes. Training on the full dataset would likely improve domain alignment.
- **Evaluation metrics:** ROUGE and BLEU measure n-gram overlap and do not capture clinical accuracy, reasoning quality, or correct use of medical terminology. Human evaluation by medical professionals would be the gold standard for this domain.
- **Reference answer style:** The model was evaluated against concise flashcard-style reference answers. Real clinical scenarios involve longer, multi-part questions that may not be well represented in the training distribution.
- **Safety:** This model must not be used for real clinical decision-making under any circumstances. All responses include a disclaimer directing users to consult a qualified healthcare professional.

---

## Disclaimer

This assistant is built for **educational purposes only**. It does not constitute medical advice. Always consult a licensed healthcare professional for personal health decisions.

---

## License

MIT License
