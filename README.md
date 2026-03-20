<div align="center">

<br/>

# 📚 Fine-Tuning Mistral-7B
## on *Frankenstein* 🧪⚡

### Large Language Model · LoRA / QLoRA · Hugging Face · PEFT

<br/>

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-FFD21E?style=for-the-badge)](https://huggingface.co/mistralai/Mistral-7B-v0.1)
[![Mistral](https://img.shields.io/badge/Mistral--7B-v0.1-FF6B35?style=for-the-badge)](https://huggingface.co/mistralai/Mistral-7B-v0.1)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-8A2BE2?style=for-the-badge)](https://github.com/huggingface/peft)
[![CUDA](https://img.shields.io/badge/CUDA-GPU_Required-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

<br/>

> *"Nothing is so painful to the human mind as a great and sudden change."*
> — Mary Shelley, *Frankenstein*

<br/>

---

</div>

## 🎯 Goal

Fine-tune [**Mistral-7B-v0.1**](https://huggingface.co/mistralai/Mistral-7B-v0.1) on the full text of Mary Shelley's *Frankenstein* — producing a model capable of generating **coherent, contextually rich, and stylistically faithful** responses inspired by the novel.

This project serves as an end-to-end demonstration of **efficient LLM fine-tuning** on domain-specific literary data using modern techniques like **LoRA / QLoRA** and the **PEFT** library.

<br/>

---

## 🚀 Project Overview

| | Detail |
|---|---|
| 🤖 **Base Model** | `mistralai/Mistral-7B-v0.1` |
| 📖 **Training Data** | *Frankenstein* by Mary Shelley (full novel text) |
| ⚙️ **Fine-Tuning Method** | LoRA / QLoRA (Parameter-Efficient Fine-Tuning) |
| 🎯 **Objective** | Generate responses in the style and context of the novel |
| 🧠 **Framework** | Hugging Face Transformers + PEFT + TRL |

<br/>

---

## 🗂️ Pipeline Architecture

```
📖 Raw Text (Frankenstein)
        │
        ▼
✂️  Text Preprocessing & Chunking
        │
        ▼
🗃️  Dataset Formatting (prompt–response pairs)
        │
        ▼
🔢 Tokenization  (Mistral-7B Tokenizer)
        │
        ▼
⚙️  LoRA / QLoRA Configuration
   (rank, alpha, target modules)
        │
        ▼
🔥 Fine-Tuning with SFTTrainer / Trainer
        │
        ▼
💾 Save Adapter Weights  (PEFT adapter)
        │
        ▼
🧪 Inference & Text Generation
```

<br/>

---

## ✨ Features

| | Feature | Description |
|---|---|---|
| 🤗 | **Mistral-7B Base Model** | State-of-the-art 7B parameter language model as the foundation |
| 📚 | **Literary Fine-Tuning** | Domain adaptation on *Frankenstein* for thematic text generation |
| ⚡ | **LoRA / QLoRA** | Memory-efficient fine-tuning without updating all model weights |
| 🗜️ | **4-bit Quantization** | `bitsandbytes` integration for reduced GPU memory footprint |
| 🧑‍🏫 | **SFTTrainer** | Supervised fine-tuning with `trl` library for clean training loops |
| 💬 | **Custom Prompting** | Structured prompt templates tailored to the novel's themes |
| 💾 | **Adapter Saving** | Save and reload lightweight LoRA adapter weights independently |
| 🧪 | **Inference Pipeline** | Generate novel-inspired responses from any input prompt |

<br/>

---

## 📦 Requirements

**Python 3.8+** and a **CUDA-enabled GPU** are recommended for fine-tuning.

Install all dependencies:

```bash
pip install -r requirements.txt
```

Core libraries:

| Library | Purpose |
|---|---|
| `transformers` | Mistral-7B model loading & tokenization |
| `peft` | LoRA / QLoRA adapter configuration |
| `trl` | SFTTrainer for supervised fine-tuning |
| `bitsandbytes` | 4-bit / 8-bit quantization |
| `datasets` | Dataset loading and preprocessing |
| `accelerate` | Distributed & mixed-precision training |
| `torch` | PyTorch backend |

<br/>

---

## 🚀 Getting Started

**1. Clone the Repository**

```bash
git clone https://github.com/USERNAME/fine-tuning-code.git
cd fine-tuning-code
```

**2. Install Dependencies**

```bash
pip install -r requirements.txt
```

**3. (Optional) Hugging Face Login**

Required to access gated models like Mistral-7B:

```bash
huggingface-cli login
```

**4. Run Fine-Tuning**

```bash
python train.py
```

**5. Run Inference**

```bash
python inference.py --prompt "What does the creature say about solitude?"
```

<br/>

---

## ⚙️ LoRA Configuration

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,                        # LoRA rank
    lora_alpha=32,               # Scaling factor
    target_modules=["q_proj", "v_proj"],   # Attention layers to adapt
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

> 💡 Adjust `r` and `lora_alpha` to trade off between model expressiveness and memory usage.

<br/>

---

## 📁 Project Structure

```
fine-tuning-code/
│
├── 📄 train.py                  # Fine-tuning script
├── 📄 inference.py              # Text generation script
├── 📄 requirements.txt          # Project dependencies
├── 📄 README.md                 # Project documentation
│
├── 📂 data/
│   └── frankenstein.txt         # Raw novel text
│
├── 📂 dataset/
│   └── prepare_dataset.py       # Chunking & formatting logic
│
├── 📂 configs/
│   └── training_config.yaml     # Hyperparameters & training settings
│
└── 📂 outputs/
    ├── adapter_model/           # Saved LoRA adapter weights
    └── logs/                    # Training logs & loss curves
```

<br/>

---

## 💬 Example Outputs

**Prompt:**
```
What does the creature feel about being abandoned by his creator?
```

**Generated Response:**
```
The creature, wretched and forsaken, wandered through the desolate wilderness,
his heart heavy with the unbearable weight of solitude. He had sought only
companionship and understanding, yet his creator — the one being who owed him
existence — had fled in horror at the sight of his own creation...
```

<br/>

---

## 📊 Training Details

| Parameter | Value |
|---|---|
| Base Model | `mistralai/Mistral-7B-v0.1` |
| Quantization | 4-bit (NF4) |
| LoRA Rank (`r`) | 16 |
| LoRA Alpha | 32 |
| Learning Rate | `2e-4` |
| Batch Size | 4 |
| Epochs | 3 |
| Max Sequence Length | 512 tokens |
| Optimizer | `paged_adamw_32bit` |

<br/>

---

## 🤝 Contributing

Contributions are welcome!

1. **Fork** the repository
2. Create a branch: `git checkout -b feature/your-improvement`
3. Commit: `git commit -m "feat: add your change"`
4. Push and open a **Pull Request** 🚀

**Ideas to explore:**
- [ ] Fine-tune on other classic novels (*Dracula*, *Jekyll & Hyde*)
- [ ] Experiment with different LoRA ranks and target modules
- [ ] Build a Gradio / Streamlit demo interface
- [ ] Merge adapter weights with base model for full deployment
- [ ] Evaluate outputs with BLEU or perplexity metrics

<br/>
## 👤 Author

Ahmad Albasha

[![GitHub](https://img.shields.io/badge/GitHub-ahmad--albasha-181717?style=flat-square&logo=github)](https://github.com/ahmad-albasha)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/ahmad-a-9a0373123)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?style=flat-square&logo=gmail)](mailto:ahmad-albasha09@hotmail.com)

---

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

The novel *Frankenstein* by Mary Shelley (1818) is in the **public domain**.

<br/>

---

<div align="center">

**Built with 🧠 curiosity, 📖 literature, and ⚡ GPU power**

⭐ **If this project sparked your imagination, consider giving it a star!** ⭐

</div>
