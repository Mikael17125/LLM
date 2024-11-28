
# Text Generation with GPT2 and Llama Models on TinyStories Dataset

This repository provides text generation models based on both GPT2 and Llama architectures. These models are fine-tuned on the TinyStories dataset, offering both greedy and sampling-based methods for generating creative text outputs.

## Overview

The goal of this project is to fine-tune GPT2 and Llama models on the TinyStories dataset for text generation tasks. The repository includes:
- A `pretrained.py` script for training the GPT2 model on the dataset.
- A `pretrained_llama.py` script for training the Llama model on the dataset.
- An `inference.py` script for generating text using the GPT2 model.
- An `inference_llama.py` script for generating text using the Llama model.
- A custom dataloader (`tiny_story_loader.py`) for handling the TinyStories dataset.
- Utilities for distributed data parallelism during training.

## Prerequisites

To run this code, you need the following dependencies:

- Python 3.7+
- PyTorch 1.10 or later
- `transformers` library by Hugging Face
- `tiktoken` for tokenization
- `datasets` library from Hugging Face

You can install the dependencies using pip:

```bash
pip install torch transformers tiktoken datasets
```

## Project Structure

```
├── checkpoint
│   ├── GPT2_TinyStory.pth
│   └── Llama_TinyStory.pth
├── dataset
│   ├── alpaca_loader.py
│   └── tiny_story_loader.py
├── inference_llama.py          # Generate text using Llama model
├── inference.py                # Generate text using GPT2 model
├── models
│   ├── gpt.py                  # GPT2 model definition
│   ├── llama.py                # Llama model definition
├── pretrained_llama.py         # Script to train Llama model
├── pretrained.py               # Script to train GPT2 model
├── README.md                   # This README file
├── samples
│   ├── alpaca.json
│   └── tinyStories
│       ├── train.json
│       └── validation.json
└── utils
    ├── train.py                # Training utility functions
    └── validate.py             # Validation utility functions
```

## Usage

### 1. Training the GPT2 Model

To train the GPT2 model on the TinyStories dataset, run the following command:

```bash
python pretrained.py
```

This will:
- Load the TinyStories dataset.
- Fine-tune the GPT2 model using the specified configuration.
- Save model checkpoints periodically to the `checkpoint/` directory.

### 2. Training the Llama Model

To train the Llama model on the TinyStories dataset, run:

```bash
python pretrained_llama.py
```

This will:
- Load the TinyStories dataset.
- Fine-tune the Llama model using the specified configuration.
- Save model checkpoints periodically to the `checkpoint/` directory.

### 3. Generating Text (Inference) with GPT2

Once the GPT2 model is trained, you can generate text with it using the following script:

```bash
python inference.py
```

This will:
- Load the pretrained GPT2 model from a checkpoint.
- Generate text based on a given prompt using both greedy and sampling-based approaches.

You can modify parameters like `max_new_tokens`, `top_k`, `temperature`, and `context_size` to control the generated text.

### 4. Generating Text (Inference) with Llama

Once the Llama model is trained, you can generate text with it using the following script:

```bash
python inference_llama.py
```

This will:
- Load the pretrained Llama model from a checkpoint.
- Generate text based on a given prompt using both greedy and sampling-based approaches.

### Text Generation Parameters:

For both models, the following parameters control the generation process:
- **`max_new_tokens`**: The number of new tokens to generate after the prompt.
- **`context_size`**: The maximum length of context (how much text the model considers at each step).
- **`top_k`**: Limits the token selection to the top `k` most probable tokens.
- **`temperature`**: Controls randomness in sampling. Higher values lead to more random output, while lower values make it more deterministic.

### 5. Checkpoints

Model checkpoints are saved to the `checkpoint/` directory:
- `GPT2_TinyStory.pth` for the GPT2 model.
- `Llama_TinyStory.pth` for the Llama model.

You can specify the checkpoint path in the script to resume training or perform inference on a specific model checkpoint.

## Example Output

For a starting context like `"Once upon a time there was"`, the generated output from the models might look like this:

```
Once upon a time there was a girl named Sarah. She loved to play with her dog in the backyard. One day, while playing, she found a strange key under a tree...
```

## Troubleshooting

- **CUDA Out of Memory Errors**: If you're training the model on a GPU with limited memory, try lowering the `batch_size` or reducing the model's size (e.g., decrease `embed_dim` or `num_layers`).
- **Checkpoint Not Found**: If the checkpoint path is incorrect, ensure the file exists at the specified location or modify the `checkpoint_file` variable to the correct path.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- TinyStories dataset from [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).
- GPT2 model architecture inspired by OpenAI's [GPT-2](https://openai.com/research/language-unsupervised).
- Llama model architecture inspired by [Meta's Llama](https://ai.facebook.com/blog/introducing-llama/).