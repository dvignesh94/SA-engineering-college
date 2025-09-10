# GPT Tokenization Demo

This comprehensive demo shows how GPT-2 processes text through each layer, demonstrating the transformation from raw text to tokens to embeddings and back.

## Features

- **Step-by-step tokenization process**: See how text is broken down into tokens
- **Layer-by-layer encoding visualization**: Watch how embeddings evolve through transformer layers
- **Attention weight analysis**: Visualize what the model focuses on
- **Embedding extraction and visualization**: Analyze token relationships
- **Decoding and interpretation**: Understand how tokens map back to text
- **CPU-optimized**: Designed to run efficiently on CPU without GPU requirements

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the demo:
```bash
python gpt_tokenization_demo.py
```

The demo will:
1. Load the GPT-2 model and tokenizer
2. Process example text through all layers
3. Show visualizations of attention weights and embeddings
4. Enter interactive mode where you can test your own text

## Key Differences from BERT

- **No [CLS] token**: GPT-2 doesn't use a classification token like BERT
- **Causal attention**: GPT-2 uses causal (unidirectional) attention, not bidirectional
- **Different special tokens**: Uses `<|endoftext|>` instead of `[CLS]` and `[SEP]`
- **Word boundary tokens**: Uses `Ġ` prefix to indicate word boundaries
- **Autoregressive**: Designed for text generation, not classification

## Model Information

- **Model**: GPT-2 (117M parameters)
- **Context Length**: 1024 tokens
- **Layers**: 12 transformer layers
- **Attention Heads**: 12 heads per layer
- **Hidden Size**: 768 dimensions

## Example Output

The demo will show:
- Tokenization breakdown
- Embedding statistics
- Layer-by-layer analysis
- Attention weight heatmaps
- Token similarity analysis
- Embedding visualizations

## Interactive Mode

After running the initial demo, you can enter your own text to see how GPT-2 processes it. Type 'quit' to exit.

## CPU Optimization

This demo is optimized for CPU execution by:
- Using GPT-2 instead of larger models
- Limiting sequence length to 512 tokens
- Forcing CPU device usage
- Efficient memory management
