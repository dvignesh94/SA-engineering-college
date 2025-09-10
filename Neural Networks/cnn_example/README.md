CNN: Prioritizes local spatial hierarchies. It assumes that pixels close to each other are more related and builds up from local patterns (edges) to global patterns (objects) using convolutional filters and pooling layers.

Transformer: Prioritizes global context via attention. It uses the self-attention mechanism to weigh the importance of every part of the input with respect to every other part, regardless of distance, right from the first layer.

# CNN vs Transformer Image Processing Demonstration

This demonstration shows the fundamental differences between CNNs and Transformers when processing images, with visual outputs to illustrate these differences.

## The Key Difference

- **CNN**: Processes images using local convolutions, building up from local patterns to global understanding
- **Transformer**: Processes images using global attention, allowing any part to attend to any other part

## What This Demo Shows

1. **Input Image**: A simple 32x32 image with distinct colored regions
2. **CNN Processing**: Shows how convolutions detect local features layer by layer
3. **Transformer Processing**: Shows how attention mechanisms create global relationships
4. **Visual Comparison**: Side-by-side visualization of both approaches

## Running the Demonstration

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the demo:
```bash
python cnn_vs_transformer_demo.py
```

## Expected Output

The script will:
- Create a sample image with colored regions
- Process it through both CNN and Transformer models
- Generate a visualization showing:
  - Original image
  - CNN activations at each layer
  - Transformer attention maps at each layer
- Save the comparison as `cnn_vs_transformer_comparison.png`

## Key Insights

### CNN Characteristics:
- **Local Processing**: Each convolution looks at small local regions
- **Hierarchical**: Builds from edges → textures → objects
- **Spatial Inductive Bias**: Assumes nearby pixels are related
- **Activations**: Show local feature detection patterns

### Transformer Characteristics:
- **Global Processing**: Attention can connect any two patches
- **Parallel**: All patches processed simultaneously
- **No Spatial Bias**: No assumption about spatial relationships
- **Attention Maps**: Show global relationship patterns

## Visual Interpretation

- **CNN Activations**: Look for local, structured patterns that build up
- **Transformer Attention**: Look for distributed, global attention patterns
- **Color Coding**: 
  - CNN: Viridis colormap showing activation strength
  - Transformer: Hot colormap showing attention weights

## Why This Matters

This demonstrates why:
- CNNs are good for local feature detection
- Transformers excel at capturing long-range dependencies
- Vision Transformers can outperform CNNs on certain tasks
- The choice between CNN and Transformer depends on the task requirements