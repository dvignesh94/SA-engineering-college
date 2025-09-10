# Interactive RNN Demo

A real-time demonstration of how Recurrent Neural Networks (RNNs) process text, showing hidden state evolution and predictions step-by-step.

## Overview

This interactive demo allows you to input your own text and see how an RNN processes it in real-time. You can analyze single texts or compare two texts to understand how RNNs build internal representations and make predictions.

## How the Program Works

### 1. **RNN Architecture**

The program implements a simple RNN with the following components:

- **Embedding Layer**: Converts word indices to dense vector representations (32 dimensions)
- **RNN Layer**: Processes sequences with hidden states (64 dimensions)
- **Fully Connected Layer**: Maps final hidden state to binary classification (2 classes)

```python
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=64, embedding_size=32):
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.RNN(embedding_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)  # Binary classification
```

### 2. **Vocabulary System**

The program includes a pre-built vocabulary of 85 common English words:
- **Special tokens**: `<PAD>` (padding), `<UNK>` (unknown words)
- **Common words**: Articles, pronouns, verbs, adjectives, nouns
- **Word-to-index mapping**: Each word gets a unique numerical ID

**Why this vocabulary system is included:**
- **Neural networks require numerical input**: RNNs can't process text directly, so words must be converted to numbers
- **Consistent representation**: Each word gets a fixed numerical ID for reproducible processing
- **Handles unknown words**: The `<UNK>` token allows the model to process words not in the vocabulary
- **Padding support**: The `<PAD>` token enables batch processing of variable-length sequences
- **Educational clarity**: A small, manageable vocabulary makes it easier to understand how word-to-number conversion works
- **Real-world simulation**: Mimics how production NLP systems handle vocabulary limitations

### 3. **Text Processing Pipeline**

When you input text, the program:

1. **Tokenization**: Splits text into individual words using regex
2. **Lowercasing**: Converts all text to lowercase for consistency
3. **Index Conversion**: Maps each word to its vocabulary index
4. **Tensor Creation**: Converts indices to PyTorch tensors for processing

### 4. **RNN Processing**

For each input text:

1. **Embedding**: Words are converted to dense vectors
2. **Sequential Processing**: RNN processes each word sequentially
3. **Hidden State Evolution**: Internal state updates with each word
4. **Final Prediction**: Last hidden state is used for classification

### 5. **Analysis Features**

#### Single Text Analysis
- **Hidden State Statistics**: Norm, mean, and standard deviation
- **Step-by-Step Evolution**: How hidden state changes with each word
- **Prediction Confidence**: Class probabilities and confidence scores
- **Visualization**: Plots showing hidden state evolution and predictions

#### Text Comparison
- **Similarity Analysis**: Cosine similarity between hidden states
- **Side-by-Side Comparison**: Visual comparison of two texts
- **Prediction Differences**: How different texts lead to different predictions

### 6. **Interactive Commands**

- **Text Input**: Enter any text to analyze it
- **Compare Mode**: Type `compare` to compare two texts
- **Exit**: Type `quit` to exit the program

## Key Concepts Demonstrated

### Hidden State Evolution
The RNN's hidden state changes as it processes each word, building up a representation of the entire sequence. This evolution is visualized step-by-step.

### Memory and Context
RNNs maintain memory of previous words through their hidden state, allowing them to consider context when processing new words.

### Text Similarity
By comparing hidden states, you can see how semantically similar texts produce similar internal representations.

### Prediction Confidence
The model outputs probabilities for each class, showing how confident it is in its predictions.

## Usage Examples

### Basic Analysis
```
> The quick brown fox
```
Shows how the RNN processes each word and builds up its understanding.

### Comparison Mode
```
> compare
Text 1> I love this movie
Text 2> I hate this movie
```
Compares how positive and negative sentiment affects the hidden state.

## Technical Details

- **Model**: Untrained RNN (random weights) for demonstration purposes
- **Input**: Variable-length text sequences
- **Output**: Binary classification with confidence scores
- **Visualization**: Real-time plots using matplotlib
- **Memory**: Hidden states are preserved and analyzed

## Dependencies

- `torch>=1.9.0` - PyTorch for neural network implementation
- `numpy>=1.21.0` - Numerical operations
- `matplotlib>=3.4.0` - Plotting and visualization

## Running the Demo

1. Install dependencies: `pip install -r requirements.txt`
2. Run the demo: `python interactive_rnn_demo.py`
3. Follow the interactive prompts to analyze text

## Educational Value

This demo helps understand:
- How RNNs process sequential data
- The role of hidden states in maintaining context
- How text similarity affects internal representations
- The relationship between input sequences and predictions
- Real-time visualization of neural network processing

Perfect for students, researchers, and anyone interested in understanding how RNNs work with text data!
