import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config
import pandas as pd
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib for better plots
plt.style.use('default')
sns.set_palette("husl")

class GPTTokenizationDemo:
    """Comprehensive GPT tokenization and encoding demonstration"""
    
    def __init__(self, model_name='gpt2'):
        """
        Initialize the GPT demo
        
        Args:
            model_name: Name of the GPT model to use (gpt2 for CPU efficiency)
        """
        self.model_name = model_name
        self.device = torch.device('cpu')  # Force CPU for compatibility
        
        print(f"🚀 Initializing GPT Demo")
        print(f"Model: {model_name}")
        print(f"Device: {self.device}")
        print("=" * 50)
        
        # Load tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # 🔧 Fix: GPT-2 has no pad token by default → set it to EOS token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.model = GPT2Model.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Get model configuration
        self.config = self.model.config
        print(f"✅ Model loaded successfully")
        print(f"Hidden size: {self.config.n_embd}")
        print(f"Number of layers: {self.config.n_layer}")
        print(f"Number of attention heads: {self.config.n_head}")
        print(f"Vocabulary size: {self.config.vocab_size}")
        print(f"Context length: {self.config.n_positions}")
        print()
    
    def step1_basic_tokenization(self, text: str) -> Dict:
        """
        Step 1: Basic tokenization process
        
        Args:
            text: Input text to tokenize
            
        Returns:
            Dictionary with tokenization details
        """
        print("📝 STEP 1: Basic Tokenization")
        print("-" * 30)
        print(f"Input text: '{text}'")
        print()
        
        # Use tokenizer's encode method for robust tokenization
        token_ids_with_special = self.tokenizer.encode(text, add_special_tokens=True)
        tokens_with_special = self.tokenizer.convert_ids_to_tokens(token_ids_with_special)
        
        # Get basic tokens without special tokens for display
        basic_token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        tokens = self.tokenizer.convert_ids_to_tokens(basic_token_ids)
        
        # Create attention mask
        attention_mask = [1] * len(token_ids_with_special)
        
        # Pad to max length if needed (GPT-2 has 1024 context length)
        max_length = min(512, self.config.n_positions)  # Use smaller length for CPU efficiency
        if len(token_ids_with_special) < max_length:
            padding_length = max_length - len(token_ids_with_special)
            token_ids_with_special.extend([self.tokenizer.pad_token_id] * padding_length)
            attention_mask.extend([0] * padding_length)
            tokens_with_special.extend([self.tokenizer.pad_token] * padding_length)
        
        result = {
            'original_text': text,
            'basic_tokens': tokens,
            'basic_token_ids': basic_token_ids,
            'tokens_with_special': tokens_with_special,
            'token_ids_with_special': token_ids_with_special,
            'attention_mask': attention_mask,
            'sequence_length': len(tokens_with_special)
        }
        
        # Display results
        print("🔤 Basic tokenization:")
        print(f"  Tokens: {tokens}")
        print(f"  Token IDs: {basic_token_ids}")
        print()
        
        print("🎯 With special tokens:")
        print(f"  Tokens: {tokens_with_special[:20]}{'...' if len(tokens_with_special) > 20 else ''}")
        print(f"  Token IDs: {token_ids_with_special[:20]}{'...' if len(token_ids_with_special) > 20 else ''}")
        print(f"  Attention mask: {attention_mask[:20]}{'...' if len(attention_mask) > 20 else ''}")
        print()
        
        return result
    
    def step2_embedding_layers(self, token_ids: List[int], attention_mask: List[int]) -> Dict:
        """
        Step 2: Embedding layers (Token, Position)
        
        Args:
            token_ids: List of token IDs
            attention_mask: List of attention mask values
            
        Returns:
            Dictionary with embedding details
        """
        print("🧮 STEP 2: Embedding Layers")
        print("-" * 30)
        
        # Convert to tensors
        input_ids = torch.tensor([token_ids]).to(self.device)
        attention_mask = torch.tensor([attention_mask]).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            # Token embeddings
            token_embeddings = self.model.wte(input_ids)  # Word token embeddings
            
            # Position embeddings
            position_embeddings = self.model.wpe(
                torch.arange(input_ids.size(1)).unsqueeze(0).to(self.device)
            )
            
            # Combined embeddings (GPT-2 only uses token + position)
            embeddings = token_embeddings + position_embeddings
            embeddings = self.model.drop(embeddings)  # Dropout layer
        
        result = {
            'token_embeddings': token_embeddings.cpu().numpy(),
            'position_embeddings': position_embeddings.cpu().numpy(),
            'combined_embeddings': embeddings.cpu().numpy(),
            'embedding_shape': embeddings.shape
        }
        
        # Display results
        print(f"📊 Embedding shapes:")
        print(f"  Token embeddings: {token_embeddings.shape}")
        print(f"  Position embeddings: {position_embeddings.shape}")
        print(f"  Combined embeddings: {embeddings.shape}")
        print()
        
        print(f"🔍 Embedding statistics:")
        print(f"  Token embedding range: [{token_embeddings.min():.3f}, {token_embeddings.max():.3f}]")
        print(f"  Position embedding range: [{position_embeddings.min():.3f}, {position_embeddings.max():.3f}]")
        print(f"  Combined embedding range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
        print()
        
        return result
    
    def step3_transformer_layers(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict:
        """
        Step 3: Transformer layers processing
        
        Args:
            input_ids: Input token IDs tensor
            attention_mask: Attention mask tensor
            
        Returns:
            Dictionary with layer outputs and attention weights
        """
        print("🔄 STEP 3: Transformer Layers")
        print("-" * 30)
        
        with torch.no_grad():
            # Get all hidden states and attention weights
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            hidden_states = outputs.hidden_states  # All layer outputs
            attentions = outputs.attentions  # All attention weights
            last_hidden_state = outputs.last_hidden_state
        
        # Analyze each layer
        layer_analysis = []
        for i, (hidden_state, attention) in enumerate(zip(hidden_states, attentions)):
            # Calculate statistics for this layer
            stats = {
                'layer': i,
                'mean': hidden_state.mean().item(),
                'std': hidden_state.std().item(),
                'min': hidden_state.min().item(),
                'max': hidden_state.max().item(),
                'shape': hidden_state.shape
            }
            layer_analysis.append(stats)
        
        result = {
            'hidden_states': [hs.cpu().numpy() for hs in hidden_states],
            'attentions': [att.cpu().numpy() for att in attentions],
            'last_hidden_state': last_hidden_state.cpu().numpy(),
            'layer_analysis': layer_analysis,
            'num_layers': len(hidden_states)
        }
        
        # Display results
        print(f"📈 Layer-by-layer analysis:")
        for i, stats in enumerate(layer_analysis[:5]):  # Show first 5 layers
            print(f"  Layer {i}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, "
                  f"range=[{stats['min']:.3f}, {stats['max']:.3f}]")
        
        if len(layer_analysis) > 5:
            print(f"  ... and {len(layer_analysis) - 5} more layers")
        print()
        
        print(f"🎯 Final output shape: {last_hidden_state.shape}")
        print(f"🔍 Attention weights shape: {attentions[0].shape}")
        print()
        
        return result
    
    def step4_attention_analysis(self, attentions: List[np.ndarray], tokens: List[str], 
                               layer_idx: int = -1, head_idx: int = 0) -> None:
        """
        Step 4: Attention weight analysis and visualization
        
        Args:
            attentions: List of attention weight arrays
            tokens: List of tokens
            layer_idx: Which layer to analyze (-1 for last layer)
            head_idx: Which attention head to visualize
        """
        print("👁️ STEP 4: Attention Analysis")
        print("-" * 30)
        
        # Get attention weights for specified layer and head
        attention_weights = attentions[layer_idx][0, head_idx]  # Remove batch dimension
        
        # Limit to actual tokens (remove padding)
        actual_length = len([t for t in tokens if t != self.tokenizer.pad_token])
        attention_weights = attention_weights[:actual_length, :actual_length]
        actual_tokens = tokens[:actual_length]
        
        print(f"🔍 Analyzing Layer {layer_idx} (last layer), Head {head_idx}")
        print(f"📊 Attention matrix shape: {attention_weights.shape}")
        print()
        
        # Find most attended tokens (GPT-2 doesn't have [CLS], so we look at the last token)
        if len(actual_tokens) > 0:
            last_token_attention = attention_weights[-1, :]  # Last token attention
            top_attended_indices = np.argsort(last_token_attention)[-5:][::-1]
            
            print("🎯 Top 5 tokens that the last token attends to most:")
            for i, idx in enumerate(top_attended_indices):
                token = actual_tokens[idx]
                attention_score = last_token_attention[idx]
                print(f"  {i+1}. '{token}' (attention: {attention_score:.3f})")
            print()
        
        # Visualize attention matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(attention_weights, 
                   xticklabels=actual_tokens, 
                   yticklabels=actual_tokens,
                   cmap='Blues',
                   cbar=True,
                   square=True)
        
        plt.title(f'GPT-2 Attention Weights\nLayer {layer_idx}, Head {head_idx}')
        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def step5_embedding_extraction(self, hidden_states: List[np.ndarray], 
                                 tokens: List[str]) -> Dict:
        """
        Step 5: Extract and analyze embeddings
        
        Args:
            hidden_states: List of hidden state arrays from all layers
            tokens: List of tokens
            
        Returns:
            Dictionary with embedding analysis
        """
        print("🎯 STEP 5: Embedding Extraction")
        print("-" * 30)
        
        # Extract last token embeddings from each layer (GPT-2 doesn't have [CLS])
        last_token_embeddings = []
        for i, hidden_state in enumerate(hidden_states):
            # Get the last non-padding token
            actual_length = len([t for t in tokens if t != self.tokenizer.pad_token])
            if actual_length > 0:
                last_token_embedding = hidden_state[0, actual_length-1, :]  # Last token
                last_token_embeddings.append(last_token_embedding)
        
        # Extract token embeddings from last layer
        last_layer_embeddings = hidden_states[-1][0]  # Remove batch dimension
        actual_length = len([t for t in tokens if t != self.tokenizer.pad_token])
        token_embeddings = last_layer_embeddings[:actual_length]
        actual_tokens = tokens[:actual_length]
        
        # Calculate similarity between last token and other tokens
        if len(last_token_embeddings) > 0 and len(token_embeddings) > 0:
            last_token_embedding = last_token_embeddings[-1]  # Last layer last token
            similarities = []
            for i, token_emb in enumerate(token_embeddings):
                similarity = np.dot(last_token_embedding, token_emb) / (
                    np.linalg.norm(last_token_embedding) * np.linalg.norm(token_emb)
                )
                similarities.append((actual_tokens[i], similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
        else:
            similarities = []
        
        result = {
            'last_token_embeddings': last_token_embeddings,
            'token_embeddings': token_embeddings,
            'similarities': similarities,
            'actual_tokens': actual_tokens
        }
        
        # Display results
        if similarities:
            print("🔍 Last token similarity to other tokens:")
            for i, (token, sim) in enumerate(similarities[:10]):
                print(f"  {i+1:2d}. '{token}' (similarity: {sim:.3f})")
            print()
        
        print(f"📊 Embedding dimensions:")
        if last_token_embeddings:
            print(f"  Last token embedding shape: {last_token_embeddings[-1].shape}")
        print(f"  Token embeddings shape: {token_embeddings.shape}")
        print()
        
        return result
    
    def step6_visualization(self, hidden_states: List[np.ndarray], tokens: List[str]) -> None:
        """
        Step 6: Visualize embeddings and layer evolution
        
        Args:
            hidden_states: List of hidden state arrays from all layers
            tokens: List of tokens
        """
        print("📊 STEP 6: Visualization")
        print("-" * 30)
        
        # Extract last token embeddings from each layer
        last_token_embeddings = []
        for hidden_state in hidden_states:
            actual_length = len([t for t in tokens if t != self.tokenizer.pad_token])
            if actual_length > 0:
                last_token_embedding = hidden_state[0, actual_length-1, :]  # Last token
                last_token_embeddings.append(last_token_embedding)
        
        if not last_token_embeddings:
            print("No valid embeddings to visualize")
            return
            
        last_token_embeddings = np.array(last_token_embeddings)
        
        # Create visualizations
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Layer-wise last token embedding evolution
        layer_norms = [np.linalg.norm(emb) for emb in last_token_embeddings]
        ax1.plot(range(len(layer_norms)), layer_norms, 'b-o', linewidth=2, markersize=6)
        ax1.set_title('Layer-wise Last Token Embedding Norm')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('L2 Norm')
        ax1.grid(True, alpha=0.3)
        
        # 2. Embedding similarity between consecutive layers
        similarities = []
        for i in range(len(last_token_embeddings) - 1):
            sim = np.dot(last_token_embeddings[i], last_token_embeddings[i+1]) / (
                np.linalg.norm(last_token_embeddings[i]) * np.linalg.norm(last_token_embeddings[i+1])
            )
            similarities.append(sim)
        
        ax2.plot(range(1, len(similarities) + 1), similarities, 'r-o', linewidth=2, markersize=6)
        ax2.set_title('Consecutive Layer Similarity')
        ax2.set_xlabel('Layer Transition')
        ax2.set_ylabel('Cosine Similarity')
        ax2.grid(True, alpha=0.3)
        
        # 3. Token embedding heatmap (last layer)
        last_layer_embeddings = hidden_states[-1][0]
        actual_length = len([t for t in tokens if t != self.tokenizer.pad_token])
        token_embeddings = last_layer_embeddings[:actual_length]
        actual_tokens = tokens[:actual_length]
        
        if len(token_embeddings) > 0:
            # Show first 20 dimensions of embeddings
            embedding_subset = token_embeddings[:, :20]
            sns.heatmap(embedding_subset, 
                       xticklabels=[f'dim_{i}' for i in range(20)],
                       yticklabels=actual_tokens,
                       cmap='RdBu_r', center=0, ax=ax3)
            ax3.set_title('Token Embeddings (First 20 Dimensions)')
            ax3.set_xlabel('Embedding Dimensions')
            ax3.set_ylabel('Tokens')
        
        # 4. Embedding magnitude by token
        if len(token_embeddings) > 0:
            token_norms = [np.linalg.norm(emb) for emb in token_embeddings]
            ax4.bar(range(len(actual_tokens)), token_norms, color='skyblue', alpha=0.7)
            ax4.set_title('Token Embedding Magnitudes')
            ax4.set_xlabel('Token Index')
            ax4.set_ylabel('L2 Norm')
            ax4.set_xticks(range(len(actual_tokens)))
            ax4.set_xticklabels(actual_tokens, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
    
    def step7_decoding_analysis(self, token_ids: List[int], tokens: List[str]) -> None:
        """
        Step 7: Decoding and interpretation
        
        Args:
            token_ids: List of token IDs
            tokens: List of tokens
        """
        print("🔍 STEP 7: Decoding Analysis")
        print("-" * 30)
        
        # Decode tokens back to text
        decoded_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        
        print("🔄 Decoding process:")
        print(f"  Original text: '{decoded_text}'")
        print()
        
        # Analyze token types
        special_tokens = []
        regular_tokens = []
        subword_tokens = []
        
        for token in tokens:
            # Convert to string if it's not already
            token_str = str(token)
            
            if token_str.startswith('<') and token_str.endswith('>'):
                special_tokens.append(token_str)
            elif token_str.startswith('Ġ'):  # GPT-2 uses Ġ for word boundaries
                regular_tokens.append(token_str)
            else:
                subword_tokens.append(token_str)
        
        print("📊 Token analysis:")
        print(f"  Special tokens: {special_tokens}")
        print(f"  Regular tokens: {regular_tokens}")
        print(f"  Subword tokens: {subword_tokens}")
        print()
        
        # Show vocabulary information
        vocab_size = self.config.vocab_size
        print(f"📚 Vocabulary information:")
        print(f"  Total vocabulary size: {vocab_size:,}")
        print(f"  Special tokens: {len(self.tokenizer.special_tokens_map)}")
        print(f"  Unknown token ID: {self.tokenizer.unk_token_id}")
        print(f"  End of text token ID: {self.tokenizer.eos_token_id}")
        print()
    
    def run_complete_demo(self, text: str) -> Dict:
        """
        Run the complete GPT tokenization and encoding demo
        
        Args:
            text: Input text to process
            
        Returns:
            Dictionary with all results
        """
        print("🎯 COMPLETE GPT TOKENIZATION DEMO")
        print("=" * 60)
        print(f"Processing text: '{text}'")
        print()
        
        # Step 1: Basic tokenization
        tokenization_result = self.step1_basic_tokenization(text)
        
        # Step 2: Embedding layers
        embedding_result = self.step2_embedding_layers(
            tokenization_result['token_ids_with_special'],
            tokenization_result['attention_mask']
        )
        
        # Step 3: Transformer layers
        input_ids = torch.tensor([tokenization_result['token_ids_with_special']]).to(self.device)
        attention_mask = torch.tensor([tokenization_result['attention_mask']]).to(self.device)
        
        transformer_result = self.step3_transformer_layers(input_ids, attention_mask)
        
        # Step 4: Attention analysis
        self.step4_attention_analysis(
            transformer_result['attentions'],
            tokenization_result['tokens_with_special']
        )
        
        # Step 5: Embedding extraction
        embedding_analysis = self.step5_embedding_extraction(
            transformer_result['hidden_states'],
            tokenization_result['tokens_with_special']
        )
        
        # Step 6: Visualization
        self.step6_visualization(
            transformer_result['hidden_states'],
            tokenization_result['tokens_with_special']
        )
        
        # Step 7: Decoding analysis
        self.step7_decoding_analysis(
            tokenization_result['token_ids_with_special'],
            tokenization_result['tokens_with_special']
        )
        
        print("✅ DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Combine all results
        complete_results = {
            'tokenization': tokenization_result,
            'embeddings': embedding_result,
            'transformer': transformer_result,
            'embedding_analysis': embedding_analysis
        }
        
        return complete_results

def main():
    """Main function to run the GPT tokenization demo"""
    
    # Initialize the demo
    demo = GPTTokenizationDemo('gpt2')  # Using GPT-2 for CPU efficiency
    
    # Example texts to demonstrate different aspects
    example_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "GPT is a generative pre-trained transformer model.",
        "Natural language processing with transformer models.",
        "Tokenization is the process of converting text into tokens."
    ]
    
    print("🎯 Available example texts:")
    for i, text in enumerate(example_texts, 1):
        print(f"  {i}. {text}")
    print()
    
    # Run demo with first example
    print("🚀 Running demo with first example...")
    results = demo.run_complete_demo(example_texts[0])
    
    # Interactive mode
    print("\n" + "="*60)
    print("🎮 INTERACTIVE MODE")
    print("="*60)
    print("You can now test with your own text!")
    print("Type 'quit' to exit.")
    print()
    
    while True:
        try:
            user_text = input("Enter your text: ").strip()
            
            if user_text.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_text:
                print("Please enter some text.")
                continue
            
            print(f"\n🔄 Processing: '{user_text}'")
            print("-" * 50)
            
            # Run demo with user text
            demo.run_complete_demo(user_text)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()
