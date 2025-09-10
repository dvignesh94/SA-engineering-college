"""
BERT Layer-by-Layer Tokenization Demo
=====================================

This comprehensive example demonstrates how BERT processes text through each layer,
showing the transformation from raw text to tokens to embeddings and back.

Features:
- Step-by-step tokenization process
- Layer-by-layer encoding visualization
- Attention weight analysis
- Embedding extraction and visualization
- Decoding and interpretation
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertModel, BertConfig
import pandas as pd
from typing import List, Dict, Tuple
import warnings
import os
import sys

# Add the parent directory to the path to import asset_manager
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from asset_manager import AssetManager

warnings.filterwarnings('ignore')

# Set up matplotlib for better plots
plt.style.use('default')
sns.set_palette("husl")

class BERTTokenizationDemo:
    """Comprehensive BERT tokenization and encoding demonstration"""
    
    def __init__(self, model_name='bert-base-uncased'):
        """
        Initialize the BERT demo
        
        Args:
            model_name: Name of the BERT model to use
        """
        # Initialize asset manager
        self.am = AssetManager()
        
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🚀 Initializing BERT Demo")
        print(f"Model: {model_name}")
        print(f"Device: {self.device}")
        print("=" * 50)
        
        # Load tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Get model configuration
        self.config = self.model.config
        print(f"✅ Model loaded successfully")
        print(f"Hidden size: {self.config.hidden_size}")
        print(f"Number of layers: {self.config.num_hidden_layers}")
        print(f"Number of attention heads: {self.config.num_attention_heads}")
        print(f"Vocabulary size: {self.config.vocab_size}")
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
        
        # Basic tokenization
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # Add special tokens
        token_ids_with_special = self.tokenizer.build_inputs_with_special_tokens(token_ids)
        tokens_with_special = self.tokenizer.convert_ids_to_tokens(token_ids_with_special)
        
        # Create attention mask
        attention_mask = [1] * len(token_ids_with_special)
        
        # Pad to max length if needed
        max_length = 512
        if len(token_ids_with_special) < max_length:
            padding_length = max_length - len(token_ids_with_special)
            token_ids_with_special.extend([self.tokenizer.pad_token_id] * padding_length)
            attention_mask.extend([0] * padding_length)
            tokens_with_special.extend([self.tokenizer.pad_token] * padding_length)
        
        result = {
            'original_text': text,
            'basic_tokens': tokens,
            'basic_token_ids': token_ids,
            'tokens_with_special': tokens_with_special,
            'token_ids_with_special': token_ids_with_special,
            'attention_mask': attention_mask,
            'sequence_length': len(tokens_with_special)
        }
        
        # Display results
        print("🔤 Basic tokenization:")
        print(f"  Tokens: {tokens}")
        print(f"  Token IDs: {token_ids}")
        print()
        
        print("🎯 With special tokens:")
        print(f"  Tokens: {tokens_with_special[:20]}{'...' if len(tokens_with_special) > 20 else ''}")
        print(f"  Token IDs: {token_ids_with_special[:20]}{'...' if len(token_ids_with_special) > 20 else ''}")
        print(f"  Attention mask: {attention_mask[:20]}{'...' if len(attention_mask) > 20 else ''}")
        print()
        
        return result
    
    def step2_embedding_layers(self, token_ids: List[int], attention_mask: List[int]) -> Dict:
        """
        Step 2: Embedding layers (Token, Position, Segment)
        
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
            token_embeddings = self.model.embeddings.word_embeddings(input_ids)
            
            # Position embeddings
            position_embeddings = self.model.embeddings.position_embeddings(
                torch.arange(input_ids.size(1)).unsqueeze(0).to(self.device)
            )
            
            # Segment embeddings (all zeros for single sentence)
            segment_embeddings = self.model.embeddings.token_type_embeddings(
                torch.zeros_like(input_ids)
            )
            
            # Combined embeddings
            embeddings = token_embeddings + position_embeddings + segment_embeddings
            embeddings = self.model.embeddings.LayerNorm(embeddings)
            embeddings = self.model.embeddings.dropout(embeddings)
        
        result = {
            'token_embeddings': token_embeddings.cpu().numpy(),
            'position_embeddings': position_embeddings.cpu().numpy(),
            'segment_embeddings': segment_embeddings.cpu().numpy(),
            'combined_embeddings': embeddings.cpu().numpy(),
            'embedding_shape': embeddings.shape
        }
        
        # Display results
        print(f"📊 Embedding shapes:")
        print(f"  Token embeddings: {token_embeddings.shape}")
        print(f"  Position embeddings: {position_embeddings.shape}")
        print(f"  Segment embeddings: {segment_embeddings.shape}")
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
        actual_length = len([t for t in tokens if t != '[PAD]'])
        attention_weights = attention_weights[:actual_length, :actual_length]
        actual_tokens = tokens[:actual_length]
        
        print(f"🔍 Analyzing Layer {layer_idx} (last layer), Head {head_idx}")
        print(f"📊 Attention matrix shape: {attention_weights.shape}")
        print()
        
        # Find most attended tokens
        cls_attention = attention_weights[0, :]  # [CLS] token attention
        top_attended_indices = np.argsort(cls_attention)[-5:][::-1]
        
        print("🎯 Top 5 tokens that [CLS] attends to most:")
        for i, idx in enumerate(top_attended_indices):
            token = actual_tokens[idx]
            attention_score = cls_attention[idx]
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
        
        plt.title(f'BERT Attention Weights\nLayer {layer_idx}, Head {head_idx}')
        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot using asset manager
        plot_path = self.am.get_asset_path('transformers', 'results', f'bert_attention_layer_{layer_idx}_head_{head_idx}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Register the plot asset
        self.am.register_asset('transformers', 'results', f'bert_attention_layer_{layer_idx}_head_{head_idx}.png', 
                              plot_path, {'model': 'BERT', 'layer': layer_idx, 'head': head_idx})
        print(f"📊 Attention plot saved: {plot_path}")
    
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
        
        # Extract [CLS] token embeddings from each layer
        cls_embeddings = []
        for i, hidden_state in enumerate(hidden_states):
            cls_embedding = hidden_state[0, 0, :]  # [CLS] token from first sequence
            cls_embeddings.append(cls_embedding)
        
        # Extract token embeddings from last layer
        last_layer_embeddings = hidden_states[-1][0]  # Remove batch dimension
        actual_length = len([t for t in tokens if t != '[PAD]'])
        token_embeddings = last_layer_embeddings[:actual_length]
        actual_tokens = tokens[:actual_length]
        
        # Calculate similarity between [CLS] and other tokens
        cls_embedding = cls_embeddings[-1]  # Last layer [CLS]
        similarities = []
        for i, token_emb in enumerate(token_embeddings):
            similarity = np.dot(cls_embedding, token_emb) / (
                np.linalg.norm(cls_embedding) * np.linalg.norm(token_emb)
            )
            similarities.append((actual_tokens[i], similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        result = {
            'cls_embeddings': cls_embeddings,
            'token_embeddings': token_embeddings,
            'similarities': similarities,
            'actual_tokens': actual_tokens
        }
        
        # Display results
        print("🔍 [CLS] token similarity to other tokens:")
        for i, (token, sim) in enumerate(similarities[:10]):
            print(f"  {i+1:2d}. '{token}' (similarity: {sim:.3f})")
        print()
        
        print(f"📊 Embedding dimensions:")
        print(f"  [CLS] embedding shape: {cls_embedding.shape}")
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
        
        # Extract [CLS] embeddings from each layer
        cls_embeddings = []
        for hidden_state in hidden_states:
            cls_embedding = hidden_state[0, 0, :]  # [CLS] token
            cls_embeddings.append(cls_embedding)
        
        cls_embeddings = np.array(cls_embeddings)
        
        # Create visualizations
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Layer-wise [CLS] embedding evolution
        layer_norms = [np.linalg.norm(emb) for emb in cls_embeddings]
        ax1.plot(range(len(layer_norms)), layer_norms, 'b-o', linewidth=2, markersize=6)
        ax1.set_title('Layer-wise [CLS] Embedding Norm')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('L2 Norm')
        ax1.grid(True, alpha=0.3)
        
        # 2. Embedding similarity between consecutive layers
        similarities = []
        for i in range(len(cls_embeddings) - 1):
            sim = np.dot(cls_embeddings[i], cls_embeddings[i+1]) / (
                np.linalg.norm(cls_embeddings[i]) * np.linalg.norm(cls_embeddings[i+1])
            )
            similarities.append(sim)
        
        ax2.plot(range(1, len(similarities) + 1), similarities, 'r-o', linewidth=2, markersize=6)
        ax2.set_title('Consecutive Layer Similarity')
        ax2.set_xlabel('Layer Transition')
        ax2.set_ylabel('Cosine Similarity')
        ax2.grid(True, alpha=0.3)
        
        # 3. Token embedding heatmap (last layer)
        last_layer_embeddings = hidden_states[-1][0]
        actual_length = len([t for t in tokens if t != '[PAD]'])
        token_embeddings = last_layer_embeddings[:actual_length]
        actual_tokens = tokens[:actual_length]
        
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
        token_norms = [np.linalg.norm(emb) for emb in token_embeddings]
        ax4.bar(range(len(actual_tokens)), token_norms, color='skyblue', alpha=0.7)
        ax4.set_title('Token Embedding Magnitudes')
        ax4.set_xlabel('Token Index')
        ax4.set_ylabel('L2 Norm')
        ax4.set_xticks(range(len(actual_tokens)))
        ax4.set_xticklabels(actual_tokens, rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot using asset manager
        plot_path = self.am.get_asset_path('transformers', 'results', 'bert_embedding_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Register the plot asset
        self.am.register_asset('transformers', 'results', 'bert_embedding_analysis.png', 
                              plot_path, {'model': 'BERT', 'type': 'embedding_analysis'})
        print(f"📊 Embedding analysis plot saved: {plot_path}")
    
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
        
        # Analyze token types - ensure we're working with strings
        special_tokens = []
        regular_tokens = []
        subword_tokens = []
        
        for token in tokens:
            # Convert to string if it's not already
            token_str = str(token)
            
            if token_str.startswith('[') and token_str.endswith(']'):
                special_tokens.append(token_str)
            elif token_str.startswith('##'):
                subword_tokens.append(token_str)
            else:
                regular_tokens.append(token_str)
        
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
        print(f"  Padding token ID: {self.tokenizer.pad_token_id}")
        print()
    
    def run_complete_demo(self, text: str) -> Dict:
        """
        Run the complete BERT tokenization and encoding demo
        
        Args:
            text: Input text to process
            
        Returns:
            Dictionary with all results
        """
        print("🎯 COMPLETE BERT TOKENIZATION DEMO")
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
    """Main function to run the BERT tokenization demo"""
    
    # Initialize the demo
    demo = BERTTokenizationDemo('bert-base-uncased')
    
    # Example texts to demonstrate different aspects
    example_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "BERT is a bidirectional encoder representations from transformers.",
        "Natural language processing with deep learning models.",
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
