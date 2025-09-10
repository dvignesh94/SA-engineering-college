import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import re
import os
import sys

# Add the parent directory to the path to import asset_manager
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from asset_manager import AssetManager

class SimpleRNN(nn.Module):
    """A simple RNN for text processing"""
    
    def __init__(self, vocab_size: int, hidden_size: int = 64, embedding_size: int = 32):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.RNN(embedding_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)  # Binary classification
        
    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, hidden = self.rnn(embedded)
        output = self.fc(rnn_out[:, -1, :])  # Use last output
        return output, hidden, rnn_out

class InteractiveRNNDemo:
    def __init__(self):
        # Initialize asset manager
        self.am = AssetManager()
        
        # Create vocabulary from common words
        self.vocab = {
            '<PAD>': 0, '<UNK>': 1,
            'the': 2, 'a': 3, 'an': 4, 'and': 5, 'or': 6, 'but': 7,
            'is': 8, 'are': 9, 'was': 10, 'were': 11, 'be': 12, 'been': 13,
            'have': 14, 'has': 15, 'had': 16, 'do': 17, 'does': 18, 'did': 19,
            'will': 20, 'would': 21, 'could': 22, 'should': 23, 'can': 24, 'may': 25,
            'i': 26, 'you': 27, 'he': 28, 'she': 29, 'it': 30, 'we': 31, 'they': 32,
            'this': 33, 'that': 34, 'these': 35, 'those': 36,
            'in': 37, 'on': 38, 'at': 39, 'by': 40, 'for': 41, 'with': 42, 'from': 43,
            'to': 44, 'of': 45, 'about': 46, 'into': 47, 'through': 48, 'during': 49,
            'good': 50, 'bad': 51, 'big': 52, 'small': 53, 'new': 54, 'old': 55,
            'first': 56, 'last': 57, 'long': 58, 'short': 59, 'high': 60, 'low': 61,
            'great': 62, 'little': 63, 'own': 64, 'other': 65, 'same': 66, 'different': 67,
            'time': 68, 'way': 69, 'day': 70, 'year': 71, 'work': 72, 'life': 73,
            'man': 74, 'woman': 75, 'child': 76, 'people': 77, 'world': 78, 'country': 79,
            'house': 80, 'home': 81, 'school': 82, 'work': 83, 'place': 84, 'thing': 85
        }
        
        self.idx_to_word = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        
        # Initialize model
        self.model = SimpleRNN(self.vocab_size, hidden_size=64, embedding_size=32)
        self.model.eval()
        
        print("🤖 Interactive RNN Demo Initialized!")
        print("=" * 50)
        print("Enter text to see how the RNN processes it step by step.")
        print("Type 'quit' to exit, 'compare' to compare two texts.")
        print("=" * 50)
    
    def text_to_tensor(self, text: str) -> torch.Tensor:
        """Convert text to tensor"""
        words = re.findall(r'\b\w+\b', text.lower())
        indices = [self.vocab.get(word, 1) for word in words]  # 1 = <UNK>
        
        if not indices:
            indices = [0]  # <PAD>
            
        return torch.tensor([indices], dtype=torch.long)
    
    def analyze_text(self, text: str) -> dict:
        """Analyze a single text"""
        tensor = self.text_to_tensor(text)
        
        with torch.no_grad():
            output, hidden, rnn_out = self.model(tensor)
            probabilities = torch.softmax(output, dim=1)
            
            # Calculate hidden state statistics
            hidden_norm = torch.norm(hidden).item()
            hidden_mean = torch.mean(hidden).item()
            hidden_std = torch.std(hidden).item()
            
            # Get step-by-step hidden states
            step_norms = [torch.norm(rnn_out[0, i, :]).item() for i in range(rnn_out.shape[1])]
            
        return {
            'text': text,
            'tensor': tensor,
            'output': output,
            'hidden': hidden,
            'rnn_out': rnn_out,
            'probabilities': probabilities,
            'hidden_norm': hidden_norm,
            'hidden_mean': hidden_mean,
            'hidden_std': hidden_std,
            'step_norms': step_norms,
            'words': re.findall(r'\b\w+\b', text.lower())
        }
    
    def compare_texts(self, text1: str, text2: str):
        """Compare two texts"""
        print(f"\n🔍 COMPARING TEXTS")
        print("=" * 60)
        
        result1 = self.analyze_text(text1)
        result2 = self.analyze_text(text2)
        
        # Calculate similarity
        similarity = torch.cosine_similarity(
            result1['hidden'].flatten(), 
            result2['hidden'].flatten(), 
            dim=0
        ).item()
        
        print(f"Text 1: '{text1}'")
        print(f"Text 2: '{text2}'")
        print(f"\n📊 HIDDEN STATE COMPARISON:")
        print(f"Text 1 hidden norm: {result1['hidden_norm']:.4f}")
        print(f"Text 2 hidden norm: {result2['hidden_norm']:.4f}")
        print(f"Hidden state similarity: {similarity:.4f}")
        
        print(f"\n🎯 PREDICTIONS:")
        pred1 = torch.argmax(result1['output']).item()
        pred2 = torch.argmax(result2['output']).item()
        conf1 = result1['probabilities'][0][pred1].item()
        conf2 = result2['probabilities'][0][pred2].item()
        
        print(f"Text 1 prediction: Class {pred1} (confidence: {conf1:.4f})")
        print(f"Text 2 prediction: Class {pred2} (confidence: {conf2:.4f})")
        
        # Visualize step-by-step processing
        self.plot_comparison(result1, result2)
    
    def plot_comparison(self, result1: dict, result2: dict):
        """Plot comparison of two texts"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Step-by-step hidden state norms
        ax1.plot(result1['step_norms'], 'b-o', label=result1['text'][:20] + '...')
        ax1.plot(result2['step_norms'], 'r-s', label=result2['text'][:20] + '...')
        ax1.set_title('Hidden State Evolution')
        ax1.set_xlabel('Word Position')
        ax1.set_ylabel('Hidden State Norm')
        ax1.legend()
        ax1.grid(True)
        
        # Hidden state distributions
        ax2.hist(result1['hidden'].flatten().numpy(), alpha=0.7, bins=20, label=result1['text'][:20] + '...')
        ax2.hist(result2['hidden'].flatten().numpy(), alpha=0.7, bins=20, label=result2['text'][:20] + '...')
        ax2.set_title('Hidden State Distribution')
        ax2.set_xlabel('Hidden State Value')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        # Prediction probabilities
        probs1 = result1['probabilities'][0].numpy()
        probs2 = result2['probabilities'][0].numpy()
        x = ['Class 0', 'Class 1']
        width = 0.35
        
        ax3.bar([x_pos - width/2 for x_pos in range(len(x))], probs1, width, label=result1['text'][:20] + '...')
        ax3.bar([x_pos + width/2 for x_pos in range(len(x))], probs2, width, label=result2['text'][:20] + '...')
        ax3.set_title('Prediction Probabilities')
        ax3.set_ylabel('Probability')
        ax3.set_xticks(range(len(x)))
        ax3.set_xticklabels(x)
        ax3.legend()
        
        # Word processing visualization
        words1 = result1['words']
        words2 = result2['words']
        max_len = max(len(words1), len(words2))
        
        # Pad shorter sequence
        words1_padded = words1 + [''] * (max_len - len(words1))
        words2_padded = words2 + [''] * (max_len - len(words2))
        
        ax4.text(0.1, 0.7, f"Text 1: {' → '.join(words1_padded)}", transform=ax4.transAxes, fontsize=10)
        ax4.text(0.1, 0.5, f"Text 2: {' → '.join(words2_padded)}", transform=ax4.transAxes, fontsize=10)
        ax4.text(0.1, 0.3, f"Similarity: {torch.cosine_similarity(result1['hidden'].flatten(), result2['hidden'].flatten(), dim=0).item():.4f}", 
                transform=ax4.transAxes, fontsize=12, weight='bold')
        ax4.set_title('Text Processing Summary')
        ax4.axis('off')
        
        plt.tight_layout()
        
        # Save plot using asset manager
        plot_path = self.am.get_asset_path('neural_networks', 'plots', 'rnn_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Register the plot asset
        self.am.register_asset('neural_networks', 'plots', 'rnn_comparison.png', 
                              plot_path, {'type': 'comparison', 'model': 'RNN'})
        print(f"📊 Comparison plot saved: {plot_path}")
    
    def analyze_single_text(self, text: str):
        """Analyze a single text with detailed output"""
        print(f"\n🔍 ANALYZING: '{text}'")
        print("=" * 50)
        
        result = self.analyze_text(text)
        
        print(f"📝 Text breakdown:")
        print(f"Words: {result['words']}")
        print(f"Tensor shape: {result['tensor'].shape}")
        
        print(f"\n🧠 Hidden State Analysis:")
        print(f"Final hidden state norm: {result['hidden_norm']:.4f}")
        print(f"Hidden state mean: {result['hidden_mean']:.4f}")
        print(f"Hidden state std: {result['hidden_std']:.4f}")
        
        print(f"\n📊 Step-by-step hidden state norms:")
        for i, norm in enumerate(result['step_norms']):
            word = result['words'][i] if i < len(result['words']) else '<END>'
            print(f"  Step {i+1} ('{word}'): {norm:.4f}")
        
        print(f"\n🎯 Prediction:")
        pred = torch.argmax(result['output']).item()
        confidence = result['probabilities'][0][pred].item()
        print(f"Predicted class: {pred}")
        print(f"Confidence: {confidence:.4f}")
        print(f"All probabilities: {result['probabilities'][0].numpy()}")
        
        # Plot single text analysis
        self.plot_single_analysis(result)
    
    def plot_single_analysis(self, result: dict):
        """Plot analysis for a single text"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Hidden state evolution
        ax1.plot(result['step_norms'], 'b-o', linewidth=2, markersize=6)
        ax1.set_title(f'Hidden State Evolution: "{result["text"]}"')
        ax1.set_xlabel('Word Position')
        ax1.set_ylabel('Hidden State Norm')
        ax1.grid(True)
        
        # Add word labels
        for i, word in enumerate(result['words']):
            ax1.annotate(word, (i, result['step_norms'][i]), 
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', fontsize=8)
        
        # Prediction probabilities
        probs = result['probabilities'][0].numpy()
        classes = ['Class 0', 'Class 1']
        bars = ax2.bar(classes, probs, color=['skyblue', 'lightcoral'])
        ax2.set_title('Prediction Probabilities')
        ax2.set_ylabel('Probability')
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot using asset manager
        plot_path = self.am.get_asset_path('neural_networks', 'plots', 'rnn_single_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Register the plot asset
        self.am.register_asset('neural_networks', 'plots', 'rnn_single_analysis.png', 
                              plot_path, {'type': 'single_analysis', 'model': 'RNN'})
        print(f"📊 Single analysis plot saved: {plot_path}")
    
    def run(self):
        """Main interactive loop"""
        while True:
            print(f"\n💬 Enter your text (or 'quit' to exit, 'compare' for comparison):")
            user_input = input("> ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            elif user_input.lower() == 'compare':
                print("Enter first text:")
                text1 = input("Text 1> ").strip()
                print("Enter second text:")
                text2 = input("Text 2> ").strip()
                
                if text1 and text2:
                    self.compare_texts(text1, text2)
                else:
                    print("❌ Please enter both texts!")
            
            elif user_input:
                self.analyze_single_text(user_input)
            
            else:
                print("❌ Please enter some text!")

if __name__ == "__main__":
    demo = InteractiveRNNDemo()
    demo.run()
