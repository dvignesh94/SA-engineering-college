import streamlit as st
import pandas as pd
import requests
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List, Dict, Tuple
import os
import sys

# Add the parent directory to the path to import asset_manager
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from asset_manager import AssetManager

# Configure Streamlit page
st.set_page_config(
    page_title="Industrial Safety Incident Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .severity-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .severity-medium {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .severity-low {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 2rem;
    }
    .bot-message {
        background-color: #f5f5f5;
        margin-right: 2rem;
    }
</style>
""", unsafe_allow_html=True)

class SafetyIncidentClassifier:
    def __init__(self, dataset_path: str):
        # Initialize asset manager
        self.am = AssetManager()
        
        self.dataset_path = dataset_path
        self.df = None
        self.embeddings = None
        self.model = None
        self.ollama_api_key = "75abea0b9b8d4d329432a2fbb6fcf1c8.dr3IdxNsGm0IMSgpN4pKIdDt"
        self.ollama_url = "https://ollama.com/api/chat"
        
    def load_data(self):
        """Load and preprocess the safety dataset"""
        try:
            self.df = pd.read_csv(self.dataset_path)
            st.success(f"✅ Loaded {len(self.df)} safety incidents from dataset")
            return True
        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)}")
            return False
    
    def initialize_embeddings(self):
        """Initialize sentence transformer model and create embeddings"""
        try:
            with st.spinner("🔄 Initializing AI model for similarity search..."):
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                
                # Create embeddings for incident descriptions
                descriptions = self.df['Description'].fillna('').astype(str).tolist()
                self.embeddings = self.model.encode(descriptions)
                
                # Save embeddings using asset manager
                embeddings_path = self.am.get_asset_path('safety_chatbot', 'embeddings', 'safety_incident_embeddings.npy')
                np.save(embeddings_path, self.embeddings)
                
                # Register the embeddings asset
                self.am.register_asset('safety_chatbot', 'embeddings', 'safety_incident_embeddings.npy', 
                                     embeddings_path, {'model': 'all-MiniLM-L6-v2', 'shape': self.embeddings.shape})
                
            st.success("✅ AI model initialized successfully")
            return True
        except Exception as e:
            st.error(f"❌ Error initializing embeddings: {str(e)}")
            return False
    
    def find_similar_incidents(self, query: str, top_k: int = 5) -> List[Dict]:
        """Find similar incidents using semantic search"""
        try:
            # Encode the query
            query_embedding = self.model.encode([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Get top-k most similar incidents
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            similar_incidents = []
            for idx in top_indices:
                incident = self.df.iloc[idx].to_dict()
                incident['similarity_score'] = similarities[idx]
                similar_incidents.append(incident)
            
            return similar_incidents
        except Exception as e:
            st.error(f"❌ Error finding similar incidents: {str(e)}")
            return []
    
    def classify_severity(self, description: str) -> str:
        """Classify incident severity based on keywords and patterns"""
        description_lower = description.lower()
        
        # High severity keywords
        high_severity_keywords = [
            'death', 'fatal', 'amputation', 'crush', 'electrocution', 'explosion',
            'fire', 'burn', 'chemical exposure', 'toxic', 'asphyxiation', 'fall from height',
            'structural collapse', 'equipment failure', 'machinery accident'
        ]
        
        # Medium severity keywords
        medium_severity_keywords = [
            'injury', 'cut', 'laceration', 'fracture', 'sprain', 'strain',
            'contusion', 'abrasion', 'minor burn', 'eye injury', 'hand injury',
            'slip', 'trip', 'fall', 'struck by', 'caught between'
        ]
        
        # Check for high severity
        for keyword in high_severity_keywords:
            if keyword in description_lower:
                return "HIGH"
        
        # Check for medium severity
        for keyword in medium_severity_keywords:
            if keyword in description_lower:
                return "MEDIUM"
        
        return "LOW"
    
    def get_recommended_actions(self, severity: str, incident_type: str) -> List[str]:
        """Get recommended actions based on severity and incident type"""
        actions = {
            "HIGH": [
                "🚨 IMMEDIATE: Evacuate the area and call emergency services",
                "🛡️ Secure the scene to prevent further incidents",
                "📋 Conduct immediate investigation and root cause analysis",
                "🔧 Implement immediate corrective actions",
                "📊 Report to regulatory authorities if required",
                "👥 Provide immediate medical attention and support",
                "📝 Document all details for future prevention"
            ],
            "MEDIUM": [
                "🏥 Provide immediate first aid and medical attention",
                "🔍 Investigate the incident thoroughly",
                "📋 Document the incident and contributing factors",
                "🔧 Implement corrective actions within 24 hours",
                "👥 Conduct safety briefing with affected personnel",
                "📊 Review and update safety procedures",
                "🎯 Provide additional training if needed"
            ],
            "LOW": [
                "🏥 Provide first aid if needed",
                "📝 Document the incident",
                "🔍 Investigate contributing factors",
                "🔧 Implement preventive measures",
                "👥 Discuss with team for awareness",
                "📊 Review safety procedures",
                "🎯 Consider additional training"
            ]
        }
        
        return actions.get(severity, actions["LOW"])
    
    def call_ollama_api(self, messages: List[Dict]) -> str:
        """Call Ollama API for generating responses"""
        try:
            headers = {
                "Authorization": f"Bearer {self.ollama_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "gpt-oss:120b",
                "messages": messages,
                "stream": False
            }
            
            response = requests.post(self.ollama_url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('message', {}).get('content', 'No response generated')
            else:
                st.error(f"❌ API Error: {response.status_code} - {response.text}")
                return "Error generating response"
                
        except Exception as e:
            st.error(f"❌ Error calling Ollama API: {str(e)}")
            return "Error generating response"

def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ Industrial Safety Incident Classifier</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Safety Analysis and Recommendations")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'classifier' not in st.session_state:
        st.session_state.classifier = None
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Initialize asset manager and get dataset path
        am = AssetManager()
        dataset_path = am.get_asset_path('safety_chatbot', 'datasets', 'Industrial_safety_and_health_database_with_accidents_description.csv')
        
        if st.button("🚀 Initialize System", type="primary"):
            with st.spinner("Initializing safety incident classifier..."):
                classifier = SafetyIncidentClassifier(dataset_path)
                
                if classifier.load_data() and classifier.initialize_embeddings():
                    st.session_state.classifier = classifier
                    st.success("✅ System ready!")
                else:
                    st.error("❌ Failed to initialize system")
        
        st.markdown("---")
        st.markdown("### 📊 Dataset Info")
        if st.session_state.classifier:
            st.info(f"**Total Incidents:** {len(st.session_state.classifier.df)}")
            st.info(f"**Industries:** {st.session_state.classifier.df['Industry Sector'].nunique()}")
            st.info(f"**Risk Categories:** {st.session_state.classifier.df['Critical Risk'].nunique()}")
        
        st.markdown("---")
        st.markdown("### 🎯 How to Use")
        st.markdown("""
        1. Click "Initialize System" to load the dataset
        2. Describe a safety incident in the chat
        3. Get AI-powered analysis including:
           - Severity classification
           - Similar past incidents
           - Recommended actions
           - Risk assessment
        """)
    
    # Main chat interface
    if st.session_state.classifier is None:
        st.warning("⚠️ Please initialize the system first using the sidebar.")
        return
    
    # Chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Describe a safety incident to analyze..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process the incident
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing incident..."):
                # Find similar incidents
                similar_incidents = st.session_state.classifier.find_similar_incidents(prompt, top_k=3)
                
                # Classify severity
                severity = st.session_state.classifier.classify_severity(prompt)
                
                # Get recommended actions
                actions = st.session_state.classifier.get_recommended_actions(severity, "general")
                
                # Prepare context for Ollama
                context = f"""
                Based on the following safety incident description and similar past incidents, provide a comprehensive analysis:
                
                Current Incident: {prompt}
                
                Similar Past Incidents:
                """
                
                for i, incident in enumerate(similar_incidents, 1):
                    context += f"""
                    {i}. Severity: {incident.get('Accident Level', 'Unknown')} | 
                    Risk: {incident.get('Critical Risk', 'Unknown')} | 
                    Description: {incident.get('Description', 'No description')[:200]}...
                    """
                
                context += f"""
                
                Please provide:
                1. Severity assessment (HIGH/MEDIUM/LOW)
                2. Key risk factors identified
                3. Immediate actions required
                4. Preventive measures
                5. Industry best practices
                
                Be concise but comprehensive in your analysis.
                """
                
                # Call Ollama API
                messages = [
                    {"role": "system", "content": "You are an expert industrial safety analyst. Analyze safety incidents and provide detailed recommendations."},
                    {"role": "user", "content": context}
                ]
                
                ai_response = st.session_state.classifier.call_ollama_api(messages)
                
                # Display results
                st.markdown("### 🎯 Incident Analysis")
                
                # Severity classification with color coding
                severity_colors = {
                    "HIGH": "severity-high",
                    "MEDIUM": "severity-medium", 
                    "LOW": "severity-low"
                }
                
                st.markdown(f'<div class="{severity_colors.get(severity, "severity-low")}">', unsafe_allow_html=True)
                st.markdown(f"**🚨 Severity Level: {severity}**")
                st.markdown("</div>", unsafe_allow_html=True)
                
                # AI Response
                st.markdown("### 🤖 AI Analysis")
                st.markdown(ai_response)
                
                # Similar incidents
                if similar_incidents:
                    st.markdown("### 📋 Similar Past Incidents")
                    for i, incident in enumerate(similar_incidents, 1):
                        with st.expander(f"Incident {i} (Similarity: {incident['similarity_score']:.2f})"):
                            st.markdown(f"**Date:** {incident.get('Data', 'Unknown')}")
                            st.markdown(f"**Industry:** {incident.get('Industry Sector', 'Unknown')}")
                            st.markdown(f"**Severity:** {incident.get('Accident Level', 'Unknown')}")
                            st.markdown(f"**Risk Category:** {incident.get('Critical Risk', 'Unknown')}")
                            st.markdown(f"**Description:** {incident.get('Description', 'No description')}")
                
                # Recommended actions
                st.markdown("### 🎯 Recommended Actions")
                for action in actions:
                    st.markdown(action)
                
                # Add assistant response to messages
                response_content = f"""
                **Severity:** {severity}
                
                **AI Analysis:**
                {ai_response}
                
                **Recommended Actions:**
                {chr(10).join(actions)}
                """
                
                st.session_state.messages.append({"role": "assistant", "content": response_content})

if __name__ == "__main__":
    main()
