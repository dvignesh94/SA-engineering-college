# 🛡️ Industrial Safety Incident Classifier

A RAG-based chatbot built with Streamlit that classifies industrial safety incidents and provides severity levels with recommended actions using the Ollama API.

## Features

- **AI-Powered Analysis**: Uses Ollama's GPT-OSS model for intelligent incident analysis
- **RAG System**: Retrieves similar past incidents from the industrial safety database
- **Severity Classification**: Automatically classifies incidents as HIGH, MEDIUM, or LOW severity
- **Recommended Actions**: Provides specific, actionable recommendations based on severity
- **Interactive Chat Interface**: User-friendly Streamlit interface for easy interaction
- **Similar Incident Search**: Finds and displays similar past incidents for context

## Dataset

The application uses the `Industrial_safety_and_health_database_with_accidents_description.csv` dataset containing:
- 426 industrial safety incidents
- Multiple industry sectors (Mining, Metals, etc.)
- Detailed incident descriptions
- Severity levels and risk categories
- Geographic and temporal information

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure the dataset is in the correct location:**
   ```
   /Users/vignesh/Documents/GitHub/Generative AI/Datasets/Industrial_safety_and_health_database_with_accidents_description.csv
   ```

## Usage

1. **Run the Streamlit application:**
   ```bash
   streamlit run safety_chatbot.py
   ```

2. **Initialize the system:**
   - Click "🚀 Initialize System" in the sidebar
   - Wait for the AI model to load (this may take a few minutes on first run)

3. **Analyze incidents:**
   - Type a description of a safety incident in the chat input
   - The system will:
     - Classify the severity level
     - Find similar past incidents
     - Generate AI-powered analysis
     - Provide recommended actions

## Example Usage

**Input:** "Worker slipped on wet floor and injured their back while carrying heavy equipment"

**Output:**
- **Severity:** MEDIUM
- **AI Analysis:** Detailed analysis of the incident
- **Similar Incidents:** Past incidents with similar characteristics
- **Recommended Actions:** Specific steps to address the incident

## API Configuration

The application uses the Ollama API with the following configuration:
- **Model:** gpt-oss:120b
- **API Key:** 75abea0b9b8d4d329432a2fbb6fcf1c8.dr3IdxNsGm0IMSgpN4pKIdDt
- **Endpoint:** https://ollama.com/api/chat

## Severity Classification

The system classifies incidents into three severity levels:

### HIGH Severity
- Fatalities, amputations, major injuries
- Chemical exposures, explosions, fires
- Structural collapses, equipment failures

### MEDIUM Severity
- Injuries requiring medical attention
- Cuts, fractures, burns
- Slips, trips, falls

### LOW Severity
- Minor injuries, first aid cases
- Near misses, property damage
- Procedural violations

## Technical Architecture

- **Frontend:** Streamlit for interactive web interface
- **RAG System:** Sentence Transformers for semantic search
- **AI Model:** Ollama's GPT-OSS for response generation
- **Data Processing:** Pandas for dataset manipulation
- **Similarity Search:** Cosine similarity for finding related incidents

## File Structure

```
├── safety_chatbot.py          # Main Streamlit application
├── requirements.txt           # Python dependencies
├── README_safety_chatbot.md   # This documentation
└── Datasets/
    └── Industrial_safety_and_health_database_with_accidents_description.csv
```

## Troubleshooting

1. **Model Loading Issues:** Ensure you have sufficient RAM (4GB+ recommended)
2. **API Errors:** Check your internet connection and API key validity
3. **Dataset Not Found:** Verify the dataset path in the code matches your file location

## Future Enhancements

- [ ] Add incident reporting functionality
- [ ] Implement trend analysis over time
- [ ] Add industry-specific recommendations
- [ ] Include regulatory compliance checking
- [ ] Add data visualization for incident patterns

To run the file
Streamlit run safety_chatbot.py
