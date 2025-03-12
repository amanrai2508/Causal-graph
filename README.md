# Prompt-based Causal Graph POC

This is a proof of concept for a prompt-based causal graph system that allows you to create and modify causal relationships using natural language prompts, powered by Anthropic's Claude API.

## Features

- Create nodes and edges through natural language prompts
- Visualize causal relationships in an interactive graph
- Get explanations for causal relationships
- Modify existing relationships using prompts
- Powered by Claude-3 for intelligent causal reasoning

## Setup

1. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your Anthropic API key:
```
ANTHROPIC_API_KEY=your_api_key_here
```

## Running the Application

Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage Examples

1. Adding a causal relationship:
   ```
   "Add a causal relationship between 'Exercise' and 'Health' because regular physical activity improves overall health outcomes"
   ```

2. Querying relationships:
   ```
   "Why does Exercise affect Health?"
   ```

3. Modifying relationships:
   ```
   "Update the relationship between Exercise and Health to include mental health benefits"
   ```

## Architecture

The system consists of three main components:

1. **Graph Management**: Uses NetworkX to maintain the causal graph structure
2. **Prompt Processing**: Utilizes Claude-3 to interpret natural language commands and reason about causality
3. **Visualization**: Uses Plotly to create interactive graph visualizations

## Limitations

- Currently uses a simple prompt processing system
- Graph layout might need optimization for larger graphs
- Limited to basic causal relationships
- Requires Claude API access 