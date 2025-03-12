import streamlit as st
import networkx as nx
import plotly.graph_objects as go
from typing import Dict, List, Tuple
import os
from dotenv import load_dotenv
import google.generativeai as genai
import json
import numpy as np

# Load environment variables
load_dotenv()

# Configure Gemini API
if 'GOOGLE_API_KEY' in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("Please set up the GOOGLE_API_KEY in your Streamlit secrets.")
    st.stop()

model = genai.GenerativeModel('gemini-2.0-flash')

class CausalGraph:
    def __init__(self):
        self.G = nx.DiGraph()
        self.edge_explanations = {}

    def add_node(self, node: str) -> None:
        """Add a node to the graph."""
        if node not in self.G.nodes:
            self.G.add_node(node)

    def add_edge(self, source: str, target: str, explanation: str = "") -> None:
        """Add an edge to the graph with an explanation."""
        self.add_node(source)
        self.add_node(target)
        if not self.G.has_edge(source, target): # Prevent duplicate edges
            self.G.add_edge(source, target)
            self.edge_explanations[(source, target)] = explanation
        else:
            self.edge_explanations[(source, target)] = explanation # Update explanation if edge exists

    def remove_node(self, node: str) -> None:
        """Remove a node from the graph."""
        if node in self.G.nodes:
            self.G.remove_node(node)
            # Clean up edge explanations related to the removed node
            self.edge_explanations = {k: v for k, v in self.edge_explanations.items() if k[0] != node and k[1] != node}

    def remove_edge(self, source: str, target: str) -> None:
        """Remove an edge from the graph."""
        if self.G.has_edge(source, target):
            self.G.remove_edge(source, target)
            if (source, target) in self.edge_explanations:
                del self.edge_explanations[(source, target)]

    def update_edge_explanation(self, source: str, target: str, new_explanation: str) -> None:
        """Update the explanation for an existing edge."""
        if self.G.has_edge(source, target):
            self.edge_explanations[(source, target)] = new_explanation

    def get_edge_explanation(self, source: str, target: str) -> str:
        """Get the explanation for an edge."""
        return self.edge_explanations.get((source, target), "No explanation available")

    def clear_graph(self) -> None:
        """Clear all nodes and edges from the graph."""
        self.G.clear()
        self.edge_explanations = {}

    def process_prompt(self, prompt: str) -> str:
        """Process a natural language prompt to modify the graph."""
        try:
            system_prompt = """You are a causal reasoning expert specializing in complex causal relationships. Your task is to analyze the input and generate a JSON response that instructs a system to modify a causal graph.

Rules:
1. Your response must be ONLY a valid JSON object.
2. Do not include any other text or explanation outside the JSON.
3. When multiple relationships are described, include all of them in a single response.
4. Follow this JSON format:

    {
        "action": "add_multiple",
        "relationships": [
            {
                "type": "direct|mediator|confounder",
                "nodes": ["source_node", "target_node"],
                "explanation": "detailed reasoning for the causal relationship"
            },
            {
                "type": "direct|mediator|confounder",
                "nodes": ["another_source", "another_target"],
                "explanation": "detailed reasoning for this relationship"
            }
            // ... more relationships as needed
        ]
    }

Example:
Input: "Create a causal graph between a,b,c,d as causes and e as effect. a → e: Variable 'a' directly influences the outcome 'e'. b → e through mediation. c and d jointly affect e."

Response:
{
    "action": "add_multiple",
    "relationships": [
        {
            "type": "direct",
            "nodes": ["a", "e"],
            "explanation": "Variable 'a' directly influences the outcome 'e'"
        },
        {
            "type": "direct",
            "nodes": ["b", "e"],
            "explanation": "Variable 'b' influences outcome 'e'"
        },
        {
            "type": "direct",
            "nodes": ["c", "e"],
            "explanation": "Variable 'c' affects outcome 'e'"
        },
        {
            "type": "direct",
            "nodes": ["d", "e"],
            "explanation": "Variable 'd' influences outcome 'e'"
        }
    ]
}

Be precise in specifying relationships and provide explanations for each causal link."""

            # Generate response using Gemini
            response = model.generate_content(
                f"{system_prompt}\n\nInput: {prompt}\nResponse (JSON only):",
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    top_p=1,
                    top_k=5
                )
            )

            # Extract the response and parse it as JSON
            try:
                response_text = response.text.strip()
                response_text = response_text.replace('```json', '').replace('```', '').strip()
                response_json = json.loads(response_text)

                if response_json.get("action") == "add_multiple":
                    relationships = response_json.get("relationships", [])
                    results = []
                    
                    for rel in relationships:
                        rel_type = rel.get("type", "direct")
                        nodes = rel.get("nodes", [])
                        explanation = rel.get("explanation", "")
                        
                        if len(nodes) == 2:
                            self.add_edge(nodes[0], nodes[1], explanation)
                            results.append(f"Added {rel_type} relationship: {nodes[0]} → {nodes[1]}")
                        elif len(nodes) == 3 and rel_type == "mediator":
                            self.add_edge(nodes[0], nodes[1], f"First part of mediation: {explanation}")
                            self.add_edge(nodes[1], nodes[2], f"Second part of mediation: {explanation}")
                            results.append(f"Added mediator relationship: {nodes[0]} → {nodes[1]} → {nodes[2]}")
                    
                    return "\n".join(results)
                else:
                    # Handle other actions as before
                    action = response_json.get("action")
                    nodes = response_json.get("nodes", [])
                    explanation = response_json.get("explanation", "")
                    relationship_type = response_json.get("relationship_type", "direct")
                    query = response_json.get("query", "")

                    if action == "add_edge" and len(nodes) == 2:
                        self.add_edge(nodes[0], nodes[1], explanation)
                        return f"Added direct causal relationship: {nodes[0]} → {nodes[1]}"
                    elif action == "add_mediator" and len(nodes) == 3:
                        self.add_edge(nodes[0], nodes[1], f"First part of mediation: {explanation}")
                        self.add_edge(nodes[1], nodes[2], f"Second part of mediation: {explanation}")
                        return f"Added mediator relationship: {nodes[0]} → {nodes[1]} → {nodes[2]}"
                    elif action == "add_confounder" and len(nodes) == 3:
                        self.add_edge(nodes[0], nodes[1], f"Confounder effect on first variable: {explanation}")
                        self.add_edge(nodes[0], nodes[2], f"Confounder effect on second variable: {explanation}")
                        return f"Added confounder relationship: {nodes[0]} affects both {nodes[1]} and {nodes[2]}"
                    elif action == "add_node" and len(nodes) == 1:
                        self.add_node(nodes[0])
                        node_type = response_json.get("node_type", "variable")
                        return f"Added {node_type} node: {nodes[0]}"
                    elif action == "remove_node" and len(nodes) == 1:
                        self.remove_node(nodes[0])
                        return f"Removed node: {nodes[0]}"
                    elif action == "remove_edge" and len(nodes) == 2:
                        self.remove_edge(nodes[0], nodes[1])
                        return f"Removed edge between {nodes[0]} and {nodes[1]}"
                    elif action == "update_explanation" and len(nodes) == 2:
                        self.update_edge_explanation(nodes[0], nodes[1], explanation)
                        return f"Updated explanation for {nodes[0]} → {nodes[1]}"
                    elif action == "clear_graph":
                        self.clear_graph()
                        return "Graph cleared"
                    elif action == "explain_relationship" and len(nodes) >= 2:
                        if self.G.has_edge(nodes[0], nodes[1]):
                            edge_explanation = self.get_edge_explanation(nodes[0], nodes[1])
                            return f"Explanation for relationship between {nodes[0]} and {nodes[1]} ({relationship_type}): {edge_explanation}"
                        elif query:
                            explanation_response = model.generate_content(
                                f"Analyze the potential causal relationship between '{nodes[0]}' and '{nodes[1]}' in a marketing context, considering: \n"
                                f"1. Direct effects\n2. Potential mediators\n3. Possible confounders\n4. Direction of causality\n\n"
                                f"User query: {query}"
                            )
                            return f"Analysis: {explanation_response.text}"
                        else:
                            return "No direct relationship found. Please specify a query for relationship analysis."

                    return "Action completed successfully"
            except json.JSONDecodeError as e:
                return f"Error: Invalid JSON response from Gemini: {str(e)}\nResponse was: {response_text}"
            except Exception as e:
                return f"Error processing prompt: {str(e)}"
        except Exception as e:
            return f"Error generating response from Gemini: {str(e)}"

# Add this after the imports
if 'node_positions' not in st.session_state:
    st.session_state.node_positions = {}

def update_node_positions(selected_points, nodes):
    """Update node positions based on drag events."""
    if selected_points and 'points' in selected_points:
        for point in selected_points['points']:
            if 'pointIndex' in point:
                node = nodes[point['pointIndex']]
                st.session_state.node_positions[node] = [point['x'], point['y']]

def visualize_graph(G: nx.DiGraph, edge_explanations: Dict, node_positions: Dict) -> go.Figure:
    """Create a Plotly visualization of the causal graph."""
    if not G.nodes:
        return go.Figure(data=[], layout=go.Layout(title='Causal Graph (Empty)'))

    # Use spring layout with more space between nodes
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Update positions with manual adjustments
    for node, (x, y) in node_positions.items():
        if node in pos:
            pos[node] = np.array([x, y])

    # Create edges with arrows
    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        # Calculate arrow position (80% along the edge)
        arrow_x = x0 + 0.8 * (x1 - x0)
        arrow_y = y0 + 0.8 * (y1 - y0)
        
        # Calculate arrow angle
        angle = np.arctan2(y1 - y0, x1 - x0)
        
        # Create edge line
        edge_trace = go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        )
        
        # Create arrow head
        arrow_size = 0.03
        arrow_trace = go.Scatter(
            x=[
                arrow_x - arrow_size * np.cos(angle - np.pi/6),
                arrow_x,
                arrow_x - arrow_size * np.cos(angle + np.pi/6)
            ],
            y=[
                arrow_y - arrow_size * np.sin(angle - np.pi/6),
                arrow_y,
                arrow_y - arrow_size * np.sin(angle + np.pi/6)
            ],
            line=dict(width=1, color='#888'),
            fill='toself',
            fillcolor='#888',
            hoverinfo='none',
            mode='lines+markers',
            showlegend=False
        )
        
        edge_traces.extend([edge_trace, arrow_trace])

    # Create nodes
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        marker=dict(
            size=35,
            line=dict(width=2, color='black'),
            color='skyblue',
            symbol='circle',
            opacity=0.8
        ),
        textfont=dict(
            size=16,
            color='black',
            family='Arial Black'
        ),
        customdata=[[node] for node in G.nodes()]
    )

    # Create the figure with all traces
    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            title=dict(
                text='Causal Graph',
                font=dict(size=24, family='Arial Black')
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False, 
                range=[-2, 2],
                fixedrange=False,
                constrain='domain'
            ),
            yaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False, 
                range=[-2, 2],
                fixedrange=False,
                scaleanchor='x',
                scaleratio=1
            ),
            plot_bgcolor='white',
            dragmode='pan',
            modebar=dict(
                orientation='v',
                bgcolor='rgba(255, 255, 255, 0.7)',
                color='#333',
                activecolor='#FF4B4B'
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=True,
                    x=0.05,
                    y=1.15,
                    direction='right',
                    buttons=[
                        dict(
                            label='Pan',
                            method='relayout',
                            args=[{'dragmode': 'pan'}]
                        ),
                        dict(
                            label='Zoom',
                            method='relayout',
                            args=[{'dragmode': 'zoom'}]
                        ),
                        dict(
                            label='Reset',
                            method='relayout',
                            args=[{
                                'xaxis.range': [-2, 2],
                                'yaxis.range': [-2, 2]
                            }]
                        )
                    ]
                )
            ]
        )
    )

    # Add hover interactions
    fig.update_traces(
        hovertemplate="<b>%{text}</b><extra></extra>",
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Arial Black"
        )
    )

    return fig

# Initialize session state
if 'causal_graph' not in st.session_state:
    st.session_state.causal_graph = CausalGraph()

# Streamlit UI
st.title("Interactive Causal Graph")

# Input prompt
prompt = st.text_area("Enter your prompt to modify the causal graph (e.g., 'Add a causal link...', 'Remove node...', 'Explain relationship between...'):")
if st.button("Process Prompt"):
    if prompt:
        result = st.session_state.causal_graph.process_prompt(prompt)
        st.write("Processing result:", result)

# Create a single column layout
if st.button("Clear Graph"):
    st.session_state.causal_graph.clear_graph()
    st.session_state.node_positions = {}
    st.success("Graph cleared!")

# Visualize the graph
if st.session_state.causal_graph.G.number_of_nodes() > 0:
    fig = visualize_graph(
        st.session_state.causal_graph.G,
        st.session_state.causal_graph.edge_explanations,
        st.session_state.node_positions
    )
    
    # Add configuration for interactivity
    config = {
        'displayModeBar': True,
        'scrollZoom': True,
        'displaylogo': False,
        'modeBarButtonsToAdd': [
            'zoom',
            'pan',
            'zoomIn',
            'zoomOut',
            'resetScale',
            'toImage'
        ],
        'editable': True,
        'showAxisDragHandles': True
    }
    
    # Display the graph
    st.plotly_chart(fig, use_container_width=True, config=config)
    
    # Show edge explanations in a collapsible section
    with st.expander("View Edge Explanations"):
        for (source, target), explanation in st.session_state.causal_graph.edge_explanations.items():
            st.write(f"**{source} → {target}**: {explanation}")

    # Add instructions for graph interaction
    st.info("""
    Graph Interaction Guide:
    - Use the Pan/Zoom buttons to switch between modes
    - In Pan mode: Click and drag to move the graph
    - In Zoom mode: Click and drag to zoom into a specific area
    - Use mouse wheel to zoom in/out
    - Click 'Reset' to restore the original view
    """)
else:
    st.info("The causal graph is currently empty. Enter prompts to add nodes and edges.") 