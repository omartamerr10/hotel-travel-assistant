import networkx as nx
import plotly.graph_objects as go
from typing import Dict, Any
import streamlit as st

def create_graph_from_results(results: Dict[str, Any]) -> nx.Graph:
    """
    Create a NetworkX graph from pipeline results.
    Adapted for the actual data structure from components.py.
    """
    G = nx.Graph()
    
    # Vibrant colors for Dark Theme
    colors = {
        'hotel': '#FF6B6B',    # Coral Red
        'city': '#4ECDC4',     # Turquoise
        'country': '#FFE66D',  # Yellow
        'visa': '#F7FFF7'      # White/Honeydew
    }
    
    combined = results.get('combined_results', [])
    
    for item in combined:
        # --- SCENARIO 1: HOTEL RESULTS ---
        if 'hotel_name' in item:
            h_name = item['hotel_name']
            
            # Hotel Node
            G.add_node(h_name, 
                      node_type='hotel', 
                      size=20, 
                      color=colors['hotel'], 
                      title=f"🏨 {h_name}<br>⭐ {item.get('star_rating', 'N/A')}")
            
            # City Node
            if 'city' in item and item['city']:
                c_name = item['city']
                G.add_node(c_name, 
                          node_type='city', 
                          size=15, 
                          color=colors['city'], 
                          title=f"🏙️ {c_name}")
                G.add_edge(h_name, c_name, relation="LOCATED_IN")
                
                # Country Node (linked to City)
                if 'country' in item and item['country']:
                    cnt_name = item['country']
                    G.add_node(cnt_name, 
                              node_type='country', 
                              size=25, 
                              color=colors['country'], 
                              title=f"🌍 {cnt_name}")
                    G.add_edge(c_name, cnt_name, relation="IN_COUNTRY")

        # --- SCENARIO 2: VISA RESULTS ---
        elif 'from_country' in item:
            f_country = item['from_country']
            t_country = item['to_country']
            
            # Country Nodes
            G.add_node(f_country, node_type='country', size=25, color=colors['country'], title=f"🛫 {f_country}")
            G.add_node(t_country, node_type='country', size=25, color=colors['country'], title=f"🛬 {t_country}")
            
            # Visa Edge
            G.add_edge(f_country, t_country, 
                      relation="REQUIRES_VISA", 
                      title=f"🛂 Visa: {item.get('visa_type', 'Required')}")

    return G

def create_plotly_graph(G: nx.Graph) -> go.Figure:
    """Render the NetworkX graph using Plotly (Dark Theme Friendly)."""
    if G.number_of_nodes() == 0:
        return None

    # 1. Layout Calculation
    pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)

    # 2. Draw Edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888888'), # Grey edges
        hoverinfo='none',
        mode='lines')

    # 3. Draw Nodes
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    node_custom_data = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        data = G.nodes[node]
        node_color.append(data.get('color', '#888'))
        node_size.append(data.get('size', 15))
        node_custom_data.append(data.get('title', str(node)))
        
        # Only show labels for Cities and Countries directly to avoid clutter
        if data.get('node_type') != 'hotel':
            node_text.append(str(node))
        else:
            node_text.append('')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        textfont=dict(color='white'), # White text for dark mode
        hoverinfo='text',
        hovertext=node_custom_data,
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            line_width=2,
            line_color='#262730' # Dark border to match Streamlit bg
        )
    )

    # 4. Create Figure
    fig = go.Figure(data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0,l=0,r=0,t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)', # Transparent background
            paper_bgcolor='rgba(0,0,0,0)', # Transparent background
            height=500
        )
    )
    return fig

def display_graph_tab(results):
    """Main function to be called from Streamlit."""
    
    # Check if we have results
    if not results or not results.get('combined_results'):
        st.info("No data available to visualize.")
        return

    col_info, col_viz = st.columns([1, 3])
    
    G = create_graph_from_results(results)
    
    with col_info:
        st.markdown("### 🏷️ Legend")
        st.markdown("🔴 **Hotel**")
        st.markdown("🔵 **City**")
        st.markdown("🟡 **Country**")
        
        st.markdown("---")
        st.metric("Nodes", G.number_of_nodes())
        st.metric("Edges", G.number_of_edges())

    with col_viz:
        if G.number_of_nodes() > 0:
            fig = create_plotly_graph(G)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Could not generate graph from results.")