"""
Hotel Booking Graph-RAG Assistant - Streamlit UI
Milestone 3 - Step 4: Build a UI

This Streamlit app demonstrates the Graph-RAG system with:
- Multiple retrieval strategies (Baseline, Embeddings, Hybrid)
- Multiple LLM model selection
- Visualization of KG-retrieved context
- Display of executed Cypher queries
- Graph visualization snippets
- Final LLM answers
"""

import streamlit as st
import json
from typing import Dict, List, Any
import time
import sys
sys.path.append('.')

try:
    from graph_visualization import display_graph_tab
except ImportError:
    display_graph_tab = None

# Import your pipeline components
# You'll need to adjust these imports based on your actual implementation
# from your_module import GraphRAGPipeline, Neo4jConnector, etc.

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Hotel Booking Assistant",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS FOR BETTER STYLING
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
    .cypher-query {
        background-color: #263238;
        color: #aed581;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        overflow-x: auto;
    }
    .answer-box {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #000000;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'query_history' not in st.session_state:
    st.session_state.query_history = []

if 'current_results' not in st.session_state:
    st.session_state.current_results = None

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Task Selection
    st.markdown("### 📋 Task Selection")
    task = st.selectbox(
        "Select Task",
        ["Question Answering", "Hotel Recommendation", "Booking Assistant"],
        help="Choose the type of assistance you need"
    )
    
    st.markdown("---")
    
    # ========================================================================
    # NEW: MODE SELECTION  (Benchmark mode is added to help compare the different models )
    # ========================================================================
    st.markdown("### 🛠️ App Mode")
    app_mode = st.radio(
        "Select Mode:",
        ["Standard Search", "Model Benchmark (Comparison)"],
        help="Standard: One query, one model. Benchmark: Compare 3 models side-by-side."
    )
    
    st.markdown("---")

    # Initialize variables to avoid errors if Benchmark mode is selected
    # (Defaults for benchmark mode)
    use_baseline = True
    use_embeddings = True
    semantic_model = 'sbert'
    llm_model = 'llama-3.1-8b' 

    # ONLY SHOW THESE CONTROLS IF IN STANDARD MODE
    if app_mode == "Standard Search":
        
        # 1. Retrieval Strategy
        st.markdown("### 🔍 Retrieval Strategy")
        retrieval_method = st.radio(
            "Choose retrieval method:",
            ["Baseline (Cypher Only)", "Embeddings Only", "Hybrid (Both)"]
        )
        
        use_baseline = retrieval_method in ["Baseline (Cypher Only)", "Hybrid (Both)"]
        use_embeddings = retrieval_method in ["Embeddings Only", "Hybrid (Both)"]
        
        st.markdown("---")

        # 2. Embedding Model (Only if embeddings used)
        if use_embeddings:
            st.markdown("### 🧠 Embedding Model")
            embedding_choice = st.radio(
                "Select Model:",
                ["SBERT (Fast)", "BGE (High Quality)"]
            )
            semantic_model = "sbert" if "SBERT" in embedding_choice else "bge"
        else:
            semantic_model = None 
            
        st.markdown("---")
        
        # 3. LLM Selection
        st.markdown("### 🤖 LLM Model")
        llm_model = st.selectbox(
            "Select LLM:",
            ["llama-3.1-8b", "llama-3.3-70b", "qwen-32b"], 
            format_func=lambda x: {
                "llama-3.1-8b": "Llama 3.1 8B (Fast)",
                "llama-3.3-70b": "Llama 3.3 70B (Smart)",
                "qwen-32b": "Qwen 3 32B (Balanced)" 
            }.get(x, x)
        )

    else:
        # BENCHMARK MODE INFO
        st.info("⚔️ **Benchmark Mode Active**\n\nRetrieval: Hybrid (SBERT)\nModels: Llama 3.1, Llama 3.3, Qwen 3")


    # Advanced Options
    with st.expander("🔧 Advanced Options"):
        max_results = st.slider(
            "Max Results to Retrieve",
            min_value=5,
            max_value=20,
            value=10,
            help="Maximum number of results to retrieve from the knowledge graph"
        )
        
        show_debug = st.checkbox(
            "Show Debug Information",
            value=False,
            help="Display additional debugging information"
        )
        
        verbose_mode = st.checkbox(
            "Verbose Mode",
            value=False,
            help="Show detailed processing steps"
        )
    
    st.markdown("---")
    
    # Info Section
    st.markdown("### ℹ️ About")
    st.info(
        "This is a Graph-RAG powered hotel booking assistant that combines "
        "knowledge graph retrieval with LLM-based generation for accurate, "
        "grounded responses."
    )
    
    # Statistics
    if st.session_state.query_history:
        st.markdown("### 📊 Statistics")
        st.metric("Total Queries", len(st.session_state.query_history))

# ============================================================================
# MAIN HEADER
# ============================================================================

st.markdown('<h1 class="main-header">🏨 Hotel Booking Assistant</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Powered by Graph-RAG: Neo4j Knowledge Graph + Large Language Models</p>',
    unsafe_allow_html=True
)

# ============================================================================
# MAIN QUERY INPUT
# ============================================================================

st.markdown("## 💬 Ask Your Question")

# Query input with examples
query_input = st.text_area(
    "Enter your query:",
    height=100,
    placeholder="E.g., 'Recommend a hotel in Paris for a family with children' or 'Find 5-star hotels in Dubai with spa facilities'",
    help="Ask about hotel recommendations, search for specific hotels, or inquire about amenities and services"
)

# Example queries
st.markdown("**💡 Example Queries:**")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🌍 Hotels in Cairo"):
        query_input = "Find hotels in Cairo with good ratings"

with col2:
    if st.button("👨‍👩‍👧 Family Hotels"):
        query_input = "Recommend family-friendly hotels with pools"

with col3:
    if st.button("💼 Business Hotels"):
        query_input = "Find business hotels with conference facilities"

# Submit button
submit_button = st.button("🔍 Search", type="primary", use_container_width=True)

# ============================================================================
# REAL PIPELINE INITIALIZATION
# ============================================================================

import sys
sys.path.append('.')

from pipeline.graph_rag_pipeline import GraphRAGPipeline
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, EMBEDDING_MODEL

@st.cache_resource
def initialize_pipeline():
    """Initialize and cache the Graph-RAG pipeline"""
    try:
        from config.settings import GROQ_API_KEY
        
        pipeline = GraphRAGPipeline(
            neo4j_uri=NEO4J_URI,
            neo4j_user=NEO4J_USER,
            neo4j_password=NEO4J_PASSWORD,
            embedding_model_name=EMBEDDING_MODEL,
            groq_api_key=GROQ_API_KEY
        )
        
        # Initialize visa search capabilities
        with st.spinner("Building visa search index..."):
            pipeline.initialize_visa_search(verbose=False)
        
        return pipeline
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {str(e)}")
        return None

pipeline = initialize_pipeline()

# ============================================================================
# QUERY PROCESSING
# ============================================================================

if submit_button and query_input and pipeline:
    
    # ------------------------------------------------------------------------
    # MODE 1: STANDARD SEARCH (Your original logic)
    # ------------------------------------------------------------------------
    if app_mode == "Standard Search":
        with st.spinner("🔄 Processing your query..."):
            try:
                results = pipeline.process_query(
                    user_query=query_input,
                    use_baseline=use_baseline,
                    use_embeddings=use_embeddings,
                    semantic_model=semantic_model,
                    llm_model=llm_model,
                    max_results=max_results,
                    verbose=verbose_mode
                )
                
                st.session_state.current_results = results
                st.session_state.query_history.append({
                    "query": query_input,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "retrieval_method": retrieval_method,
                    "llm_model": llm_model
                })
                
                st.success("✅ Query processed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
                if show_debug:
                    st.exception(e)

    # ------------------------------------------------------------------------
    # MODE 2: BENCHMARK MODE (The new comparison logic)
    # ------------------------------------------------------------------------
    elif app_mode == "Model Benchmark (Comparison)":
        
        st.markdown(f"### ⚔️ Model Battle: {query_input}")
        
        with st.spinner("Running Benchmark... (This checks aLlama 3.1, 3.3, and Qwen 3)"):
            try:
                # 1. Run Comparison
                benchmark_data = pipeline.compare_models(query_input)
                results = benchmark_data['model_results']
                
                st.success("✅ Benchmark Complete!")
                st.markdown(f"**Context Retrieved:** {benchmark_data['context_results']} items | **Intent:** `{benchmark_data['intent']}`")
                
                # 2. Display Answers Side-by-Side
                st.markdown("---")
                cols = st.columns(3)
                model_keys = list(results.keys())
                
                # List for DataFrame
                eval_data = []
                
                for i, col in enumerate(cols):
                    m_key = model_keys[i]
                    data = results[m_key]
                    
                    # Prepare quantitative data for table
                    eval_data.append({
                        "Model": m_key,
                        "Time (s)": round(data['time'], 2),
                        "Tokens": data['tokens'],
                        "Cost ($)": float(f"{data['cost']:.5f}"),
                        "Context Adherence (%)": round(data['accuracy'], 1), 
                    })
                    
                    with col:
                        st.subheader(m_key)
                        # Display the new accuracy metric in the card
                        st.caption(f"⏱️ {data['time']:.2f}s | 🎯 Acc: {data['accuracy']:.1f}%")
                        st.info(data['answer'])
                
               # 3. Evaluation Table
                st.markdown("### 📊 Evaluation Metrics")
                
                import pandas as pd
                df = pd.DataFrame(eval_data)
                
                # Add columns for Qualitative (Human) Scoring
                df["Human Relevance (1-5)"] = 0   # Renamed for clarity
                df["Human Naturalness (1-5)"] = 0 # Renamed for clarity
                
                st.markdown("Automated metrics are filled. Please rate Relevance & Naturalness:")
                edited_df = st.data_editor(
                    df, 
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Context Adherence (%)": st.column_config.ProgressColumn(
                            "KG Accuracy %", 
                            help="Percentage of retrieved KG entities mentioned in the answer",
                            format="%.1f%%",
                            min_value=0,
                            max_value=100,
                        ),
                        "Human Relevance (1-5)": st.column_config.NumberColumn(min_value=1, max_value=5, step=1),
                        "Human Naturalness (1-5)": st.column_config.NumberColumn(min_value=1, max_value=5, step=1),
                    }
                )
                     
                # 4. Calculate Winner Button
                if st.button("🏆 Calculate Final Score"):
                    edited_df["Total Score"] = (
                        edited_df["Accuracy (1-5)"] + 
                        edited_df["Relevance (1-5)"] + 
                        edited_df["Naturalness (1-5)"]
                    )
                    
                    # Find winner
                    winner = edited_df.sort_values(by="Total Score", ascending=False).iloc[0]
                    st.success(f"🎉 The Winner is **{winner['Model']}** with {winner['Total Score']} points!")
                    
                    # Display Final Sorted Table
                    st.dataframe(
                        edited_df.sort_values(by="Total Score", ascending=False), 
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"❌ Benchmark failed: {str(e)}")
                if show_debug:
                    st.exception(e)

elif submit_button and not pipeline:
    st.error("Pipeline not initialized. Check your configuration.")
    
# ============================================================================
# RESULTS DISPLAY
# ============================================================================

if st.session_state.current_results:
    results = st.session_state.current_results
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Final Answer",
        "🗂️ Retrieved Context",
        "🔧 Cypher Queries",
        "📊 Graph Visualization",
        "📈 Metrics"
    ])
    
    # ========================================================================
    # TAB 1: FINAL LLM ANSWER
    # ========================================================================
    with tab1:
        st.markdown("## 🎯 Assistant's Answer")
        
        if results['llm_response']['success']:
            st.info(results["llm_response"]["answer"], icon="🤖")
            
            # Show model info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model Used", results['llm_response']['model'])
            with col2:
                st.metric("Response Time", f"{results['llm_response']['response_time']:.2f}s")
            with col3:
                st.metric("Tokens Used", results['llm_response'].get('tokens_used', 'N/A'))
        else:
            st.error("Failed to generate response from LLM")
            st.write(f"**Error Details:** {results['llm_response'].get('error', 'Unknown Error')}")
    
    # ========================================================================
    # TAB 2: RETRIEVED CONTEXT
    # ========================================================================
    with tab2:
        st.markdown("## 🗂️ Knowledge Graph Retrieved Context")
        st.markdown("This is the raw information retrieved from the knowledge graph before LLM processing.")
        
        # Show retrieval metadata
        metadata = results.get('retrieval_metadata', {})
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Results", metadata.get('total_results', 0))
        with col2:
            st.metric("Cypher Only", metadata.get('results_cypher_only', 0))
        with col3:
            st.metric("Semantic Only", metadata.get('results_semantic_only', 0))
        with col4:
            st.metric("In Both", metadata.get('results_in_both', 0))
        
        st.markdown("---")
        
        # Show extracted entities
        if 'entities' in results:
            st.markdown("### 🏷️ Extracted Entities")
            st.json(results['entities'])
        
        st.markdown("---")
        
        # Show retrieved results
        st.markdown("### 📋 Retrieved Hotel Data")
        
        if results.get('combined_results'):
            for idx, hotel in enumerate(results['combined_results'], 1):
                # Safely handle different data types
                if isinstance(hotel, dict):
                    hotel_name = hotel.get('hotel_name', f'Hotel {idx}')
                    source = hotel.get('source', 'unknown')
                    
                    with st.expander(f"🏨 {hotel_name} - Source: {source.upper()}"):
                        st.json(hotel)
                else:
                    with st.expander(f"🏨 Result {idx}"):
                        st.write(hotel)
        else:
            st.info("No results retrieved from the knowledge graph.")
    
    # ========================================================================
    # TAB 3: CYPHER QUERIES
    # ========================================================================
    with tab3:
        st.markdown("## 🔧 Executed Cypher Queries")
        st.markdown("These are the actual Cypher queries executed against the Neo4j database.")
        
        if 'cypher_queries' in results and results['cypher_queries']:
            for idx, query_info in enumerate(results['cypher_queries'], 1):
                st.markdown(f"### Query {idx}: {query_info.get('description', 'N/A')}")
                st.markdown(f'<div class="cypher-query">{query_info.get("query", "")}</div>', 
                          unsafe_allow_html=True)
                st.markdown("---")
        else:
            st.info("No Cypher queries were executed (using embeddings-only mode)")
    
    # ========================================================================
    # TAB 4: GRAPH VISUALIZATION
    # ========================================================================
    with tab4:
        st.markdown("## 📊 Graph Visualization")
        st.markdown("Visual representation of the retrieved connections from Neo4j.")
        
        if display_graph_tab:
            display_graph_tab(results)
        else:
            st.error("Graph visualization module (`graph_visualization.py`) not found.")
    
    # ========================================================================
    # TAB 5: METRICS AND COMPARISON
    # ========================================================================
    with tab5:
        st.markdown("## 📈 Performance Metrics")
        
        # Retrieval Performance
        st.markdown("### ⚡ Retrieval Performance")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Retrieval Time",
                f"{metadata.get('retrieval_time', 0):.3f}s"
            )
        with col2:
            st.metric(
                "LLM Response Time",
                f"{results['llm_response']['response_time']:.2f}s"
            )
        with col3:
            total_time = metadata.get('retrieval_time', 0) + results['llm_response']['response_time']
            st.metric(
                "Total Time",
                f"{total_time:.2f}s"
            )
        
        st.markdown("---")
        
        # Query Information
        st.markdown("### 🔍 Query Information")
        query_info_data = {
            "User Query": results.get('user_query', ''),
            "Detected Intent": results.get('intent', ''),
            "Retrieval Method": retrieval_method,
            "Embedding Model": semantic_model if use_embeddings else "N/A",
            "LLM Model": results['llm_response']['model'],
            "Baseline Used": "✅" if use_baseline else "❌",
            "Embeddings Used": "✅" if use_embeddings else "❌"
        }
        
        for key, value in query_info_data.items():
            st.write(f"**{key}:** {value}")

# ============================================================================
# QUERY HISTORY SIDEBAR
# ============================================================================

if st.session_state.query_history:
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📜 Query History")
        
        for idx, hist in enumerate(reversed(st.session_state.query_history[-5:]), 1):
            with st.expander(f"{hist['timestamp'][:10]} - Query {len(st.session_state.query_history) - idx + 1}"):
                st.write(f"**Query:** {hist['query'][:50]}...")
                st.write(f"**Method:** {hist['retrieval_method']}")
                st.write(f"**Model:** {hist['llm_model']}")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🏗️ Built with Streamlit | 🗄️ Powered by Neo4j | 🤖 Enhanced with LLMs</p>
    <p><small>Milestone 3 - Graph-RAG Travel Assistant | German University in Cairo</small></p>
</div>
""", unsafe_allow_html=True)
