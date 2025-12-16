"""
LLM Evaluation Runner
Location: Create this file in your project root directory

Usage:
    python run_llm_evaluation.py --test-ids HS_001 HS_002 VI_001
    python run_llm_evaluation.py --difficulty Easy
    python run_llm_evaluation.py --intent HOTEL_SEARCH
    python run_llm_evaluation.py --all  # Run all test cases
"""

import sys
import argparse
import pandas as pd
from pathlib import Path
import time
from typing import List, Dict
import json

# Add pipeline to path
sys.path.append('.')

from pipeline.graph_rag_pipeline import GraphRAGPipeline
from pipeline.llm_test_cases import (
    LLM_TEST_CASES,
    get_test_case_by_id,
    get_test_cases_by_difficulty,
    get_test_cases_by_intent,
    get_all_test_ids,
    QUANTITATIVE_METRICS,
    QUALITATIVE_METRICS
)
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, EMBEDDING_MODEL, GROQ_API_KEY

# ==============================================================================
# INITIALIZATION
# ==============================================================================

def initialize_pipeline():
    """Initialize the Graph-RAG pipeline"""
    print("🔧 Initializing Graph-RAG Pipeline...")
    
    pipeline = GraphRAGPipeline(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        embedding_model_name=EMBEDDING_MODEL,
        groq_api_key=GROQ_API_KEY
    )
    
    print("   Building visa search index...")
    pipeline.initialize_visa_search(verbose=False)
    
    print("✅ Pipeline initialized!\n")
    return pipeline

# ==============================================================================
# EVALUATION FUNCTIONS
# ==============================================================================

def run_test_case(pipeline, test_case: Dict, models: List[str]):
    """
    Run a single test case against multiple models
    
    Returns:
        Dict with results for each model
    """
    test_id = test_case["id"]
    query = test_case["query"]
    
    print(f"\n{'='*80}")
    print(f"Test Case: {test_id} - {test_case['difficulty']}")
    print(f"Query: {query}")
    print(f"{'='*80}")
    
    results = {}
    
    for model_key in models:
        print(f"\n🤖 Running with {model_key}...")
        
        try:
            # Run the query
            response = pipeline.process_query(
                user_query=query,
                use_baseline=True,
                use_embeddings=True,
                semantic_model='sbert',
                llm_model=model_key,
                max_results=10,
                verbose=False
            )
            
            # Extract quantitative metrics
            quantitative = {
                "response_time": response['llm_response']['response_time'],
                "token_usage": response['llm_response'].get('tokens_used', 0),
                "cost": calculate_cost(
                    response['llm_response'].get('tokens_used', 0),
                    model_key
                ),
                "context_adherence": calculate_context_adherence(
                    response['llm_response']['answer'],
                    response['combined_results']
                )
            }
            
            results[model_key] = {
                "test_id": test_id,
                "model": model_key,
                "query": query,
                "answer": response['llm_response']['answer'],
                "success": response['llm_response']['success'],
                "quantitative": quantitative,
                "intent_detected": response['intent'],
                "entities_extracted": response['entities'],
                "results_count": len(response['combined_results']),
                "retrieval_metadata": response['retrieval_metadata']
            }
            
            print(f"   ✅ Response time: {quantitative['response_time']:.2f}s")
            print(f"   ✅ Tokens used: {quantitative['token_usage']}")
            print(f"   ✅ Context adherence: {quantitative['context_adherence']:.1f}%")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            results[model_key] = {
                "test_id": test_id,
                "model": model_key,
                "query": query,
                "error": str(e),
                "success": False
            }
    
    return results

def calculate_cost(tokens: int, model_key: str) -> float:
    """Calculate estimated cost based on token usage"""
    # Groq pricing (as of implementation)
    pricing = {
        "llama-3.1-8b": 0.00008,  # per 1K tokens
        "llama-3.3-70b": 0.00079,
        "qwen-32b": 0.00059
    }
    
    price_per_1k = pricing.get(model_key, 0.0001)
    return (tokens / 1000) * price_per_1k

def calculate_context_adherence(answer: str, context_results: List[Dict]) -> float:
    """
    Estimate what % of the answer comes from provided context
    Simple heuristic: check if key facts from answer appear in context
    """
    if not context_results:
        return 0.0
    
    # Extract key entities from context
    context_text = ""
    for result in context_results:
        context_text += " ".join(str(v) for v in result.values())
    
    context_text = context_text.lower()
    answer_lower = answer.lower()
    
    # Count how many answer sentences have support in context
    sentences = [s.strip() for s in answer.split('.') if len(s.strip()) > 10]
    if not sentences:
        return 100.0
    
    supported = 0
    for sentence in sentences:
        # Check if key nouns from sentence appear in context
        words = sentence.split()
        if any(word in context_text for word in words if len(word) > 4):
            supported += 1
    
    return (supported / len(sentences)) * 100

# ==============================================================================
# QUALITATIVE EVALUATION (MANUAL)
# ==============================================================================

def collect_qualitative_scores(results: Dict, test_case: Dict) -> Dict:
    """
    Display results and collect manual qualitative scores
    Returns dict of {model: {metric: score}}
    """
    print(f"\n{'='*80}")
    print(f"MANUAL EVALUATION - Test Case: {test_case['id']}")
    print(f"Query: {test_case['query']}")
    print(f"{'='*80}\n")
    
    qualitative_scores = {}
    
    for model_key, result in results.items():
        if not result.get('success', False):
            continue
            
        print(f"\n{'-'*80}")
        print(f"MODEL: {model_key}")
        print(f"{'-'*80}")
        print(f"ANSWER:\n{result['answer']}\n")
        print(f"Expected to mention: {test_case['evaluation_criteria']['must_mention']}")
        print(f"Correctness check: {test_case['evaluation_criteria']['correctness_check']}\n")
        
        scores = {}
        
        print(f"Rate this answer on a scale of 1-5:")
        for metric, info in QUALITATIVE_METRICS.items():
            while True:
                try:
                    score = input(f"  {metric.capitalize()} ({info['description']}): ")
                    score = int(score)
                    if 1 <= score <= 5:
                        scores[metric] = score
                        break
                    else:
                        print("    Please enter a number between 1 and 5")
                except ValueError:
                    print("    Please enter a valid number")
        
        qualitative_scores[model_key] = scores
    
    return qualitative_scores

# ==============================================================================
# BATCH EVALUATION (Automated)
# ==============================================================================

def run_batch_evaluation(pipeline, test_cases: List[Dict], models: List[str]):
    """Run multiple test cases and collect all results"""
    
    all_results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'#'*80}")
        print(f"RUNNING TEST {i}/{len(test_cases)}")
        print(f"{'#'*80}")
        
        # Run the test case
        test_results = run_test_case(pipeline, test_case, models)
        
        # Store results
        for model_key, result in test_results.items():
            if result.get('success', False):
                all_results.append({
                    'test_id': test_case['id'],
                    'query': test_case['query'],
                    'difficulty': test_case['difficulty'],
                    'intent': test_case['expected_intent'],
                    'model': model_key,
                    'answer': result['answer'],
                    'response_time': result['quantitative']['response_time'],
                    'tokens': result['quantitative']['token_usage'],
                    'cost': result['quantitative']['cost'],
                    'context_adherence': result['quantitative']['context_adherence'],
                    'results_retrieved': result['results_count']
                })
    
    return all_results

# ==============================================================================
# RESULTS EXPORT
# ==============================================================================

def export_results(results: List[Dict], output_file: str = "llm_evaluation_results.csv"):
    """Export results to CSV for analysis"""
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\n✅ Results exported to {output_file}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        print(f"\n{model}:")
        print(f"  Average Response Time: {model_df['response_time'].mean():.2f}s")
        print(f"  Average Tokens: {model_df['tokens'].mean():.0f}")
        print(f"  Total Cost: ${model_df['cost'].sum():.4f}")
        print(f"  Average Context Adherence: {model_df['context_adherence'].mean():.1f}%")

# ==============================================================================
# MAIN CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run LLM evaluation tests")
    parser.add_argument('--test-ids', nargs='+', help='Specific test IDs to run')
    parser.add_argument('--difficulty', choices=['Easy', 'Medium', 'Hard'], help='Run tests of specific difficulty')
    parser.add_argument('--intent', help='Run tests for specific intent (e.g., HOTEL_SEARCH)')
    parser.add_argument('--all', action='store_true', help='Run all test cases')
    parser.add_argument('--models', nargs='+', default=['llama-3.1-8b', 'llama-3.3-70b', 'qwen-32b'],
                       help='Models to test')
    parser.add_argument('--output', default='llm_evaluation_results.csv', help='Output CSV file')
    parser.add_argument('--manual', action='store_true', help='Collect manual qualitative scores')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = initialize_pipeline()
    
    # Select test cases
    if args.test_ids:
        test_cases = [get_test_case_by_id(tid) for tid in args.test_ids]
        test_cases = [tc for tc in test_cases if tc is not None]
    elif args.difficulty:
        test_cases = get_test_cases_by_difficulty(args.difficulty)
    elif args.intent:
        test_cases = get_test_cases_by_intent(args.intent)
    elif args.all:
        test_cases = LLM_TEST_CASES
    else:
        # Default: run a sample of 5 test cases
        test_cases = LLM_TEST_CASES[:5]
    
    print(f"\n📋 Running {len(test_cases)} test cases with {len(args.models)} models")
    print(f"Models: {', '.join(args.models)}\n")
    
    # Run batch evaluation
    results = run_batch_evaluation(pipeline, test_cases, args.models)
    
    # Export results
    export_results(results, args.output)
    
    print(f"\n{'='*80}")
    print("✅ EVALUATION COMPLETE!")
    print(f"{'='*80}")
    print(f"\nNext steps:")
    print(f"1. Open {args.output} to view quantitative results")
    print(f"2. Manually rate answers for qualitative evaluation")
    print(f"3. Use the data for your project report")

if __name__ == "__main__":
    main()