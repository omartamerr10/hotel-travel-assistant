"""
Main Graph-RAG Pipeline
Uses your actual LLMClient and PromptBuilder
"""

from typing import Dict, List, Any
import time
from .components import (
    Intent,
    EntityExtractor,
    HotelIntentClassifier,
    QueryMapper,
    QueryExecutor,
    EnhancedSemanticSearch,
    ResultCombiner,
    PromptBuilder,
    LLMClient
)


class GraphRAGPipeline:
    """
    Complete Graph-RAG pipeline that orchestrates all components.
    """
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, 
                 embedding_model_name: str, groq_api_key: str):
        """
        Initialize the Graph-RAG pipeline with all components
        """
        print("🔧 Initializing Graph-RAG Pipeline...")
        
        # Initialize Neo4j driver
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password)
        )
        
        # Initialize components
        print("   - Initializing Entity Extractor...")
        self.entity_extractor = EntityExtractor(neo4j_driver=self.driver)
        
        print("   - Initializing Intent Classifier...")
        self.intent_classifier = HotelIntentClassifier(neo4j_driver=self.driver)
        
        print("   - Initializing Query Mapper...")
        self.query_mapper = QueryMapper()
        
        print("   - Initializing Query Executor...")
        self.query_executor = QueryExecutor(neo4j_uri, neo4j_user, neo4j_password)
        
        print("   - Initializing Semantic Search...")
        self.semantic_searcher = EnhancedSemanticSearch(
            neo4j_uri, neo4j_user, neo4j_password
        )
        
        print("   - Initializing LLM Client...")
        self.llm_client = LLMClient(groq_api_key=groq_api_key)
        
        self.embedding_model_name = embedding_model_name
        
        print("✅ Pipeline initialized successfully!")
    
    def process_query(self, 
                     user_query: str,
                     use_baseline: bool = True,
                     use_embeddings: bool = True,
                     llm_model: str = 'llama-3.1-8b',
                     max_results: int = 10,
                     verbose: bool = False) -> Dict[str, Any]:
        """
        Process a user query through the complete Graph-RAG pipeline.
        """
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Processing Query: {user_query}")
            print(f"{'='*80}")
        
        start_time = time.time()
        
        # STEP 1: Intent Classification
        if verbose:
            print("\n📍 STEP 1: Classifying Intent...")
        
        intent, confidence, debug_info = self.intent_classifier.classify(user_query)
        intent_str = intent.value if hasattr(intent, 'value') else str(intent)
        
        if verbose:
            print(f"   Detected Intent: {intent_str} (confidence: {confidence:.2f})")
        
        # STEP 2: Entity Extraction
        if verbose:
            print("\n📍 STEP 2: Extracting Entities...")
        
        entities = self.entity_extractor.extract_entities(user_query)
        
        if verbose:
            print(f"   Extracted Entities: {entities}")
        
        # STEP 3a: Baseline Retrieval (Cypher Queries)
        cypher_results = []
        cypher_queries = []
        
        if use_baseline:
            if verbose:
                print("\n📍 STEP 3a: Baseline Retrieval (Cypher)...")
            
            try:
                params = self.query_mapper.map_entities_to_parameters(
                    intent_str, entities
                )
                
                cypher_results = self.query_executor.execute_query(
                    intent_str, params
                )
                
                cypher_queries.append({
                    "query": self.query_executor._load_query_templates().get(intent_str, ""),
                    "description": f"Baseline query for {intent_str}",
                    "params": params
                })
                
                if verbose:
                    print(f"   Retrieved {len(cypher_results)} results from Cypher")
                    
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  Baseline retrieval failed: {str(e)}")
                cypher_results = []
        
        # STEP 3b: Embedding-based Retrieval
        semantic_results = []

        if use_embeddings:
            if verbose:
                print("\n📍 STEP 3b: Embedding-based Retrieval...")
            
            try:
                # Detect if this is a visa query
                if 'VISA' in intent_str:
                    if verbose:
                        print("   Using visa semantic search...")
                    
                    semantic_results = self.semantic_searcher.search_visa_semantic(
                        query=user_query,
                        model='sbert',
                        top_k=max_results
                    )
                else:
                    if verbose:
                        print("   Using hotel semantic search...")
                    
                    city_filter = entities.get('city', [None])[0] if entities.get('city') else None
                    country_filter = entities.get('country', [None])[0] if entities.get('country') else None
                    
                    semantic_results = self.semantic_searcher.search_hotels_semantic(
                        query=user_query,
                        model='sbert',
                        top_k=max_results,
                        city_filter=city_filter,
                        country_filter=country_filter
                    )
            except Exception as e:
                print("❌ Error during semantic retrieval:", e)
                semantic_results = []
                
        # STEP 4: Combine Results
        if verbose:
            print("\n📍 STEP 4: Combining Results...")

        combined_results, metadata = ResultCombiner.combine_results(
            cypher_results=cypher_results,
            semantic_results=semantic_results,
            intent=intent_str,
            max_results=max_results
        )

        # Add query type to metadata
        metadata['query_type'] = 'visa' if 'VISA' in intent_str else 'hotel'

        retrieval_time = time.time() - start_time
        metadata['retrieval_time'] = retrieval_time
        
        # STEP 5: Build Prompt
        if verbose:
            print("\n📍 STEP 5: Building Prompt...")

        # Detect query type for prompt building
        query_type = 'visa' if 'VISA' in intent_str else 'hotel'

        prompt = PromptBuilder.build_prompt(
            user_query=user_query,
            combined_results=combined_results,
            metadata=metadata,
            query_type=query_type  # ✅ Dynamic based on intent!
        )

        if verbose:
            print(f"   Using query type: {query_type}")
        
        # STEP 6: Generate LLM Response
        if verbose:
            print(f"\n📍 STEP 6: Generating answer with {llm_model}...")
        
        llm_response = self.llm_client.generate_response(
            prompt=prompt,
            model_key=llm_model,
            temperature=0.3
        )
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\n✅ Query processed in {total_time:.2f}s")
            print(f"{'='*80}\n")
        
        # Return complete results
        return {
            'user_query': user_query,
            'intent': intent_str,
            'entities': entities,
            'cypher_queries': cypher_queries,
            'combined_results': combined_results,
            'retrieval_metadata': metadata,
            'llm_response': llm_response,
            'total_pipeline_time': total_time
        }
    

    def initialize_visa_search(self, verbose=False):
        """Build FAISS index for visa requirements"""
        if verbose:
            print("Building FAISS index for visa requirements...")
        try:
            self.semantic_searcher.build_visa_faiss_index(model='both')
            if verbose:
                print("✓ FAISS index built successfully!")
        except Exception as e:
            if verbose:
                print(f"⚠️ Warning: Could not build visa index: {e}")

    def close(self):
        """Clean up resources"""
        if hasattr(self, 'driver'):
            self.driver.close()
        if hasattr(self, 'query_executor'):
            self.query_executor.close()
        if hasattr(self, 'semantic_searcher'):
            self.semantic_searcher.close()