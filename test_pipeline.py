"""
Test Script - Verify Pipeline Before Running Streamlit
Run this to make sure everything works before launching the UI

Usage: python test_pipeline.py
"""

import sys
import os

print("="*80)
print("🧪 PIPELINE TEST SCRIPT")
print("="*80)

# Test 1: Check directory structure
print("\n📁 TEST 1: Checking directory structure...")
required_dirs = ['pipeline', 'config']
required_files = [
    'pipeline/__init__.py',
    'config/__init__.py',
    'pipeline/components.py',
    'config/settings.py',
    '.env'
]

all_good = True
for directory in required_dirs:
    if os.path.exists(directory):
        print(f"   ✅ {directory}/ exists")
    else:
        print(f"   ❌ {directory}/ MISSING - create it!")
        all_good = False

for filepath in required_files:
    if os.path.exists(filepath):
        print(f"   ✅ {filepath} exists")
    else:
        print(f"   ❌ {filepath} MISSING - create it!")
        all_good = False

if not all_good:
    print("\n❌ Directory structure incomplete. Fix the issues above.")
    sys.exit(1)

# Test 2: Check environment variables
print("\n🔐 TEST 2: Checking environment variables...")
try:
    from dotenv import load_dotenv
    load_dotenv()
    
    neo4j_uri = os.getenv('NEO4J_URI')
    neo4j_user = os.getenv('NEO4J_USER')
    neo4j_password = os.getenv('NEO4J_PASSWORD')
    
    if neo4j_uri:
        print(f"   ✅ NEO4J_URI: {neo4j_uri}")
    else:
        print(f"   ❌ NEO4J_URI not set in .env")
        all_good = False
    
    if neo4j_user:
        print(f"   ✅ NEO4J_USER: {neo4j_user}")
    else:
        print(f"   ❌ NEO4J_USER not set in .env")
        all_good = False
    
    if neo4j_password:
        print(f"   ✅ NEO4J_PASSWORD: {'*' * len(neo4j_password)}")
    else:
        print(f"   ❌ NEO4J_PASSWORD not set in .env")
        all_good = False
        
except ImportError:
    print("   ❌ python-dotenv not installed. Run: pip install python-dotenv")
    all_good = False
except Exception as e:
    print(f"   ❌ Error loading environment: {e}")
    all_good = False

if not all_good:
    print("\n❌ Environment variables incomplete. Check your .env file.")
    sys.exit(1)

# Test 3: Check imports
print("\n📦 TEST 3: Checking imports...")
try:
    print("   Importing config...")
    from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    print("   ✅ Config imports work")
except Exception as e:
    print(f"   ❌ Config import failed: {e}")
    all_good = False

try:
    print("   Importing components...")
    from pipeline.components import (
        Intent,
        EntityExtractor,
        HotelIntentClassifier,
        QueryMapper,
        QueryExecutor,
        EnhancedSemanticSearch,
        ResultCombiner
    )
    print("   ✅ Component imports work")
except Exception as e:
    print(f"   ❌ Component import failed: {e}")
    print(f"      Make sure you copied all classes from your notebook!")
    all_good = False

if not all_good:
    print("\n❌ Import errors. Fix the issues above.")
    sys.exit(1)

# Test 4: Check Neo4j connection
print("\n🗄️  TEST 4: Checking Neo4j connection...")
try:
    from neo4j import GraphDatabase
    
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD)
    )
    
    # Try to verify connectivity
    driver.verify_connectivity()
    print("   ✅ Neo4j connection successful!")
    
    # Try a simple query
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN count(n) as count")
        count = result.single()['count']
        print(f"   ✅ Database has {count} nodes")
    
    driver.close()
    
except Exception as e:
    print(f"   ❌ Neo4j connection failed: {e}")
    print(f"      Make sure Neo4j is running and credentials are correct")
    all_good = False

if not all_good:
    print("\n❌ Neo4j connection failed. Fix the issues above.")
    sys.exit(1)

# Test 5: Try to initialize components (optional but recommended)
print("\n🔧 TEST 5: Testing component initialization...")
try:
    from neo4j import GraphDatabase
    
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD)
    )
    
    print("   Initializing EntityExtractor...")
    entity_extractor = EntityExtractor(neo4j_driver=driver)
    print("   ✅ EntityExtractor initialized")
    
    print("   Initializing HotelIntentClassifier...")
    intent_classifier = HotelIntentClassifier(neo4j_driver=driver)
    print("   ✅ HotelIntentClassifier initialized")
    
    print("   Initializing QueryExecutor...")
    query_executor = QueryExecutor(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    print("   ✅ QueryExecutor initialized")
    
    # Clean up
    driver.close()
    query_executor.close()
    
except Exception as e:
    print(f"   ⚠️  Component initialization warning: {e}")
    print(f"      This might be OK if you haven't fully extracted all code yet")

# Test 6: Check if pipeline module exists
print("\n🚀 TEST 6: Checking pipeline module...")
if os.path.exists('pipeline/graph_rag_pipeline.py'):
    print("   ✅ pipeline/graph_rag_pipeline.py exists")
    
    try:
        from pipeline.graph_rag_pipeline import GraphRAGPipeline
        print("   ✅ GraphRAGPipeline can be imported")
    except Exception as e:
        print(f"   ❌ GraphRAGPipeline import failed: {e}")
        all_good = False
else:
    print("   ❌ pipeline/graph_rag_pipeline.py MISSING")
    print("      You need to create this file with the GraphRAGPipeline class")
    all_good = False

# Test 7: Check Streamlit
print("\n🎨 TEST 7: Checking Streamlit...")
try:
    import streamlit
    print(f"   ✅ Streamlit installed (version {streamlit.__version__})")
except ImportError:
    print("   ❌ Streamlit not installed. Run: pip install streamlit")
    all_good = False

# Final verdict
print("\n" + "="*80)
if all_good:
    print("✅ ALL TESTS PASSED! You're ready to run the Streamlit app!")
    print("\nNext steps:")
    print("1. If you haven't created pipeline/graph_rag_pipeline.py, create it now")
    print("2. Run the minimal demo first: streamlit run streamlit_minimal_demo.py")
    print("3. Then run the full app: streamlit run streamlit_app.py")
else:
    print("❌ SOME TESTS FAILED. Please fix the issues above before running Streamlit.")
    print("\nCommon fixes:")
    print("- Create missing directories: mkdir -p pipeline config")
    print("- Create __init__.py files: touch pipeline/__init__.py config/__init__.py")
    print("- Copy classes from notebook to pipeline/components.py")
    print("- Create .env file with your Neo4j credentials")
    print("- Make sure Neo4j is running")

print("="*80)


# Bonus: Quick pipeline test if everything is ready
if all_good and os.path.exists('pipeline/graph_rag_pipeline.py'):
    print("\n🎯 BONUS TEST: Quick pipeline test...")
    
    test_it = input("Do you want to test the pipeline with a simple query? (y/n): ")
    
    if test_it.lower() == 'y':
        try:
            from pipeline.graph_rag_pipeline import GraphRAGPipeline
            
            print("\nInitializing pipeline...")
            pipeline = GraphRAGPipeline(
                neo4j_uri=NEO4J_URI,
                neo4j_user=NEO4J_USER,
                neo4j_password=NEO4J_PASSWORD,
                embedding_model_name='sentence-transformers/all-MiniLM-L6-v2'
            )
            
            print("\nProcessing test query: 'Find hotels in Cairo'")
            results = pipeline.process_query(
                user_query="Find hotels in Cairo",
                use_baseline=True,
                use_embeddings=False,  # Start simple
                verbose=True
            )
            
            print("\n✅ Pipeline test successful!")
            print(f"   Retrieved {len(results['combined_results'])} results")
            print(f"   Intent detected: {results['intent']}")
            print(f"   Entities: {results['entities']}")
            
            pipeline.close()
            
        except Exception as e:
            print(f"\n❌ Pipeline test failed: {e}")
            import traceback
            traceback.print_exc()
