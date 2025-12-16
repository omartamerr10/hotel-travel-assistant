"""
LLM Comparison Test Cases for Graph-RAG Hotel Assistant
Location: Create this file as: pipeline/llm_test_cases.py
"""

# ==============================================================================
# TEST CASES FOR LLM COMPARISON
# ==============================================================================

LLM_TEST_CASES = [
    # ==========================================================================
    # 1. HOTEL_SEARCH Tests (5 cases)
    # ==========================================================================
    {
        "id": "HS_001",
        "query": "Find 5-star hotels in Paris with excellent cleanliness",
        "expected_intent": "HOTEL_SEARCH",
        "expected_entities": {
            "city": ["paris"],
            "star_rating": ["5"],
            "feature": ["cleanliness"]
        },
        "evaluation_criteria": {
            "must_mention": ["L'Étoile Palace", "Paris", "5-star", "cleanliness"],
            "should_include_scores": True,
            "correctness_check": "Should return L'Étoile Palace as it's the only 5-star in Paris"
        },
        "difficulty": "Easy"
    },
    {
        "id": "HS_002",
        "query": "Show me hotels in Dubai with good facilities",
        "expected_intent": "HOTEL_SEARCH",
        "expected_entities": {
            "city": ["dubai"],
            "feature": ["facilities"]
        },
        "evaluation_criteria": {
            "must_mention": ["The Golden Oasis", "Dubai"],
            "should_include_scores": True,
            "correctness_check": "Should mention The Golden Oasis and its facilities score"
        },
        "difficulty": "Easy"
    },
    {
        "id": "HS_003",
        "query": "I need a comfortable hotel in Tokyo",
        "expected_intent": "HOTEL_SEARCH",
        "expected_entities": {
            "city": ["tokyo"],
            "feature": ["comfort"]
        },
        "evaluation_criteria": {
            "must_mention": ["Kyo-to Grand", "Tokyo"],
            "should_include_scores": True,
            "correctness_check": "Should return Kyo-to Grand with comfort ratings"
        },
        "difficulty": "Easy"
    },
    {
        "id": "HS_004",
        "query": "Find hotels in Cairo with star rating of at least 4",
        "expected_intent": "HOTEL_SEARCH",
        "expected_entities": {
            "city": ["cairo"],
            "star_rating": ["4"]
        },
        "evaluation_criteria": {
            "must_mention": ["Nile Grandeur", "Cairo"],
            "should_include_scores": True,
            "correctness_check": "Should list hotels with 4+ stars in Cairo"
        },
        "difficulty": "Medium"
    },
    {
        "id": "HS_005",
        "query": "What are the cleanest hotels in London?",
        "expected_intent": "HOTEL_SEARCH",
        "expected_entities": {
            "city": ["london"],
            "feature": ["cleanliness"]
        },
        "evaluation_criteria": {
            "must_mention": ["The Royal Compass", "London", "cleanliness"],
            "should_include_scores": True,
            "correctness_check": "Should rank hotels by cleanliness score"
        },
        "difficulty": "Medium"
    },

    # ==========================================================================
    # 2. HOTEL_RECOMMENDATION Tests (4 cases)
    # ==========================================================================
    {
        "id": "HR_001",
        "query": "Recommend a hotel for a solo traveler in Paris",
        "expected_intent": "HOTEL_RECOMMENDATION",
        "expected_entities": {
            "city": ["paris"],
            "traveler_type": ["solo", "solo traveler"]
        },
        "evaluation_criteria": {
            "must_mention": ["Paris", "solo traveler"],
            "should_explain_recommendation": True,
            "correctness_check": "Should recommend based on solo traveler preferences"
        },
        "difficulty": "Medium"
    },
    {
        "id": "HR_002",
        "query": "Suggest a good hotel for business travelers in Singapore",
        "expected_intent": "HOTEL_RECOMMENDATION",
        "expected_entities": {
            "city": ["singapore"],
            "traveler_type": ["business", "business traveler"]
        },
        "evaluation_criteria": {
            "must_mention": ["Marina Bay Zenith", "Singapore", "business"],
            "should_explain_recommendation": True,
            "correctness_check": "Should recommend Marina Bay Zenith for business travelers"
        },
        "difficulty": "Medium"
    },
    {
        "id": "HR_003",
        "query": "Best hotel for families in Sydney near the harbour",
        "expected_intent": "HOTEL_RECOMMENDATION",
        "expected_entities": {
            "city": ["sydney"],
            "traveler_type": ["family"]
        },
        "evaluation_criteria": {
            "must_mention": ["Sydney Harbour Grand", "family"],
            "should_explain_recommendation": True,
            "correctness_check": "Should recommend Sydney Harbour Grand as family-friendly"
        },
        "difficulty": "Medium"
    },
    {
        "id": "HR_004",
        "query": "I'm looking for a romantic hotel in Amsterdam for couples",
        "expected_intent": "HOTEL_RECOMMENDATION",
        "expected_entities": {
            "city": ["amsterdam"],
            "traveler_type": ["couple"]
        },
        "evaluation_criteria": {
            "must_mention": ["Canal House Grand", "Amsterdam", "couple"],
            "should_explain_recommendation": True,
            "correctness_check": "Should recommend Canal House Grand for couples"
        },
        "difficulty": "Hard"
    },

    # ==========================================================================
    # 3. REVIEW_QUERY Tests (3 cases)
    # ==========================================================================
    {
        "id": "RQ_001",
        "query": "What do people say about hotels in Dubai?",
        "expected_intent": "REVIEW_QUERY",
        "expected_entities": {
            "city": ["dubai"]
        },
        "evaluation_criteria": {
            "must_mention": ["Dubai", "reviews", "The Golden Oasis"],
            "should_include_review_details": True,
            "correctness_check": "Should summarize reviews from The Golden Oasis"
        },
        "difficulty": "Medium"
    },
    {
        "id": "RQ_002",
        "query": "Show me reviews from business travelers about hotels in Tokyo",
        "expected_intent": "REVIEW_QUERY",
        "expected_entities": {
            "city": ["tokyo"],
            "traveler_type": ["business", "business traveler"]
        },
        "evaluation_criteria": {
            "must_mention": ["Tokyo", "business traveler", "reviews"],
            "should_include_review_details": True,
            "correctness_check": "Should show reviews from business travelers"
        },
        "difficulty": "Hard"
    },
    {
        "id": "RQ_003",
        "query": "What are families saying about hotels in Rome?",
        "expected_intent": "REVIEW_QUERY",
        "expected_entities": {
            "city": ["rome"],
            "traveler_type": ["family"]
        },
        "evaluation_criteria": {
            "must_mention": ["Rome", "family", "Colosseum Gardens"],
            "should_include_review_details": True,
            "correctness_check": "Should summarize family reviews"
        },
        "difficulty": "Hard"
    },

    # ==========================================================================
    # 4. HOTEL_COMPARISON Tests (3 cases)
    # ==========================================================================
    {
        "id": "HC_001",
        "query": "Compare hotels in Paris vs London",
        "expected_intent": "HOTEL_COMPARISON",
        "expected_entities": {
            "city": ["paris", "london"]
        },
        "evaluation_criteria": {
            "must_mention": ["L'Étoile Palace", "The Royal Compass", "comparison"],
            "should_include_metrics": True,
            "correctness_check": "Should compare with ratings"
        },
        "difficulty": "Medium"
    },
    {
        "id": "HC_002",
        "query": "Which is better: The Golden Oasis or Marina Bay Zenith?",
        "expected_intent": "HOTEL_COMPARISON",
        "expected_entities": {
            "hotel": ["the golden oasis", "marina bay zenith"]
        },
        "evaluation_criteria": {
            "must_mention": ["The Golden Oasis", "Marina Bay Zenith", "ratings"],
            "should_include_metrics": True,
            "correctness_check": "Should compare with specific scores"
        },
        "difficulty": "Hard"
    },
    {
        "id": "HC_003",
        "query": "Compare the best hotels in New York and Tokyo",
        "expected_intent": "HOTEL_COMPARISON",
        "expected_entities": {
            "city": ["new york", "tokyo"]
        },
        "evaluation_criteria": {
            "must_mention": ["The Azure Tower", "Kyo-to Grand", "comparison"],
            "should_include_metrics": True,
            "correctness_check": "Should compare top hotels from each city"
        },
        "difficulty": "Hard"
    },

    # ==========================================================================
    # 5. VISA_INQUIRY Tests (4 cases)
    # ==========================================================================
    {
        "id": "VI_001",
        "query": "Do I need a visa to travel from Egypt to France?",
        "expected_intent": "VISA_INQUIRY",
        "expected_entities": {
            "country": ["egypt", "france"]
        },
        "evaluation_criteria": {
            "must_mention": ["Egypt", "France", "visa", "required"],
            "should_specify_visa_type": True,
            "correctness_check": "Should state visa required with type"
        },
        "difficulty": "Easy"
    },
    {
        "id": "VI_002",
        "query": "Visa requirements from India to United Kingdom",
        "expected_intent": "VISA_INQUIRY",
        "expected_entities": {
            "country": ["india", "united kingdom"]
        },
        "evaluation_criteria": {
            "must_mention": ["India", "United Kingdom", "visa"],
            "should_specify_visa_type": True,
            "correctness_check": "Should state visa requirements correctly"
        },
        "difficulty": "Easy"
    },
    {
        "id": "VI_003",
        "query": "Can Chinese citizens visit the United States without a visa?",
        "expected_intent": "VISA_INQUIRY",
        "expected_entities": {
            "country": ["china", "united states"]
        },
        "evaluation_criteria": {
            "must_mention": ["China", "United States", "visa"],
            "should_specify_visa_type": True,
            "correctness_check": "Should state visa is required"
        },
        "difficulty": "Medium"
    },
    {
        "id": "VI_004",
        "query": "What visa do I need to go from Russia to Germany?",
        "expected_intent": "VISA_INQUIRY",
        "expected_entities": {
            "country": ["russia", "germany"]
        },
        "evaluation_criteria": {
            "must_mention": ["Russia", "Germany", "visa type"],
            "should_specify_visa_type": True,
            "correctness_check": "Should specify exact visa type"
        },
        "difficulty": "Medium"
    },

    # ==========================================================================
    # 6. LOCATION_BASED_QUERY Tests (2 cases)
    # ==========================================================================
    {
        "id": "LB_001",
        "query": "Which cities have the most 5-star hotels?",
        "expected_intent": "LOCATION_BASED_QUERY",
        "expected_entities": {
            "star_rating": ["5"]
        },
        "evaluation_criteria": {
            "must_mention": ["cities", "5-star", "hotels"],
            "should_include_statistics": True,
            "correctness_check": "Should list cities with counts"
        },
        "difficulty": "Medium"
    },
    {
        "id": "LB_002",
        "query": "Show me statistics about hotels in France",
        "expected_intent": "LOCATION_BASED_QUERY",
        "expected_entities": {
            "country": ["france"]
        },
        "evaluation_criteria": {
            "must_mention": ["France", "statistics", "average rating"],
            "should_include_statistics": True,
            "correctness_check": "Should show statistics for Paris"
        },
        "difficulty": "Medium"
    },

    # ==========================================================================
    # 7. TRAVELER_PREFERENCE_QUERY Tests (3 cases)
    # ==========================================================================
    {
        "id": "TP_001",
        "query": "Which hotels do business travelers prefer in London?",
        "expected_intent": "TRAVELER_PREFERENCE_QUERY",
        "expected_entities": {
            "city": ["london"],
            "traveler_type": ["business", "business traveler"]
        },
        "evaluation_criteria": {
            "must_mention": ["The Royal Compass", "business travelers", "London"],
            "should_include_traveler_count": True,
            "correctness_check": "Should rank by business traveler preference"
        },
        "difficulty": "Hard"
    },
    {
        "id": "TP_002",
        "query": "Popular hotels for families in Sydney",
        "expected_intent": "TRAVELER_PREFERENCE_QUERY",
        "expected_entities": {
            "city": ["sydney"],
            "traveler_type": ["family"]
        },
        "evaluation_criteria": {
            "must_mention": ["Sydney", "family", "Sydney Harbour Grand"],
            "should_include_traveler_count": True,
            "correctness_check": "Should rank by family counts"
        },
        "difficulty": "Medium"
    },
    {
        "id": "TP_003",
        "query": "What hotels do couples like in Paris?",
        "expected_intent": "TRAVELER_PREFERENCE_QUERY",
        "expected_entities": {
            "city": ["paris"],
            "traveler_type": ["couple"]
        },
        "evaluation_criteria": {
            "must_mention": ["Paris", "couple", "L'Étoile Palace"],
            "should_include_traveler_count": True,
            "correctness_check": "Should show couple preferences"
        },
        "difficulty": "Hard"
    },

    # ==========================================================================
    # 8. RATING_ANALYSIS Tests (2 cases)
    # ==========================================================================
    {
        "id": "RA_001",
        "query": "What are the highest rated hotels in Dubai?",
        "expected_intent": "RATING_ANALYSIS",
        "expected_entities": {
            "city": ["dubai"]
        },
        "evaluation_criteria": {
            "must_mention": ["Dubai", "highest rated", "The Golden Oasis"],
            "should_include_detailed_ratings": True,
            "correctness_check": "Should show all rating categories"
        },
        "difficulty": "Medium"
    },
    {
        "id": "RA_002",
        "query": "Show me rating breakdown for 5-star hotels in Tokyo",
        "expected_intent": "RATING_ANALYSIS",
        "expected_entities": {
            "city": ["tokyo"],
            "star_rating": ["5"]
        },
        "evaluation_criteria": {
            "must_mention": ["Tokyo", "5-star", "rating", "cleanliness", "comfort"],
            "should_include_detailed_ratings": True,
            "correctness_check": "Should provide breakdown"
        },
        "difficulty": "Hard"
    },

    # ==========================================================================
    # 9. STATISTICAL_QUERY Tests (3 cases)
    # ==========================================================================
    {
        "id": "SQ_001",
        "query": "How many hotels are there in Paris?",
        "expected_intent": "STATISTICAL_QUERY",
        "expected_entities": {
            "city": ["paris"]
        },
        "evaluation_criteria": {
            "must_mention": ["Paris", "total", "hotels"],
            "should_include_numbers": True,
            "correctness_check": "Should provide exact count"
        },
        "difficulty": "Easy"
    },
    {
        "id": "SQ_002",
        "query": "What's the average rating of hotels in Singapore?",
        "expected_intent": "STATISTICAL_QUERY",
        "expected_entities": {
            "city": ["singapore"]
        },
        "evaluation_criteria": {
            "must_mention": ["Singapore", "average", "rating"],
            "should_include_numbers": True,
            "correctness_check": "Should calculate average"
        },
        "difficulty": "Medium"
    },
    {
        "id": "SQ_003",
        "query": "Give me statistics about 5-star hotels in Egypt",
        "expected_intent": "STATISTICAL_QUERY",
        "expected_entities": {
            "country": ["egypt"],
            "star_rating": ["5"]
        },
        "evaluation_criteria": {
            "must_mention": ["Egypt", "5-star", "statistics"],
            "should_include_numbers": True,
            "correctness_check": "Should show comprehensive statistics"
        },
        "difficulty": "Hard"
    },

    # ==========================================================================
    # 10. DEMOGRAPHIC_RECOMMENDATION Tests (3 cases)
    # ==========================================================================
    {
        "id": "DR_001",
        "query": "Recommend a hotel for a 28-year-old female solo traveler in Paris",
        "expected_intent": "DEMOGRAPHIC_RECOMMENDATION",
        "expected_entities": {
            "age": ["28"],
            "gender": ["female"],
            "traveler_type": ["solo", "solo traveler"],
            "city": ["paris"]
        },
        "evaluation_criteria": {
            "must_mention": ["28", "female", "solo", "Paris"],
            "should_explain_demographic_match": True,
            "correctness_check": "Should match demographics"
        },
        "difficulty": "Hard"
    },
    {
        "id": "DR_002",
        "query": "Best hotel for a 35-year-old male business traveler in Dubai",
        "expected_intent": "DEMOGRAPHIC_RECOMMENDATION",
        "expected_entities": {
            "age": ["35"],
            "gender": ["male"],
            "traveler_type": ["business", "business traveler"],
            "city": ["dubai"]
        },
        "evaluation_criteria": {
            "must_mention": ["35", "male", "business", "Dubai"],
            "should_explain_demographic_match": True,
            "correctness_check": "Should match demographic profiles"
        },
        "difficulty": "Hard"
    },
    {
        "id": "DR_003",
        "query": "Where should a 42-year-old couple stay in London?",
        "expected_intent": "DEMOGRAPHIC_RECOMMENDATION",
        "expected_entities": {
            "age": ["42"],
            "traveler_type": ["couple"],
            "city": ["london"]
        },
        "evaluation_criteria": {
            "must_mention": ["42", "couple", "London"],
            "should_explain_demographic_match": True,
            "correctness_check": "Should match age and type"
        },
        "difficulty": "Hard"
    }
]

# ==============================================================================
# EVALUATION METRICS DEFINITIONS
# ==============================================================================

QUANTITATIVE_METRICS = {
    "response_time": {
        "description": "Time taken to generate response (seconds)",
        "weight": 0.15
    },
    "token_usage": {
        "description": "Number of tokens consumed",
        "weight": 0.10
    },
    "cost": {
        "description": "Estimated cost per query ($)",
        "weight": 0.10
    },
    "context_adherence": {
        "description": "% of facts from context only",
        "weight": 0.25
    }
}

QUALITATIVE_METRICS = {
    "correctness": {
        "description": "Factual correctness based on KG",
        "weight": 0.30,
        "scale": "1-5"
    },
    "relevance": {
        "description": "Relevance to the query",
        "weight": 0.25,
        "scale": "1-5"
    },
    "completeness": {
        "description": "Includes all key information",
        "weight": 0.20,
        "scale": "1-5"
    },
    "naturalness": {
        "description": "Natural language quality",
        "weight": 0.15,
        "scale": "1-5"
    },
    "hallucination_check": {
        "description": "Absence of made-up info",
        "weight": 0.10,
        "scale": "1-5"
    }
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_test_case_by_id(test_id: str):
    """Retrieve a specific test case by ID"""
    for case in LLM_TEST_CASES:
        if case["id"] == test_id:
            return case
    return None

def get_test_cases_by_intent(intent: str):
    """Get all test cases for a specific intent"""
    return [case for case in LLM_TEST_CASES if case["expected_intent"] == intent]

def get_test_cases_by_difficulty(difficulty: str):
    """Get all test cases of a specific difficulty"""
    return [case for case in LLM_TEST_CASES if case["difficulty"] == difficulty]

def get_all_test_ids():
    """Get list of all test case IDs"""
    return [case["id"] for case in LLM_TEST_CASES]

# Print summary when module is imported
if __name__ == "__main__":
    print(f"\n{'='*80}")
    print("LLM TEST CASES SUMMARY")
    print(f"{'='*80}")
    print(f"Total Test Cases: {len(LLM_TEST_CASES)}")
    print(f"\nBreakdown by Intent:")
    for intent in set(case["expected_intent"] for case in LLM_TEST_CASES):
        count = len(get_test_cases_by_intent(intent))
        print(f"  - {intent}: {count} cases")
    
    print(f"\nBreakdown by Difficulty:")
    for difficulty in ["Easy", "Medium", "Hard"]:
        count = len(get_test_cases_by_difficulty(difficulty))
        print(f"  - {difficulty}: {count} cases")
    print(f"{'='*80}\n")