# pipeline/intent_classifier.py

from groq import Groq

client = Groq(api_key="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")

HOTEL_INTENT_PROMPT = """
You are an intent classifier for a Hotel Travel Assistant.
Your task: read the user query and return ONLY ONE intent label from this list:

HOTEL_SEARCH
HOTEL_RECOMMENDATION
REVIEW_QUERY
HOTEL_COMPARISON
VISA_INQUIRY
LOCATION_BASED_QUERY
TRAVELER_PREFERENCE_QUERY
RATING_ANALYSIS
STATISTICAL_QUERY
DEMOGRAPHIC_RECOMMENDATION

Rules:
- Return ONLY the label.
- No sentences.
- No explanation.
- No JSON.

User Query: "{query}"
"""

def classify_hotel_intent(query: str) -> str:
    prompt = HOTEL_INTENT_PROMPT.format(query=query)
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10
    )
    
    return response.choices[0].message.content.strip()