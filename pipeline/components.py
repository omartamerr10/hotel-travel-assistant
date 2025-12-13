import re
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import spacy
nlp = spacy.load('en_core_web_sm')
from spacy.matcher import PhraseMatcher
from enum import Enum
from groq import Groq
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
import numpy as np
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
import numpy as np
from typing import List, Dict, Tuple, Any
import faiss
import pickle
import os
from groq import Groq
import time

class Intent(Enum):
    """Hotel travel assistant intent types"""
    HOTEL_SEARCH = "HOTEL_SEARCH"
    HOTEL_RECOMMENDATION = "HOTEL_RECOMMENDATION"
    REVIEW_QUERY = "REVIEW_QUERY"
    HOTEL_COMPARISON = "HOTEL_COMPARISON"
    VISA_INQUIRY = "VISA_INQUIRY"
    LOCATION_BASED_QUERY = "LOCATION_BASED_QUERY"
    TRAVELER_PREFERENCE_QUERY = "TRAVELER_PREFERENCE_QUERY"
    RATING_ANALYSIS = "RATING_ANALYSIS"
    STATISTICAL_QUERY = "STATISTICAL_QUERY"
    DEMOGRAPHIC_RECOMMENDATION = "DEMOGRAPHIC_RECOMMENDATION"
    UNKNOWN = "UNKNOWN"

class EntityExtractor:
    """
    Extract entities from user queries using hybrid approach:
    - spaCy PhraseMatcher for named entities
    - Regex for structured patterns
    """

    def __init__(self, neo4j_driver=None):
        """
        Initialize entity extractor.

        Args:
            neo4j_driver: Optional Neo4j driver to fetch entities from database
        """
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

        # Initialize entity lists
        if neo4j_driver:
            # Fetch from Neo4j (production)
            self.cities = self._fetch_cities_from_neo4j(neo4j_driver)
            self.countries = self._fetch_countries_from_neo4j(neo4j_driver)
            self.hotels = self._fetch_hotels_from_neo4j(neo4j_driver)
        else:
            # Use sample lists (development/testing)
            self.cities = [
              'new york', 'nyc', 'london', 'paris', 'tokyo', 'dubai',
              'singapore', 'sydney', 'rio de janeiro', 'rio', 'berlin',
              'toronto', 'shanghai', 'mexico city', 'mumbai', 'rome',
              'cape town', 'seoul', 'moscow', 'cairo', 'barcelona',
              'bangkok', 'istanbul', 'amsterdam', 'buenos aires', 'lagos',
              'wellington'
            ]
            self.countries = [
              'united states', 'usa', 'us','united kingdom', 'uk', 'france', 'japan',
              'united arab emirates', 'uae', 'singapore', 'australia', 'brazil',
              'germany', 'canada', 'china', 'mexico', 'india', 'italy',
              'south africa', 'south korea', 'russia', 'egypt', 'spain',
              'thailand', 'turkey', 'netherlands', 'argentina', 'nigeria',
              'new zealand'
            ]
            self.hotels = [
              'the azure tower', 'the royal compass', "l'étoile palace",
              'kyo-to grand', 'the golden oasis', 'marina bay zenith',
              'sydney harbour grand', 'copacabana lux', 'berlin mitte elite',
              'the maple grove', 'the bund palace', 'aztec heights',
              'the gateway royale', 'colosseum gardens', 'table mountain view',
              'han river oasis', 'kremlin suites', 'nile grandeur',
              "gaudi's retreat", 'the orchid palace', 'the bosphorus inn',
              'canal house grand', 'tango boutique', 'the savannah house',
              'the kiwi grand'
            ]

        # Traveler types (domain knowledge)
        self.traveler_types = [
            'business traveler', 'business', 'solo traveler', 'solo',
            'family', 'couple'
        ]

        # Setup spaCy matchers
        self._setup_matchers()

    def _fetch_cities_from_neo4j(self, driver):
        """Fetch cities from Neo4j database"""
        query = "MATCH (c:City) RETURN DISTINCT c.name as name"
        with driver.session() as session:
            result = session.run(query)
            return [record["name"].lower() for record in result]

    def _fetch_countries_from_neo4j(self, driver):
        """Fetch countries from Neo4j database"""
        query = "MATCH (c:Country) RETURN DISTINCT c.name as name"
        with driver.session() as session:
            result = session.run(query)
            countries = [record["name"].lower() for record in result]

        # Add common abbreviations
        abbreviations = {
            'usa': 'united states',
            'us': 'united states',
            'uk': 'united kingdom',
            'uae': 'united arab emirates'
        }
        countries.extend(abbreviations.keys())
        return countries

    def _fetch_hotels_from_neo4j(self, driver):
        """Fetch hotels from Neo4j database"""
        query = "MATCH (h:Hotel) RETURN DISTINCT h.name as name"
        with driver.session() as session:
            result = session.run(query)
            return [record["name"].lower() for record in result]

    def _setup_matchers(self):
        """Setup PhraseMatcher for named entities"""
        # Create matchers
        self.city_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.country_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.hotel_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.traveler_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")

        # Add patterns (sort by length descending to match longest phrases first)
        city_patterns = [self.nlp.make_doc(city) for city in sorted(self.cities, key=len, reverse=True)]
        country_patterns = [self.nlp.make_doc(country) for country in sorted(self.countries, key=len, reverse=True)]
        hotel_patterns = [self.nlp.make_doc(hotel) for hotel in sorted(self.hotels, key=len, reverse=True)]
        traveler_patterns = [self.nlp.make_doc(t) for t in sorted(self.traveler_types, key=len, reverse=True)]

        self.city_matcher.add("CITY", city_patterns)
        self.country_matcher.add("COUNTRY", country_patterns)
        self.hotel_matcher.add("HOTEL", hotel_patterns)
        self.traveler_matcher.add("TRAVELER_TYPE", traveler_patterns)

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract all entities from text using hybrid approach.

        Args:
            text: User query text

        Returns:
            Dictionary of entity types and their values
        """
        entities = defaultdict(list)

        # Preprocess text
        text_lower = text.lower().strip()

        # Process with spaCy
        doc = self.nlp(text_lower)

        # === SPACY PHRASEMATCHER: For named entities ===

        # Extract cities
        city_matches = self.city_matcher(doc)
        for match_id, start, end in city_matches:
            entities['city'].append(doc[start:end].text)

        # Extract countries
        country_matches = self.country_matcher(doc)
        for match_id, start, end in country_matches:
            entities['country'].append(doc[start:end].text)

        # Extract hotels
        hotel_matches = self.hotel_matcher(doc)
        for match_id, start, end in hotel_matches:
            entities['hotel'].append(doc[start:end].text)

        # Extract traveler types
        traveler_matches = self.traveler_matcher(doc)
        for match_id, start, end in traveler_matches:
            entities['traveler_type'].append(doc[start:end].text)

        # === REGEX: For structured patterns ===

        # Age patterns
        age_pattern = r'\b(\d+)\s*(?:year|yr)s?\s*old\b'
        age_matches = re.findall(age_pattern, text_lower)
        if age_matches:
            entities['age'].extend(age_matches)

        # Star rating patterns
        star_pattern = r'\b(\d)\s*[-\s]?star\b'
        star_matches = re.findall(star_pattern, text_lower)
        if star_matches:
            entities['star_rating'].extend(star_matches)

        # Gender patterns
        if re.search(r'\b(?:male|man|men|boy)\b', text_lower):
            entities['gender'].append('male')
        if re.search(r'\b(?:female|woman|women|girl|lady)\b', text_lower):
            entities['gender'].append('female')

        # Hotel features (using spaCy lemmatization)
        feature_keywords = {
            'clean': 'cleanliness',
            'cleanliness': 'cleanliness',
            'comfort': 'comfort',
            'comfortable': 'comfort',
            'facility': 'facilities',
            'facilities': 'facilities',
            'amenity': 'facilities',
            'amenities': 'facilities',
            'location': 'location',
            'staff': 'staff',
            'service': 'staff',
            'value': 'value_for_money',
            'price': 'value_for_money',
            'money': 'value_for_money'
        }

        for token in doc:
            if token.lemma_ in feature_keywords:
                entities['feature'].append(feature_keywords[token.lemma_])

        # Remove duplicates while preserving order
        for key in entities:
            seen = set()
            entities[key] = [x for x in entities[key] if not (x in seen or seen.add(x))]

        return dict(entities)    
    
class HotelIntentClassifier:
    """
    Classify user intent for hotel travel queries using rule-based approach.
    Integrated with entity extraction.
    """

    def __init__(self, neo4j_driver=None):
        """
        Initialize intent classifier.

        Args:
            neo4j_driver: Optional Neo4j driver for entity extraction
        """
        self.entity_extractor = EntityExtractor(neo4j_driver)

        # Define intent patterns with keywords and weights
        self.intent_patterns = {
            Intent.HOTEL_SEARCH: {
                'keywords': ['find', 'search', 'show', 'list', 'hotels in', 'hotels near', 'available'],
            },
            Intent.HOTEL_RECOMMENDATION: {
                'keywords': ['recommend', 'suggest', 'best hotel', 'good hotel', 'looking for'],
            },
            Intent.REVIEW_QUERY: {
                'keywords': ['review', 'rating', 'what do people say', 'opinion', 'feedback', 'comment'],
            },
            Intent.HOTEL_COMPARISON: {
                'keywords': ['compare', 'versus', 'vs', 'difference', 'between', 'better'],
            },
            Intent.VISA_INQUIRY: {
                'keywords': ['visa', 'visa requirement', 'need visa', 'travel from', 'passport'],
            },
            Intent.LOCATION_BASED_QUERY: {
                'keywords': ['cities', 'countries', 'destinations', 'where', 'location'],
            },
            Intent.TRAVELER_PREFERENCE_QUERY: {
                'keywords': ['travelers prefer', 'popular with', 'suited for', 'type of traveler'],
            },
            Intent.RATING_ANALYSIS: {
                'keywords': ['best rated', 'highest rating', 'top rated', 'rating for'],
            },
            Intent.STATISTICAL_QUERY: {
                'keywords': ['average', 'how many', 'total', 'count', 'statistics', 'distribution'],
            },
            Intent.DEMOGRAPHIC_RECOMMENDATION: {
                'keywords': ['year old', 'years old', 'age', 'male', 'female', 'for me'],
            },
        }

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for intent classification.

        Args:
            text: Raw user query

        Returns:
            Preprocessed text
        """
        # Convert to lowercase and strip whitespace
        text = text.lower().strip()

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def classify(self, text: str) -> Tuple[Intent, float, Dict]:
        """
        Classify user intent and extract entities.

        Args:
            text: User query text

        Returns:
            Tuple of (intent, confidence, metadata)
            - intent: Intent enum value
            - confidence: Confidence score (0-1)
            - metadata: Dictionary containing entities and other info
        """
        # Preprocess text
        preprocessed = self.preprocess_text(text)

        # Extract entities
        entities = self.entity_extractor.extract_entities(text)

        # Calculate intent scores
        intent_scores = {}

        for intent, pattern in self.intent_patterns.items():
            score = 0
            keyword_matches = 0

            for keyword in pattern['keywords']:
                if keyword in preprocessed:
                    keyword_matches += 1

            if keyword_matches > 0:
                # Base score from keyword matches
                score = min(keyword_matches * 0.2, 0.7)

                # Boost score based on entities (entity presence indicates relevance)
                entity_boost = self._calculate_entity_boost(intent, entities)
                score = min(1.0, score + entity_boost)

                intent_scores[intent] = score

        # Special case: demographic recommendation if age/gender found
        if ('age' in entities or 'gender' in entities) and 'traveler_type' in entities:
            if Intent.DEMOGRAPHIC_RECOMMENDATION not in intent_scores or \
               intent_scores.get(Intent.DEMOGRAPHIC_RECOMMENDATION, 0) < 0.8:
                intent_scores[Intent.DEMOGRAPHIC_RECOMMENDATION] = 0.80

        # Special case: visa inquiry if multiple countries mentioned
        if len(entities.get('country', [])) >= 2:
            if Intent.VISA_INQUIRY not in intent_scores or \
               intent_scores.get(Intent.VISA_INQUIRY, 0) < 0.7:
                intent_scores[Intent.VISA_INQUIRY] = 0.70

        # Get best intent
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            intent = best_intent[0]
            confidence = best_intent[1]
        else:
            intent = Intent.UNKNOWN
            confidence = 0.0

        # Prepare metadata
        metadata = {
            'entities': entities,
            'preprocessed_text': preprocessed,
            'original_text': text,
            'all_intent_scores': dict(intent_scores)
        }

        return intent, confidence, metadata

    def _calculate_entity_boost(self, intent: Intent, entities: Dict) -> float:
        """
        Calculate confidence boost based on entity presence for specific intents.

        Args:
            intent: Current intent being evaluated
            entities: Extracted entities

        Returns:
            Boost value (0-0.3)
        """
        boost = 0.0

        # Different intents benefit from different entities
        if intent == Intent.HOTEL_SEARCH:
            if 'city' in entities or 'country' in entities:
                boost += 0.15
            if 'star_rating' in entities or 'feature' in entities:
                boost += 0.10

        elif intent == Intent.HOTEL_RECOMMENDATION:
            if 'traveler_type' in entities:
                boost += 0.15
            if 'city' in entities:
                boost += 0.10

        elif intent == Intent.REVIEW_QUERY:
            if 'hotel' in entities or 'city' in entities:
                boost += 0.15

        elif intent == Intent.VISA_INQUIRY:
            if 'country' in entities and len(entities['country']) >= 1:
                boost += 0.20

        elif intent == Intent.DEMOGRAPHIC_RECOMMENDATION:
            if 'age' in entities or 'gender' in entities:
                boost += 0.15
            if 'traveler_type' in entities:
                boost += 0.10

        return min(0.3, boost)  # Cap at 0.3


class QueryMapper:
    """Maps extracted entities to query parameters for different intents."""
    
    def map_entities_to_parameters(self, intent: str, entities: dict) -> dict:
        """
        Convert extracted entities into query parameters for the selected intent.
        
        Args:
            intent: The classified intent (e.g., 'HOTEL_SEARCH')
            entities: Dictionary of extracted entities
            
        Returns:
            Dictionary of parameters for the Cypher query
        """
        params = {}
        
        # Common mappings used across multiple intents
        if 'city' in entities and entities['city']:
            params['city'] = entities['city'][0]  # Take first match
        
        if 'country' in entities and entities['country']:
            params['country'] = entities['country'][0]
        
        if 'hotel' in entities and entities['hotel']:
            params['hotel_name'] = entities['hotel'][0]
        
        # Intent-specific mappings
        if intent == 'HOTEL_SEARCH':
            # Star rating
            if 'star_rating' in entities and entities['star_rating']:
                params['min_star_rating'] = int(entities['star_rating'][0])
            
            # Features (cleanliness, comfort, facilities)
            if 'feature' in entities and entities['feature']:
                params['feature_type'] = entities['feature'][0]
                params['min_feature_score'] = 4.0  # Threshold for "good" features
            
            params['limit'] = 10
        
        elif intent == 'HOTEL_RECOMMENDATION':
            if 'feature' in entities and entities['feature']:
                params['feature_preference'] = entities['feature'][0]
                params['min_score'] = 4.0
            
            if 'traveler_type' in entities and entities['traveler_type']:
                params['traveler_type'] = entities['traveler_type'][0].capitalize()
            
            params['limit'] = 10
        
        elif intent == 'REVIEW_QUERY':
            params['limit'] = 20
            
            # Score filtering
            params['min_score'] = None
            params['max_score'] = None
            
            # Demographic filters
            if 'gender' in entities and entities['gender']:
                params['gender'] = entities['gender'][0].capitalize()
            
            if 'age' in entities and entities['age']:
                age = int(entities['age'][0])
                params['age_min'] = age - 5
                params['age_max'] = age + 5
            
            if 'traveler_type' in entities and entities['traveler_type']:
                params['traveler_type'] = entities['traveler_type'][0].capitalize()
        
        elif intent == 'HOTEL_COMPARISON':
            if 'hotel' in entities and len(entities['hotel']) >= 2:
                params['hotel_name_1'] = entities['hotel'][0]
                params['hotel_name_2'] = entities['hotel'][1]
                params['city_1'] = None
                params['city_2'] = None
            elif 'city' in entities and len(entities['city']) >= 2:
                params['hotel_name_1'] = None
                params['hotel_name_2'] = None
                params['city_1'] = entities['city'][0]
                params['city_2'] = entities['city'][1]
            elif 'hotel' in entities and len(entities['hotel']) == 1:
                params['hotel_name_1'] = entities['hotel'][0]
                params['hotel_name_2'] = None
                params['city_1'] = None
                params['city_2'] = None
        
        elif intent == 'VISA_INQUIRY':
            if 'country' in entities and len(entities['country']) >= 2:
                params['from_country'] = entities['country'][0]
                params['to_country'] = entities['country'][1]
            elif 'country' in entities and len(entities['country']) == 1:
                params['from_country'] = entities['country'][0]
                params['to_country'] = None
        
        elif intent == 'LOCATION_BASED_QUERY':
            if 'star_rating' in entities and entities['star_rating']:
                params['min_star_rating'] = int(entities['star_rating'][0])
            
            params['min_hotels'] = 1
            params['limit'] = 20
        
        elif intent == 'TRAVELER_PREFERENCE_QUERY':
            if 'traveler_type' in entities and entities['traveler_type']:
                params['traveler_type'] = entities['traveler_type'][0].capitalize()
            
            if 'age' in entities and entities['age']:
                age = int(entities['age'][0])
                params['age_min'] = age - 5
                params['age_max'] = age + 5
            
            if 'gender' in entities and entities['gender']:
                params['gender'] = entities['gender'][0].capitalize()
            
            params['min_travelers'] = 2
            params['limit'] = 10
        
        elif intent == 'RATING_ANALYSIS':
            if 'star_rating' in entities and entities['star_rating']:
                params['star_rating'] = int(entities['star_rating'][0])
            
            params['min_reviews'] = 1
            params['sort_by'] = 'overall'
            params['limit'] = 10
        
        elif intent == 'STATISTICAL_QUERY':
            if 'star_rating' in entities and entities['star_rating']:
                params['star_rating'] = int(entities['star_rating'][0])
            
            params['limit'] = 20
        
        elif intent == 'DEMOGRAPHIC_RECOMMENDATION':
            if 'age' in entities and entities['age']:
                params['age'] = int(entities['age'][0])
            else:
                params['age'] = 30  # Default age
            
            params['age_range'] = 5  # +/- 5 years
            
            if 'gender' in entities and entities['gender']:
                params['gender'] = entities['gender'][0].capitalize()
            
            if 'traveler_type' in entities and entities['traveler_type']:
                params['traveler_type'] = entities['traveler_type'][0].capitalize()
            
            params['min_similar_travelers'] = 2
            params['limit'] = 10
        
        return params


class QueryExecutor:
    """Executes Cypher queries against Neo4j database."""
    
    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize the query executor with Neo4j connection.
        
        Args:
            uri: Neo4j connection URI (e.g., 'neo4j://localhost:7687')
            user: Database username
            password: Database password
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.query_templates = self._load_query_templates()
    
    def _load_query_templates(self) -> Dict[str, str]:
        """Load all Cypher query templates aligned with Intent enum."""
        return {
            # ================================================================
            # 1. HOTEL_SEARCH
            # ================================================================
            'HOTEL_SEARCH': """
                MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
                WHERE (toLower(c.name) = toLower($city) OR $city IS NULL)
                  AND (toLower(co.name) = toLower($country) OR $country IS NULL)
                  AND (h.star_rating >= $min_star_rating OR $min_star_rating IS NULL)
                  AND (
                    ($feature_type IS NULL) OR
                    ($feature_type = 'cleanliness' AND h.cleanliness_base >= $min_feature_score) OR
                    ($feature_type = 'comfort' AND h.comfort_base >= $min_feature_score) OR
                    ($feature_type = 'facilities' AND h.facilities_base >= $min_feature_score) OR
                    ($feature_type = 'location' AND h.average_reviews_score >= $min_feature_score) OR
                    ($feature_type = 'staff' AND h.average_reviews_score >= $min_feature_score)
                  )
                RETURN h.hotel_id AS hotel_id,
                       h.name AS hotel_name, 
                       h.star_rating AS star_rating,
                       h.average_reviews_score AS avg_score,
                       h.cleanliness_base AS cleanliness,
                       h.comfort_base AS comfort,
                       h.facilities_base AS facilities,
                       c.name AS city,
                       co.name AS country
                ORDER BY 
                  CASE $feature_type
                    WHEN 'cleanliness' THEN h.cleanliness_base
                    WHEN 'comfort' THEN h.comfort_base
                    WHEN 'facilities' THEN h.facilities_base
                    ELSE h.average_reviews_score
                  END DESC,
                  h.average_reviews_score DESC
                LIMIT $limit
            """,
            
            # ================================================================
            # 2. HOTEL_RECOMMENDATION
            # ================================================================
            'HOTEL_RECOMMENDATION': """
                MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
                WHERE (toLower(c.name) = toLower($city) OR $city IS NULL)
                  AND (toLower(co.name) = toLower($country) OR $country IS NULL)
                
                OPTIONAL MATCH (t:Traveller)-[:STAYED_AT]->(h)
                WHERE (toLower(t.type) = toLower($traveler_type) OR $traveler_type IS NULL)
                
                WITH h, c, co, COUNT(DISTINCT t) AS similar_traveler_count,
                  CASE $feature_preference
                    WHEN 'cleanliness' THEN h.cleanliness_base
                    WHEN 'comfort' THEN h.comfort_base
                    WHEN 'facilities' THEN h.facilities_base
                    WHEN 'location' THEN h.average_reviews_score
                    WHEN 'staff' THEN h.average_reviews_score
                    WHEN 'value' THEN h.average_reviews_score
                    ELSE h.average_reviews_score
                  END AS feature_score
                
                WHERE feature_score >= $min_score OR $min_score IS NULL
                
                RETURN h.hotel_id AS hotel_id,
                       h.name AS hotel_name,
                       h.star_rating AS star_rating,
                       h.cleanliness_base AS cleanliness,
                       h.comfort_base AS comfort,
                       h.facilities_base AS facilities,
                       h.average_reviews_score AS overall_score,
                       c.name AS city,
                       co.name AS country,
                       similar_traveler_count AS travelers_like_you,
                       feature_score AS recommended_score
                ORDER BY feature_score DESC, similar_traveler_count DESC, h.average_reviews_score DESC
                LIMIT $limit
            """,
            
            # ================================================================
            # 3. REVIEW_QUERY
            # ================================================================
            'REVIEW_QUERY': """
                MATCH (r:Review)-[:REVIEWED]->(h:Hotel)-[:LOCATED_IN]->(c:City)
                MATCH (t:Traveller)-[:WROTE]->(r)
                
                WHERE (toLower(c.name) = toLower($city) OR $city IS NULL)
                  AND (toLower(h.name) CONTAINS toLower($hotel_name) OR $hotel_name IS NULL)
                  AND (r.score_overall >= $min_score OR $min_score IS NULL)
                  AND (r.score_overall <= $max_score OR $max_score IS NULL)
                  AND (toLower(t.type) = toLower($traveler_type) OR $traveler_type IS NULL)
                  AND (t.age >= $age_min OR $age_min IS NULL)
                  AND (t.age <= $age_max OR $age_max IS NULL)
                  AND (toLower(t.gender) = toLower($gender) OR $gender IS NULL)
                
                RETURN h.hotel_id AS hotel_id,
                       h.name AS hotel_name,
                       c.name AS city,
                       r.text AS review_text,
                       r.score_overall AS overall_score,
                       r.score_cleanliness AS cleanliness_score,
                       r.score_comfort AS comfort_score,
                       r.score_facilities AS facilities_score,
                       r.score_location AS location_score,
                       r.score_staff AS staff_score,
                       r.date AS review_date,
                       t.type AS traveler_type,
                       t.age AS traveler_age,
                       t.gender AS traveler_gender
                ORDER BY r.date DESC
                LIMIT $limit
            """,
            
            # ================================================================
            # 4. HOTEL_COMPARISON
            # ================================================================
            'HOTEL_COMPARISON': """
                MATCH (h1:Hotel)-[:LOCATED_IN]->(c1:City)-[:LOCATED_IN]->(co1:Country)
                WHERE (toLower(h1.name) CONTAINS toLower($hotel_name_1) OR 
                       (toLower(c1.name) = toLower($city_1) AND $hotel_name_1 IS NULL))
                WITH h1, c1, co1
                ORDER BY h1.average_reviews_score DESC
                LIMIT 1
                
                MATCH (h2:Hotel)-[:LOCATED_IN]->(c2:City)-[:LOCATED_IN]->(co2:Country)
                WHERE (toLower(h2.name) CONTAINS toLower($hotel_name_2) OR 
                       (toLower(c2.name) = toLower($city_2) AND $hotel_name_2 IS NULL))
                  AND h2.hotel_id <> h1.hotel_id
                WITH h1, c1, co1, h2, c2, co2
                ORDER BY h2.average_reviews_score DESC
                LIMIT 1
                
                OPTIONAL MATCH (t1:Traveller)-[:STAYED_AT]->(h1)
                OPTIONAL MATCH (t2:Traveller)-[:STAYED_AT]->(h2)
                
                WITH h1, c1, co1, h2, c2, co2,
                     COUNT(DISTINCT t1) AS travelers_h1,
                     COUNT(DISTINCT t2) AS travelers_h2
                
                RETURN h1.hotel_id AS hotel_1_id,
                       h1.name AS hotel_1_name,
                       c1.name AS city_1,
                       co1.name AS country_1,
                       h1.star_rating AS hotel_1_stars,
                       h1.average_reviews_score AS hotel_1_rating,
                       h1.cleanliness_base AS hotel_1_cleanliness,
                       h1.comfort_base AS hotel_1_comfort,
                       h1.facilities_base AS hotel_1_facilities,
                       travelers_h1 AS hotel_1_traveler_count,
                       
                       h2.hotel_id AS hotel_2_id,
                       h2.name AS hotel_2_name,
                       c2.name AS city_2,
                       co2.name AS country_2,
                       h2.star_rating AS hotel_2_stars,
                       h2.average_reviews_score AS hotel_2_rating,
                       h2.cleanliness_base AS hotel_2_cleanliness,
                       h2.comfort_base AS hotel_2_comfort,
                       h2.facilities_base AS hotel_2_facilities,
                       travelers_h2 AS hotel_2_traveler_count,
                       
                       (h1.average_reviews_score - h2.average_reviews_score) AS rating_difference,
                       (h1.cleanliness_base - h2.cleanliness_base) AS cleanliness_difference
            """,
            
            # ================================================================
            # 5. VISA_INQUIRY
            # ================================================================
            'VISA_INQUIRY': """
                MATCH (from:Country)
                WHERE toLower(from.name) = toLower($from_country)
                
                MATCH (to:Country)
                WHERE toLower(to.name) = toLower($to_country)
                
                OPTIONAL MATCH (from)-[v:NEEDS_VISA]->(to)
                
                RETURN from.name AS from_country,
                       to.name AS to_country,
                       CASE WHEN v IS NOT NULL 
                            THEN 'Yes' 
                            ELSE 'No' 
                       END AS visa_required,
                       v.visa_type AS visa_type
            """,
            
            # ================================================================
            # 6. LOCATION_BASED_QUERY
            # ================================================================
            'LOCATION_BASED_QUERY': """
                MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
                WHERE (toLower(co.name) = toLower($country) OR $country IS NULL)
                  AND (h.star_rating >= $min_star_rating OR $min_star_rating IS NULL)
                
                WITH c, co, 
                     COUNT(h) AS hotel_count, 
                     AVG(h.average_reviews_score) AS avg_rating,
                     AVG(h.cleanliness_base) AS avg_cleanliness,
                     AVG(h.comfort_base) AS avg_comfort,
                     AVG(h.facilities_base) AS avg_facilities,
                     MAX(h.star_rating) AS max_stars
                
                WHERE hotel_count >= $min_hotels OR $min_hotels IS NULL
                
                RETURN c.name AS city,
                       co.name AS country,
                       hotel_count AS number_of_hotels,
                       ROUND(avg_rating * 100) / 100 AS average_rating,
                       ROUND(avg_cleanliness * 100) / 100 AS average_cleanliness,
                       ROUND(avg_comfort * 100) / 100 AS average_comfort,
                       ROUND(avg_facilities * 100) / 100 AS average_facilities,
                       max_stars AS highest_star_rating
                ORDER BY hotel_count DESC, avg_rating DESC
                LIMIT $limit
            """,
            
            # ================================================================
            # 7. TRAVELER_PREFERENCE_QUERY
            # ================================================================
            'TRAVELER_PREFERENCE_QUERY': """
                MATCH (t:Traveller)-[:STAYED_AT]->(h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
                WHERE (toLower(t.type) = toLower($traveler_type) OR $traveler_type IS NULL)
                  AND (t.age >= $age_min OR $age_min IS NULL)
                  AND (t.age <= $age_max OR $age_max IS NULL)
                  AND (toLower(t.gender) = toLower($gender) OR $gender IS NULL)
                  AND (toLower(c.name) = toLower($city) OR $city IS NULL)
                  AND (toLower(co.name) = toLower($country) OR $country IS NULL)
                
                WITH h, c, co, t.type AS traveler_type,
                     COUNT(DISTINCT t) AS traveler_count,
                     AVG(t.age) AS avg_age
                
                WHERE traveler_count >= $min_travelers OR $min_travelers IS NULL
                
                RETURN h.hotel_id AS hotel_id,
                       h.name AS hotel_name,
                       h.star_rating AS star_rating,
                       h.average_reviews_score AS avg_score,
                       c.name AS city,
                       co.name AS country,
                       traveler_type AS most_common_traveler_type,
                       traveler_count AS matching_travelers,
                       ROUND(avg_age) AS average_traveler_age
                ORDER BY traveler_count DESC, h.average_reviews_score DESC
                LIMIT $limit
            """,
            
            # ================================================================
            # 8. RATING_ANALYSIS
            # ================================================================
            'RATING_ANALYSIS': """
                MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
                WHERE (toLower(c.name) = toLower($city) OR $city IS NULL)
                  AND (toLower(co.name) = toLower($country) OR $country IS NULL)
                  AND (h.star_rating = $star_rating OR $star_rating IS NULL)
                
                OPTIONAL MATCH (r:Review)-[:REVIEWED]->(h)
                
                WITH h, c, co,
                     COUNT(DISTINCT r) AS review_count,
                     AVG(r.score_overall) AS avg_review_score,
                     AVG(r.score_cleanliness) AS avg_review_cleanliness,
                     AVG(r.score_comfort) AS avg_review_comfort,
                     AVG(r.score_facilities) AS avg_review_facilities
                
                WHERE review_count >= $min_reviews OR $min_reviews IS NULL
                
                RETURN h.hotel_id AS hotel_id,
                       h.name AS hotel_name,
                       h.star_rating AS star_rating,
                       c.name AS city,
                       co.name AS country,
                       review_count AS total_reviews,
                       ROUND(h.average_reviews_score * 100) / 100 AS base_overall_rating,
                       ROUND(h.cleanliness_base * 100) / 100 AS base_cleanliness,
                       ROUND(h.comfort_base * 100) / 100 AS base_comfort,
                       ROUND(h.facilities_base * 100) / 100 AS base_facilities,
                       ROUND(avg_review_score * 100) / 100 AS review_overall_rating,
                       CASE 
                         WHEN h.cleanliness_base >= h.comfort_base AND h.cleanliness_base >= h.facilities_base
                         THEN 'Cleanliness'
                         WHEN h.comfort_base >= h.facilities_base
                         THEN 'Comfort'
                         ELSE 'Facilities'
                       END AS strongest_aspect
                ORDER BY 
                  CASE $sort_by
                    WHEN 'overall' THEN h.average_reviews_score
                    WHEN 'cleanliness' THEN h.cleanliness_base
                    WHEN 'comfort' THEN h.comfort_base
                    WHEN 'facilities' THEN h.facilities_base
                    WHEN 'reviews' THEN review_count
                    ELSE h.average_reviews_score
                  END DESC
                LIMIT $limit
            """,
            
            # ================================================================
            # 9. STATISTICAL_QUERY
            # ================================================================
            'STATISTICAL_QUERY': """
                MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
                WHERE (toLower(c.name) = toLower($city) OR $city IS NULL)
                  AND (toLower(co.name) = toLower($country) OR $country IS NULL)
                  AND (h.star_rating = $star_rating OR $star_rating IS NULL)
                
                OPTIONAL MATCH (t:Traveller)-[:STAYED_AT]->(h)
                OPTIONAL MATCH (r:Review)-[:REVIEWED]->(h)
                
                WITH c, co,
                     COUNT(DISTINCT h) AS total_hotels,
                     AVG(h.average_reviews_score) AS avg_overall_rating,
                     AVG(h.cleanliness_base) AS avg_cleanliness,
                     AVG(h.comfort_base) AS avg_comfort,
                     AVG(h.facilities_base) AS avg_facilities,
                     AVG(h.star_rating) AS avg_star_rating,
                     COUNT(DISTINCT r) AS total_reviews,
                     COUNT(DISTINCT t) AS total_travelers
                
                WHERE total_hotels > 0
                
                RETURN c.name AS city,
                       co.name AS country,
                       total_hotels,
                       ROUND(avg_overall_rating * 100) / 100 AS average_rating,
                       ROUND(avg_cleanliness * 100) / 100 AS average_cleanliness,
                       ROUND(avg_comfort * 100) / 100 AS average_comfort,
                       ROUND(avg_facilities * 100) / 100 AS average_facilities,
                       ROUND(avg_star_rating * 10) / 10 AS average_star_rating,
                       total_reviews,
                       total_travelers
                ORDER BY total_hotels DESC, avg_overall_rating DESC
                LIMIT $limit
            """,
            
            # ================================================================
            # 10. DEMOGRAPHIC_RECOMMENDATION
            # ================================================================
            'DEMOGRAPHIC_RECOMMENDATION': """
                MATCH (similar:Traveller)-[:STAYED_AT]->(h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
                
                WHERE (similar.age >= $age - $age_range AND similar.age <= $age + $age_range)
                  AND (toLower(similar.gender) = toLower($gender) OR $gender IS NULL)
                  AND (toLower(similar.type) = toLower($traveler_type) OR $traveler_type IS NULL)
                  AND (toLower(c.name) = toLower($city) OR $city IS NULL)
                  AND (toLower(co.name) = toLower($country) OR $country IS NULL)
                
                OPTIONAL MATCH (similar)-[:WROTE]->(r:Review)-[:REVIEWED]->(h)
                
                WITH h, c, co,
                     COUNT(DISTINCT similar) AS similar_traveler_count,
                     AVG(r.score_overall) AS avg_rating_from_similar,
                     AVG(similar.age) AS avg_similar_age
                
                WHERE similar_traveler_count >= $min_similar_travelers OR $min_similar_travelers IS NULL
                
                RETURN h.hotel_id AS hotel_id,
                       h.name AS hotel_name,
                       h.star_rating AS star_rating,
                       h.average_reviews_score AS overall_avg_score,
                       h.cleanliness_base AS cleanliness,
                       h.comfort_base AS comfort,
                       h.facilities_base AS facilities,
                       c.name AS city,
                       co.name AS country,
                       similar_traveler_count AS travelers_like_you,
                       ROUND(COALESCE(avg_rating_from_similar, h.average_reviews_score) * 100) / 100 AS score_from_similar_travelers,
                       ROUND(avg_similar_age) AS average_age_of_similar_travelers
                ORDER BY similar_traveler_count DESC, score_from_similar_travelers DESC
                LIMIT $limit
            """
        }
    
    def execute_query(self, intent: str, params: Dict[str, Any]) -> List[Dict]:
        """
        Execute the appropriate query with parameters.
        
        Args:
            intent: The intent type (matches query template key)
            params: Dictionary of parameters for the query
            
        Returns:
            List of result dictionaries
        """
        query = self.query_templates.get(intent)
        if not query:
            raise ValueError(f"Unknown intent: {intent}. Available intents: {list(self.query_templates.keys())}")
        
        # Ensure all possible parameters have a value (None if not provided)
        all_possible_params = [
            'city', 'country', 'hotel_name', 'min_star_rating', 'star_rating',
            'feature_type', 'min_feature_score', 'feature_preference', 'min_score', 'max_score',
            'traveler_type', 'age', 'age_min', 'age_max', 'age_range', 'gender',
            'limit', 'min_hotels', 'min_travelers', 'min_similar_travelers', 'min_reviews',
            'from_country', 'to_country',
            'hotel_name_1', 'hotel_name_2', 'city_1', 'city_2',
            'sort_by'
        ]
        
        for key in all_possible_params:
            if key not in params:
                params[key] = None
        
        try:
            with self.driver.session() as session:
                result = session.run(query, **params)
                return [dict(record) for record in result]
        except Exception as e:
            print(f"Error executing query for intent '{intent}': {str(e)}")
            print(f"Parameters: {params}")
            raise
    
    def close(self):
        """Close the database connection."""
        self.driver.close()


class EnhancedSemanticSearch:
    """
    Perform semantic similarity search for:
    1. Hotels (using Neo4j vector index)
    2. Visa requirements (using FAISS since Neo4j can't index relationships)
    """

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        """Initialize with Neo4j connection and embedding models."""
        # Connect to Neo4j
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

        # Load embedding models
        print("Loading embedding models...")
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.bge_model = SentenceTransformer('BAAI/bge-large-en-v1.5')

        # FAISS indexes for visa requirements
        self.visa_faiss_index_sbert = None
        self.visa_faiss_index_bge = None
        self.visa_id_to_data = {}  # Maps index position to visa data
        self.index_to_visa_id = {}  # Maps FAISS index to visa_id

        print("✓ Initialization complete")

    # ========================================================================
    # HOTEL SEMANTIC SEARCH (Neo4j Vector Index)
    # ========================================================================

    def search_hotels_semantic(
        self,
        query: str,
        model: str = 'sbert',
        top_k: int = 10,
        city_filter: str = None,  # ADD THIS PARAMETER
        country_filter: str = None  # ADD THIS PARAMETER
    ) -> List[Dict]:
        """
        Search hotels using Neo4j's native vector index.

        Args:
            query: User search query
            model: 'sbert' or 'bge'
            top_k: Number of results to return
            city_filter: Optional city name to filter results
            country_filter: Optional country name to filter results

        Returns:
            List of hotel dictionaries with similarity scores
        """
        # Step 1: Embed the query
        if model == 'sbert':
            query_embedding = self.sbert_model.encode(query).tolist()
            index_name = 'hotel_sbert_index'
        elif model == 'bge':
            query_embedding = self.bge_model.encode(query).tolist()
            index_name = 'hotel_bge_index'
        else:
            raise ValueError("model must be 'sbert' or 'bge'")

        # Step 2: Build WHERE clause for filters
        where_clauses = []
        if city_filter:
            where_clauses.append("toLower(c.name) = toLower($city_filter)")
        if country_filter:
            where_clauses.append("toLower(co.name) = toLower($country_filter)")
        
        where_clause = ""
        if where_clauses:
            where_clause = "WHERE " + " AND ".join(where_clauses)

        # Step 3: Query Neo4j vector index WITH FILTERS
        cypher_query = f"""
        CALL db.index.vector.queryNodes(
            $index_name,
            $top_k * 3,
            $query_embedding
        )
        YIELD node, score
        MATCH (node)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        {where_clause}
        RETURN node.hotel_id AS hotel_id,
            node.name AS hotel_name,
            node.star_rating AS star_rating,
            node.average_reviews_score AS avg_score,
            node.cleanliness_base AS cleanliness,
            node.comfort_base AS comfort,
            node.facilities_base AS facilities,
            c.name AS city,
            co.name AS country,
            score AS similarity_score
        ORDER BY score DESC
        LIMIT $top_k
        """

        with self.driver.session() as session:
            result = session.run(
                cypher_query,
                index_name=index_name,
                top_k=top_k,
                query_embedding=query_embedding,
                city_filter=city_filter,
                country_filter=country_filter
            )
            hotels = [dict(record) for record in result]

        return hotels

    # ========================================================================
    # VISA SEMANTIC SEARCH (FAISS - since Neo4j can't index relationships)
    # ========================================================================

    def build_visa_faiss_index(self, model: str = 'both'):
        """
        Build FAISS index for visa requirements from Neo4j.
        Must be called once before searching.

        Args:
            model: 'sbert', 'bge', or 'both'
        """
        print("\nBuilding FAISS index for visa requirements...")

        # Fetch all visa embeddings from Neo4j
        query = """
        MATCH (from:Country)-[v:NEEDS_VISA]->(to:Country)
        WHERE v.embedding_sbert IS NOT NULL
        RETURN from.name AS from_country,
               to.name AS to_country,
               v.visa_type AS visa_type,
               v.embedding_sbert AS embedding_sbert,
               v.embedding_bge AS embedding_bge
        """

        with self.driver.session() as session:
            result = session.run(query)
            visa_data = [dict(record) for record in result]

        if not visa_data:
            print("⚠ No visa embeddings found in Neo4j!")
            return

        print(f"Found {len(visa_data)} visa requirements")

        # Prepare embeddings and metadata
        sbert_embeddings = []
        bge_embeddings = []

        for idx, visa in enumerate(visa_data):
            # Create visa_id
            visa_id = f"visa_{visa['from_country'].lower().replace(' ', '_')}_to_{visa['to_country'].lower().replace(' ', '_')}"

            # Store mapping
            self.visa_id_to_data[idx] = {
                'visa_id': visa_id,
                'from_country': visa['from_country'],
                'to_country': visa['to_country'],
                'visa_type': visa['visa_type']
            }
            self.index_to_visa_id[idx] = visa_id

            # Collect embeddings
            if model in ['sbert', 'both'] and visa['embedding_sbert']:
                sbert_embeddings.append(visa['embedding_sbert'])

            if model in ['bge', 'both'] and visa['embedding_bge']:
                bge_embeddings.append(visa['embedding_bge'])

        # Build FAISS indexes
        if model in ['sbert', 'both'] and sbert_embeddings:
            sbert_array = np.array(sbert_embeddings, dtype='float32')
            dimension = sbert_array.shape[1]
            self.visa_faiss_index_sbert = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
            faiss.normalize_L2(sbert_array)  # Normalize for cosine similarity
            self.visa_faiss_index_sbert.add(sbert_array)
            print(f"✓ Built SBERT FAISS index ({dimension}D, {len(sbert_embeddings)} vectors)")

        if model in ['bge', 'both'] and bge_embeddings:
            bge_array = np.array(bge_embeddings, dtype='float32')
            dimension = bge_array.shape[1]
            self.visa_faiss_index_bge = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(bge_array)
            self.visa_faiss_index_bge.add(bge_array)
            print(f"✓ Built BGE FAISS index ({dimension}D, {len(bge_embeddings)} vectors)")

    def search_visa_semantic(
        self,
        query: str,
        model: str = 'sbert',
        top_k: int = 10
    ) -> List[Dict]:
        """
        Search visa requirements using FAISS semantic similarity.

        Args:
            query: User search query (e.g., "visa from Egypt to France")
            model: 'sbert' or 'bge'
            top_k: Number of results to return

        Returns:
            List of visa requirement dictionaries with similarity scores
        """
        # Check if index is built
        if model == 'sbert' and self.visa_faiss_index_sbert is None:
            raise ValueError("SBERT FAISS index not built. Call build_visa_faiss_index() first.")
        if model == 'bge' and self.visa_faiss_index_bge is None:
            raise ValueError("BGE FAISS index not built. Call build_visa_faiss_index() first.")

        # Step 1: Embed the query
        if model == 'sbert':
            query_embedding = self.sbert_model.encode(query)
            faiss_index = self.visa_faiss_index_sbert
        else:
            query_embedding = self.bge_model.encode(query)
            faiss_index = self.visa_faiss_index_bge

        # Normalize query embedding for cosine similarity
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)

        # Step 2: Search FAISS index
        distances, indices = faiss_index.search(query_embedding, top_k)

        # Step 3: Convert to results
        results = []
        for i, (idx, score) in enumerate(zip(indices[0], distances[0])):
            if idx == -1:  # No more results
                break

            visa_data = self.visa_id_to_data[idx]
            results.append({
                'from_country': visa_data['from_country'],
                'to_country': visa_data['to_country'],
                'visa_type': visa_data['visa_type'],
                'similarity_score': float(score),
                'rank': i + 1
            })

        return results

    # ========================================================================
    # COMBINED SEARCH (Both Hotels and Visas)
    # ========================================================================

    def search_combined(
        self,
        query: str,
        search_hotels: bool = True,
        search_visas: bool = True,
        model: str = 'sbert',
        top_k: int = 10
    ) -> Dict[str, List[Dict]]:
        """
        Search both hotels and visa requirements.

        Args:
            query: User search query
            search_hotels: Whether to search hotels
            search_visas: Whether to search visas
            model: 'sbert' or 'bge'
            top_k: Number of results per category

        Returns:
            Dictionary with 'hotels' and 'visas' keys
        """
        results = {}

        if search_hotels:
            results['hotels'] = self.search_hotels_semantic(query, model, top_k)

        if search_visas:
            results['visas'] = self.search_visa_semantic(query, model, top_k)

        return results

    # ========================================================================
    # UTILITY
    # ========================================================================

    def save_visa_index(self, filepath: str = 'visa_faiss_index.pkl'):
        """Save FAISS index and metadata to disk."""
        data = {
            'visa_id_to_data': self.visa_id_to_data,
            'index_to_visa_id': self.index_to_visa_id
        }

        # Save FAISS indexes
        if self.visa_faiss_index_sbert:
            faiss.write_index(self.visa_faiss_index_sbert, 'visa_sbert.index')
        if self.visa_faiss_index_bge:
            faiss.write_index(self.visa_faiss_index_bge, 'visa_bge.index')

        # Save metadata
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"✓ Saved FAISS index to {filepath}")

    def load_visa_index(self, filepath: str = 'visa_faiss_index.pkl'):
        """Load FAISS index and metadata from disk."""
        # Load metadata
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.visa_id_to_data = data['visa_id_to_data']
        self.index_to_visa_id = data['index_to_visa_id']

        # Load FAISS indexes
        if os.path.exists('visa_sbert.index'):
            self.visa_faiss_index_sbert = faiss.read_index('visa_sbert.index')
            print("✓ Loaded SBERT FAISS index")

        if os.path.exists('visa_bge.index'):
            self.visa_faiss_index_bge = faiss.read_index('visa_bge.index')
            print("✓ Loaded BGE FAISS index")

    def close(self):
        """Close Neo4j connection."""
        self.driver.close()


# ============================================================================
# Example Usage
# ============================================================================

def main():
    """Example usage of enhanced semantic search."""

    # Initialize
    searcher = EnhancedSemanticSearch(
        neo4j_uri="neo4j://127.0.0.1:7687",
        neo4j_user="neo4j",
        neo4j_password="Omarino2003"
    )

    try:
        # Build FAISS index for visa requirements (only need to do this once)
        searcher.build_visa_faiss_index(model='both')

        # Example 1: Search hotels
        print("\n" + "="*80)
        print("EXAMPLE 1: Hotel Search")
        print("="*80)

        hotel_query = "luxury hotel with spa and excellent cleanliness"
        hotel_results = searcher.search_hotels_semantic(
            query=hotel_query,
            model='sbert',
            top_k=5
        )

        print(f"\nQuery: {hotel_query}")
        print("\nTop 5 Hotels:")
        for i, hotel in enumerate(hotel_results, 1):
            print(f"{i}. {hotel['hotel_name']} - {hotel['city']}, {hotel['country']}")
            print(f"   Similarity: {hotel['similarity_score']:.4f}")
            print(f"   Rating: {hotel['avg_score']/2:.1f}/5")

        # Example 2: Search visa requirements
        print("\n" + "="*80)
        print("EXAMPLE 2: Visa Search")
        print("="*80)

        visa_query = "visa requirements from Egypt to France"
        visa_results = searcher.search_visa_semantic(
            query=visa_query,
            model='sbert',
            top_k=5
        )

        print(f"\nQuery: {visa_query}")
        print("\nTop 5 Visa Requirements:")
        for i, visa in enumerate(visa_results, 1):
            print(f"{i}. {visa['from_country']} → {visa['to_country']}")
            print(f"   Visa Type: {visa['visa_type']}")
            print(f"   Similarity: {visa['similarity_score']:.4f}")

        # Example 3: Combined search
        print("\n" + "="*80)
        print("EXAMPLE 3: Combined Search")
        print("="*80)

        combined_query = "travel to Paris with visa"
        combined_results = searcher.search_combined(
            query=combined_query,
            search_hotels=True,
            search_visas=True,
            model='sbert',
            top_k=3
        )

        print(f"\nQuery: {combined_query}")

        if 'hotels' in combined_results:
            print("\nHotels:")
            for hotel in combined_results['hotels']:
                print(f"  - {hotel['hotel_name']} ({hotel['similarity_score']:.4f})")

        if 'visas' in combined_results:
            print("\nVisa Requirements:")
            for visa in combined_results['visas']:
                print(f"  - {visa['from_country']} → {visa['to_country']} ({visa['similarity_score']:.4f})")

        # Save index for future use
        searcher.save_visa_index()

    finally:
        searcher.close()


if __name__ == "__main__":
    main()


class ResultCombiner:
    """
    Combines results from baseline (Cypher) and embeddings (semantic search).
    Handles BOTH hotels AND visas intelligently.
    """
    
    @staticmethod
    def combine_results(
        cypher_results: List[Dict],
        semantic_results: List[Dict],
        query_type: str = None,
        intent: str = None,
        max_results: int = 10
    ) -> Tuple[List[Dict], Dict]:
        """
        Intelligently combine results based on query type.
        
        Args:
            cypher_results: Results from Cypher queries (baseline)
            semantic_results: Results from semantic similarity search
            query_type: 'hotel' or 'visa' (auto-detected if not provided)
            intent: The classified intent (optional)
            max_results: Maximum number of results to return
            
        Returns:
            Tuple of (combined_results, metadata)
        """
        # Auto-detect query type if not provided
        if not query_type:
            if intent and 'VISA' in intent:
                query_type = 'visa'
            else:
                query_type = 'hotel'
        
        # Route to appropriate combiner
        if query_type == 'visa':
            return ResultCombiner._combine_visa_results(
                cypher_results, semantic_results, max_results
            )
        else:
            return ResultCombiner._combine_hotel_results(
                cypher_results, semantic_results, max_results
            )
    
    @staticmethod
    def _combine_hotel_results(
        cypher_results: List[Dict],
        semantic_results: List[Dict],
        max_results: int
    ) -> Tuple[List[Dict], Dict]:
        """
        Combine hotel results from both retrieval methods.
        """
        hotel_map = {}
        
        # Process Cypher results first (exact matches have priority)
        for idx, result in enumerate(cypher_results):
            hotel_id = result.get('hotel_id') or result.get('hotel_name')
            if hotel_id and hotel_id not in hotel_map:
                hotel_map[hotel_id] = {
                    **result,
                    'source': 'cypher',
                    'cypher_rank': idx + 1,
                    'semantic_rank': None,
                    'similarity_score': None
                }
        
        # Add semantic results, merge if hotel already exists
        for idx, result in enumerate(semantic_results):
            hotel_id = result.get('hotel_id') or result.get('hotel_name')
            if hotel_id:
                if hotel_id in hotel_map:
                    # Hotel found in BOTH - merge information
                    hotel_map[hotel_id]['source'] = 'both'
                    hotel_map[hotel_id]['semantic_rank'] = idx + 1
                    hotel_map[hotel_id]['similarity_score'] = result.get('similarity_score')
                else:
                    # New hotel from semantic search only
                    hotel_map[hotel_id] = {
                        **result,
                        'source': 'semantic',
                        'cypher_rank': None,
                        'semantic_rank': idx + 1
                    }
        
        # Convert to list and sort by priority
        combined = list(hotel_map.values())
        
        # Sorting priority: both > cypher > semantic
        def sort_key(hotel):
            if hotel['source'] == 'both':
                return (0, hotel['cypher_rank'] or 999, -(hotel['similarity_score'] or 0))
            elif hotel['source'] == 'cypher':
                return (1, hotel['cypher_rank'], 0)
            else:  # semantic
                return (2, hotel['semantic_rank'], -(hotel['similarity_score'] or 0))
        
        combined.sort(key=sort_key)
        combined = combined[:max_results]
        
        # Create metadata
        metadata = {
            'query_type': 'hotel',
            'total_cypher_results': len(cypher_results),
            'total_semantic_results': len(semantic_results),
            'combined_unique_results': len(hotel_map),
            'returned_results': len(combined),
            'results_in_both': sum(1 for h in combined if h['source'] == 'both'),
            'results_cypher_only': sum(1 for h in combined if h['source'] == 'cypher'),
            'results_semantic_only': sum(1 for h in combined if h['source'] == 'semantic'),
            'combination_strategy': 'hotel_merge_and_rank'
        }
        
        return combined, metadata
    
    @staticmethod
    def _combine_visa_results(
        cypher_results: List[Dict],
        semantic_results: List[Dict],
        max_results: int
    ) -> Tuple[List[Dict], Dict]:
        """
        Combine visa results from both retrieval methods.
        """
        visa_map = {}
        
        # Process Cypher results first
        for idx, result in enumerate(cypher_results):
            # Create unique key for visa requirement
            from_c = result.get('from_country', '').lower().replace(' ', '_')
            to_c = result.get('to_country', '').lower().replace(' ', '_')
            visa_key = f"{from_c}_{to_c}"
            
            if visa_key and visa_key != '_' and visa_key not in visa_map:
                visa_map[visa_key] = {
                    **result,
                    'source': 'cypher',
                    'cypher_rank': idx + 1,
                    'semantic_rank': None,
                    'similarity_score': None
                }
        
        # Add semantic results, merge if visa already exists
        for idx, result in enumerate(semantic_results):
            from_c = result.get('from_country', '').lower().replace(' ', '_')
            to_c = result.get('to_country', '').lower().replace(' ', '_')
            visa_key = f"{from_c}_{to_c}"
            
            if visa_key and visa_key != '_':
                if visa_key in visa_map:
                    # Visa found in BOTH - merge information
                    visa_map[visa_key]['source'] = 'both'
                    visa_map[visa_key]['semantic_rank'] = idx + 1
                    visa_map[visa_key]['similarity_score'] = result.get('similarity_score')
                else:
                    # New visa from semantic search only
                    visa_map[visa_key] = {
                        **result,
                        'source': 'semantic',
                        'cypher_rank': None,
                        'semantic_rank': idx + 1
                    }
        
        # Convert to list and sort by priority
        combined = list(visa_map.values())
        
        # Sorting priority: both > cypher > semantic
        def sort_key(visa):
            if visa['source'] == 'both':
                return (0, visa['cypher_rank'] or 999, -(visa['similarity_score'] or 0))
            elif visa['source'] == 'cypher':
                return (1, visa['cypher_rank'], 0)
            else:  # semantic
                return (2, visa['semantic_rank'], -(visa['similarity_score'] or 0))
        
        combined.sort(key=sort_key)
        combined = combined[:max_results]
        
        # Create metadata
        metadata = {
            'query_type': 'visa',
            'total_cypher_results': len(cypher_results),
            'total_semantic_results': len(semantic_results),
            'combined_unique_results': len(visa_map),
            'returned_results': len(combined),
            'results_in_both': sum(1 for v in combined if v['source'] == 'both'),
            'results_cypher_only': sum(1 for v in combined if v['source'] == 'cypher'),
            'results_semantic_only': sum(1 for v in combined if v['source'] == 'semantic'),
            'combination_strategy': 'visa_merge_and_rank'
        }
        
        return combined, metadata

class PromptBuilder:
    """
    Builds structured prompts with context, persona, and task components.
    Handles BOTH hotel and visa queries.
    """
    
    @staticmethod
    def build_prompt(
        user_query: str,
        combined_results: List[Dict],
        metadata: Dict,
        query_type: str = None,
        persona: str = None
    ) -> str:
        """
        Build a structured prompt for the LLM.
        Auto-detects query type and routes appropriately.
        """
        # Auto-detect query type from metadata
        if query_type is None:
            query_type = metadata.get('query_type', 'hotel')
        
        # Route to appropriate builder
        if query_type == 'visa':
            return PromptBuilder._build_visa_prompt(
                user_query, combined_results, metadata, persona
            )
        else:
            return PromptBuilder._build_hotel_prompt(
                user_query, combined_results, metadata, persona
            )
    
    @staticmethod
    def _build_hotel_prompt(
        user_query: str,
        combined_results: List[Dict],
        metadata: Dict,
        persona: str = None
    ) -> str:
        """Build prompt for hotel queries."""
        if persona is None:
            persona = "helpful hotel travel assistant"
        
        context = PromptBuilder._format_hotel_context(combined_results, metadata)
        
        prompt = f"""You are a {persona} with access to a knowledge graph of hotels, reviews, and travel information.

CONTEXT FROM KNOWLEDGE GRAPH:
{context}

RETRIEVAL METADATA:
- Total results found: {metadata['combined_unique_results']}
- Results from structured queries: {metadata['total_cypher_results']}
- Results from semantic search: {metadata['total_semantic_results']}
- Results appearing in both: {metadata['results_in_both']}

USER QUESTION:
{user_query}

TASK:
Answer the user's question using ONLY the information provided in the context above. 

INSTRUCTIONS:
1. Base your answer solely on the retrieved hotel information
2. If the context doesn't contain enough information, say so clearly
3. Cite specific hotel names and details when possible
4. Be concise but informative
5. If multiple hotels match, mention the top 3-5 options
6. Include relevant ratings, locations, and key features
7. Do not make up or hallucinate information not present in the context

ANSWER:"""
        
        return prompt
    
    @staticmethod
    def _build_visa_prompt(
        user_query: str,
        combined_results: List[Dict],
        metadata: Dict,
        persona: str = None
    ) -> str:
        """Build prompt for visa queries."""
        if persona is None:
            persona = "helpful visa and travel requirements assistant"
        
        context = PromptBuilder._format_visa_context(combined_results, metadata)
        
        prompt = f"""You are a {persona} with access to a knowledge graph of visa requirements and travel regulations.

CONTEXT FROM KNOWLEDGE GRAPH:
{context}

RETRIEVAL METADATA:
- Total results found: {metadata['combined_unique_results']}
- Results from structured queries: {metadata['total_cypher_results']}
- Results from semantic search: {metadata['total_semantic_results']}
- Results appearing in both: {metadata['results_in_both']}

USER QUESTION:
{user_query}

TASK:
Answer the user's question about visa requirements using ONLY the information provided in the context above.

INSTRUCTIONS:
1. Base your answer solely on the retrieved visa information
2. Clearly state whether a visa is required or not
3. Specify the visa type if applicable
4. Be precise about the countries involved
5. If multiple country pairs are relevant, list them clearly
6. Do not make up or hallucinate visa requirements
7. If information is missing, state this clearly

ANSWER:"""
        
        return prompt
    
    @staticmethod
    def _format_hotel_context(results: List[Dict], metadata: Dict) -> str:
        """Format hotel results into readable context."""
        if not results:
            return "No relevant hotels found in the knowledge graph."
        
        context_parts = []
        
        for idx, hotel in enumerate(results, 1):
            hotel_info = []
            hotel_info.append(f"\n{idx}. {hotel.get('hotel_name', 'Unknown Hotel')}")
            
            # Location
            city = hotel.get('city', '')
            country = hotel.get('country', '')
            if city or country:
                hotel_info.append(f"   Location: {city}, {country}")
            
            # Ratings
            if 'star_rating' in hotel:
                hotel_info.append(f"   Star Rating: {hotel['star_rating']} stars")
            
            if 'avg_score' in hotel:
                hotel_info.append(f"   Average Score: {hotel['avg_score']:.1f}/10")
            elif 'overall_avg_score' in hotel:
                hotel_info.append(f"   Average Score: {hotel['overall_avg_score']:.1f}/10")
            
            # Features
            if 'cleanliness' in hotel:
                hotel_info.append(f"   Cleanliness: {hotel['cleanliness']:.1f}/10")
            if 'comfort' in hotel:
                hotel_info.append(f"   Comfort: {hotel['comfort']:.1f}/10")
            if 'facilities' in hotel:
                hotel_info.append(f"   Facilities: {hotel['facilities']:.1f}/10")
            
            # Source info
            source_info = []
            if hotel['source'] == 'both':
                source_info.append("Found in structured search AND semantic similarity")
            elif hotel['source'] == 'cypher':
                source_info.append("Found in structured search")
            else:
                source_info.append("Found in semantic similarity search")
            
            if hotel.get('similarity_score'):
                source_info.append(f"Similarity: {hotel['similarity_score']:.3f}")
            
            hotel_info.append(f"   [Source: {', '.join(source_info)}]")
            
            context_parts.append('\n'.join(hotel_info))
        
        return '\n'.join(context_parts)
    
    @staticmethod
    def _format_visa_context(results: List[Dict], metadata: Dict) -> str:
        """Format visa results into readable context."""
        if not results:
            return "No relevant visa requirements found in the knowledge graph."
        
        context_parts = []
        
        for idx, visa in enumerate(results, 1):
            visa_info = []
            from_country = visa.get('from_country', 'Unknown')
            to_country = visa.get('to_country', 'Unknown')
            visa_type = visa.get('visa_type', 'Unknown')
            
            visa_info.append(f"\n{idx}. Travel from {from_country} to {to_country}")
            visa_info.append(f"   Visa Required: YES")
            visa_info.append(f"   Visa Type: {visa_type}")
            
            # Source info
            source_info = []
            if visa['source'] == 'both':
                source_info.append("Found in structured search AND semantic similarity")
            elif visa['source'] == 'cypher':
                source_info.append("Found in structured search")
            else:
                source_info.append("Found in semantic similarity search")
            
            if visa.get('similarity_score'):
                source_info.append(f"Similarity: {visa['similarity_score']:.3f}")
            
            visa_info.append(f"   [Source: {', '.join(source_info)}]")
            
            context_parts.append('\n'.join(visa_info))
        
        return '\n'.join(context_parts)
    
    # Backward compatibility
    @staticmethod
    def build_hotel_assistant_prompt(user_query, combined_results, metadata, persona=None):
        """Backward compatible method."""
        return PromptBuilder.build_prompt(
            user_query, combined_results, metadata, 'hotel', persona
        )


class LLMClient:
    """
    Handles communication with LLM providers via Groq.
    UPDATED: Strips <think> tags from reasoning models (Qwen/DeepSeek).
    """
    
    def __init__(self, groq_api_key: str, huggingface_api_key: str = None):
        self.groq_client = Groq(api_key=groq_api_key)
        self.hf_api_key = huggingface_api_key
        
        # Define available models
        self.models = {
            'llama-3.1-8b': {
                'name': 'llama-3.1-8b-instant',
                'provider': 'groq',
                'description': 'Meta Llama 3.1 8B - Instant speed'
            },
            'llama-3.3-70b': {
                'name': 'llama-3.3-70b-versatile',
                'provider': 'groq',
                'description': 'Meta Llama 3.3 70B - High reasoning'
            },
            'qwen-32b': {
                'name': 'qwen-2.5-32b',
                'provider': 'groq',
                'description': 'Qwen 2.5 32B - Strong performance'
            }
        }
    
    def generate_response(
        self,
        prompt: str,
        model_key: str = 'llama-3.1-8b',
        temperature: float = 0.3,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        
        # Auto-map old keys
        if 'mistral' in model_key or '3.1-70b' in model_key: model_key = 'llama-3.3-70b'
        if 'gemma' in model_key or 'deepseek' in model_key: model_key = 'qwen-32b'
        
        if model_key not in self.models:
             model_key = 'llama-3.1-8b'
        
        model_info = self.models[model_key]
        
        return self._generate_groq(prompt, model_key, model_info, temperature, max_tokens)
    
    def _generate_groq(
        self,
        prompt: str,
        model_key: str,
        model_info: Dict,
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Generate response using Groq API"""
        start_time = time.time()
        
        try:
            response = self.groq_client.chat.completions.create(
                model=model_info['name'],
                messages=[
                    {"role": "system", "content": "You are a helpful hotel travel assistant. Answer based only on the context provided."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            elapsed_time = time.time() - start_time
            raw_answer = response.choices[0].message.content
            # Strip out <think>...</think> tags and their content
            clean_answer = re.sub(r'<think>.*?</think>', '', raw_answer, flags=re.DOTALL).strip()
            
            return {
                'success': True,
                'answer': clean_answer, # Return the cleaned answer
                'model': model_key,
                'model_name': model_info['name'],
                'response_time': elapsed_time,
                'tokens_used': response.usage.total_tokens,
                'finish_reason': response.choices[0].finish_reason
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'model': model_key,
                'response_time': time.time() - start_time
            }

    def get_available_models(self) -> Dict[str, Dict]:
        return self.models
    
    
def embed_query(query_text):
    """
    Embed the same query using both SBERT and BGE.

    Args:
        query_text: User query string

    Returns:
        Dictionary with embeddings from both models
    """

    # SBERT embedding (384 dims)
    sbert_embedding = sbert_model.encode(query_text)

    # BGE embedding (1024 dims)
    bge_embedding = bge_model.encode(query_text)

    return {
        'sbert': sbert_embedding,
        'bge': bge_embedding
    }

def classify_hotel_intent(query: str) -> str:
    prompt = HOTEL_INTENT_PROMPT.format(query=query)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10
    )

    return response.choices[0].message.content.strip()