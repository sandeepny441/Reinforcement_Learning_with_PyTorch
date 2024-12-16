from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime, timedelta

@dataclass
class CustomerProfile:
    customer_id: str
    lifestyle_segments: List[str]  # e.g., ['health_conscious', 'busy_parent', 'organic_preferring']
    brand_preferences: Dict[str, float]  # brand_name: affinity_score
    purchase_history: List[Dict]  # List of purchase transactions
    rfm_scores: Dict[str, float]  # Recency, Frequency, Monetary scores

@dataclass
class Product:
    product_id: str
    name: str
    category: str
    brand: str
    price: float
    margin: float
    lifestyle_tags: List[str]

class PreferenceBasedRecommender:
    def __init__(self):
        self.lifestyle_analyzer = self._init_lifestyle_analyzer()
        self.brand_analyzer = self._init_brand_analyzer()
        self.purchase_pattern_analyzer = self._init_purchase_pattern_analyzer()
        self.rfm_analyzer = self._init_rfm_analyzer()
        self.revenue_optimizer = self._init_revenue_optimizer()
        
    def _init_lifestyle_analyzer(self):
        """
        Analyzes customer lifestyle preferences based on:
        - Purchase categories (organic, conventional, prepared meals)
        - Shopping frequency patterns
        - Price sensitivity
        - Product category affinities
        """
        return LifestyleAnalyzer()
        
    def _init_brand_analyzer(self):
        """
        Tracks and predicts brand preferences through:
        - Brand purchase history
        - Brand switching patterns
        - Price elasticity per brand
        - Brand loyalty scores
        """
        return BrandAnalyzer()
        
    def _init_purchase_pattern_analyzer(self):
        """
        Identifies purchase patterns including:
        - Category purchase cycles
        - Basket combinations
        - Seasonal preferences
        - Price point preferences
        """
        return PurchasePatternAnalyzer()
        
    def _init_rfm_analyzer(self):
        """
        Calculates and maintains RFM metrics:
        - Recency: Days since last purchase
        - Frequency: Number of purchases in period
        - Monetary: Total spend in period
        """
        return RFMAnalyzer()
        
    def _init_revenue_optimizer(self):
        """
        Optimizes recommendations for revenue/profit:
        - Product margin analysis
        - Cross-sell opportunity scoring
        - Upsell potential calculation
        - Bundle profitability assessment
        """
        return RevenueOptimizer()

    def generate_recommendations(self, customer_id: str, n_recommendations: int = 5) -> List[Product]:
        """
        Generates optimized recommendations balancing customer preferences and business metrics.
        
        Args:
            customer_id: Unique identifier for the customer
            n_recommendations: Number of recommendations to generate
            
        Returns:
            List of recommended products
        """
        # Get customer profile
        customer = self._get_customer_profile(customer_id)
        
        # Generate base recommendations from each analyzer
        lifestyle_recs = self.lifestyle_analyzer.get_recommendations(customer)
        brand_recs = self.brand_analyzer.get_recommendations(customer)
        pattern_recs = self.purchase_pattern_analyzer.get_recommendations(customer)
        rfm_recs = self.rfm_analyzer.get_recommendations(customer)
        
        # Optimize for revenue and profit
        optimized_recs = self.revenue_optimizer.optimize_recommendations(
            customer=customer,
            lifestyle_recs=lifestyle_recs,
            brand_recs=brand_recs,
            pattern_recs=pattern_recs,
            rfm_recs=rfm_recs,
            n_recommendations=n_recommendations
        )
        
        return optimized_recs
    
    def _calculate_recommendation_score(self, product: Product, customer: CustomerProfile) -> float:
        """
        Calculates a composite score for a product recommendation based on:
        - Lifestyle match score (30%)
        - Brand preference score (25%)
        - Purchase pattern fit (20%)
        - RFM-based relevance (15%)
        - Profit potential (10%)
        
        Returns:
            Float between 0 and 1 representing recommendation strength
        """
        lifestyle_score = self.lifestyle_analyzer.calculate_match_score(product, customer)
        brand_score = self.brand_analyzer.calculate_preference_score(product, customer)
        pattern_score = self.purchase_pattern_analyzer.calculate_fit_score(product, customer)
        rfm_score = self.rfm_analyzer.calculate_relevance_score(product, customer)
        profit_score = self.revenue_optimizer.calculate_profit_potential(product, customer)
        
        weighted_score = (
            0.30 * lifestyle_score +
            0.25 * brand_score +
            0.20 * pattern_score +
            0.15 * rfm_score +
            0.10 * profit_score
        )
        
        return weighted_score

class MetricsTracker:
    """
    Tracks and reports on key business metrics for the recommendation system
    """
    def __init__(self):
        self.metrics = {
            'gross_revenue': [],
            'gross_profit': [],
            'recommendation_acceptance_rate': [],
            'customer_lifetime_value': [],
            'average_order_value': []
        }
    
    def update_metrics(self, 
                      customer_id: str,
                      recommended_products: List[Product],
                      purchased_products: List[Product]):
        """
        Updates metrics based on customer interactions with recommendations
        """
        revenue = sum(p.price for p in purchased_products)
        profit = sum(p.price * p.margin for p in purchased_products)
        acceptance_rate = len(set(p.product_id for p in purchased_products) & 
                            set(p.product_id for p in recommended_products)) / len(recommended_products)
        
        self.metrics['gross_revenue'].append(revenue)
        self.metrics['gross_profit'].append(profit)
        self.metrics['recommendation_acceptance_rate'].append(acceptance_rate)
        
    def get_metrics_report(self) -> Dict[str, float]:
        """
        Generates a report of current metrics
        """
        return {
            'avg_gross_revenue': np.mean(self.metrics['gross_revenue']),
            'avg_gross_profit': np.mean(self.metrics['gross_profit']),
            'avg_acceptance_rate': np.mean(self.metrics['recommendation_acceptance_rate'])
        }