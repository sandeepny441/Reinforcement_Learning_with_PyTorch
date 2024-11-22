import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from collections import defaultdict
import numpy as np

class ProductRecommender:
    def __init__(self, min_support=0.01, min_confidence=0.3, min_lift=2):
        """
        Initialize the recommender with minimum thresholds for support, confidence, and lift
        
        Parameters:
        -----------
        min_support : float
            Minimum support threshold for frequent itemsets (default: 0.01 or 1%)
        min_confidence : float
            Minimum confidence threshold for rules (default: 0.3 or 30%)
        min_lift : float
            Minimum lift threshold for rules (default: 2)
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.rules = None
        self.frequent_itemsets = None
        self.product_mappings = None
        
    def prepare_data(self, transactions_df, order_id_col='order_id', product_id_col='product_id'):
        """
        Transform transaction data into one-hot encoded format
        
        Parameters:
        -----------
        transactions_df : pandas DataFrame
            DataFrame containing transaction data with order IDs and product IDs
        order_id_col : str
            Name of the column containing order IDs
        product_id_col : str
            Name of the column containing product IDs
        """
        # Create product mappings for later reference
        unique_products = transactions_df[product_id_col].unique()
        self.product_mappings = {
            'id_to_index': {pid: idx for idx, pid in enumerate(unique_products)},
            'index_to_id': {idx: pid for idx, pid in enumerate(unique_products)}
        }
        
        # Create pivot table (one-hot encoded format)
        return pd.crosstab(
            transactions_df[order_id_col],
            transactions_df[product_id_col]
        ).astype(bool)
    
    def fit(self, transactions_df, order_id_col='order_id', product_id_col='product_id'):
        """
        Fit the recommender on transaction data
        
        Parameters:
        -----------
        transactions_df : pandas DataFrame
            DataFrame containing transaction data
        """
        # Prepare data in correct format
        encoded_df = self.prepare_data(transactions_df, order_id_col, product_id_col)
        
        # Generate frequent itemsets
        self.frequent_itemsets = apriori(
            encoded_df,
            min_support=self.min_support,
            use_colnames=True
        )
        
        # Generate association rules
        self.rules = association_rules(
            self.frequent_itemsets,
            metric="confidence",
            min_threshold=self.min_confidence
        )
        
        # Filter rules by lift
        self.rules = self.rules[self.rules['lift'] >= self.min_lift]
        
        # Sort rules by lift and confidence
        self.rules = self.rules.sort_values(['lift', 'confidence'], ascending=[False, False])
        
        return self
    
    def get_recommendations(self, product_ids, n_recommendations=5):
        """
        Get product recommendations based on items in the cart
        
        Parameters:
        -----------
        product_ids : list
            List of product IDs currently in the cart
        n_recommendations : int
            Number of recommendations to return
            
        Returns:
        --------
        list : List of recommended product IDs with their confidence scores
        """
        if not self.rules is not None:
            raise ValueError("Model needs to be fit first!")
            
        # Convert product IDs to frozenset
        cart_items = frozenset(product_ids)
        
        # Find rules that match items in cart
        matching_rules = []
        for _, rule in self.rules.iterrows():
            if rule['antecedents'].issubset(cart_items):
                for item in rule['consequents']:
                    if item not in cart_items:  # Don't recommend items already in cart
                        matching_rules.append({
                            'product_id': item,
                            'confidence': rule['confidence'],
                            'lift': rule['lift']
                        })
        
        # Sort by confidence and lift
        matching_rules = sorted(
            matching_rules,
            key=lambda x: (x['confidence'], x['lift']),
            reverse=True
        )
        
        # Remove duplicates while maintaining order
        seen = set()
        unique_recommendations = []
        for rule in matching_rules:
            if rule['product_id'] not in seen:
                seen.add(rule['product_id'])
                unique_recommendations.append(rule)
                if len(unique_recommendations) >= n_recommendations:
                    break
                    
        return unique_recommendations
    
    def get_rule_metrics(self):
        """
        Get summary metrics about the generated rules
        
        Returns:
        --------
        dict : Dictionary containing rule metrics
        """
        if self.rules is None:
            raise ValueError("Model needs to be fit first!")
            
        return {
            'total_rules': len(self.rules),
            'avg_confidence': self.rules['confidence'].mean(),
            'avg_lift': self.rules['lift'].mean(),
            'max_lift': self.rules['lift'].max(),
            'min_lift': self.rules['lift'].min()
        }

# Example usage:
if __name__ == "__main__":
    # Sample transaction data
    data = {
        'order_id': [1, 1, 1, 2, 2, 3, 3, 3, 4],
        'product_id': ['A', 'B', 'C', 'A', 'C', 'B', 'C', 'D', 'B']
    }
    df = pd.DataFrame(data)
    
    # Initialize and fit recommender
    recommender = ProductRecommender(
        min_support=0.1,
        min_confidence=0.3,
        min_lift=1.5
    )
    recommender.fit(df)
    
    # Get recommendations for a cart with products ['A', 'B']
    recommendations = recommender.get_recommendations(['A', 'B'])
    print("Recommendations:", recommendations)
    
    # Get rule metrics
    metrics = recommender.get_rule_metrics()
    print("Rule Metrics:", metrics)