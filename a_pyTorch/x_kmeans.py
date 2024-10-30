import streamlit as st
import pandas as pd

# Sample grocery products and ratings (for demonstration)
products = pd.DataFrame({
    'product_id': [101, 102, 103, 104, 105],
    'product_name': ['Organic Apples', 'Almond Milk', 'Whole Wheat Bread', 'Greek Yogurt', 'Avocado'],
    'category': ['Fruits', 'Beverages', 'Bakery', 'Dairy', 'Fruits'],
    'average_rating': [4.6, 4.3, 4.7, 4.8, 4.5]
})

# Sample actual purchased products for a demonstration customer
actual_purchases = ['Organic Apples', 'Greek Yogurt', 'Avocado']

def recommend_products(selected_category):
    # Filter products based on selected category and sort by rating
    recommended_products = products[products['category'] == selected_category].sort_values(by='average_rating', ascending=False)
    return recommended_products

# Streamlit App
st.title("HEB Grocery Product Recommendations Prototype")

# Customer selects a product category
selected_category = st.selectbox("Choose a product category you like:", products['category'].unique())

# Show recommendations when the button is clicked
if st.button("Recommend Products"):
    recommendations = recommend_products(selected_category)

    # Display two columns for Recommendations and Actual Purchases
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Recommended Products")
        if not recommendations.empty:
            for index, row in recommendations.iterrows():
                st.write(f"{row['product_name']} - Rating: {row['average_rating']}")
        else:
            st.write("No recommendations available for this category.")

    with col2:
        st.header("Actual Products Purchased")
        for product in actual_purchases:
            st.write(product)
