import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
import plotly.express as px
import numpy as np

# Page configuration
st.set_page_config(page_title="Product Analysis Dashboard", layout="wide")
st.title("Product Analysis Dashboard")

# Create sample DataFrame
data = {
    'customer': [101, 101, 101, 102, 102, 102],
    'date': ['2 Dec', '2 Dec', '4 Dec', '10 Nov', '18 Nov', '10 Nov'],
    'product': ['apple', 'Banana', 'Cheese', 'Mango', 'Curry Leaves', 'Cream'],
    'category': ['Fruits', 'Fruits', 'Dairy', 'Fruits', 'Vegetables', 'Dairy']
}
df = pd.DataFrame(data)

# Initialize BERT model and tokenizer
@st.cache_resource
def load_bert_model():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_bert_model()

# Function to get embeddings
@st.cache_data
def get_embeddings(words):
    word_vectors = []
    with torch.no_grad():
        for word in words:
            inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[0][0].numpy()
            word_vectors.append(embedding)
    return np.array(word_vectors)

# Sidebar filters
st.sidebar.header("Filters")

# Category filter
categories = ['All'] + sorted(df['category'].unique().tolist())
selected_category = st.sidebar.radio("Select Category:", categories)

# Date filter
dates = ['All'] + sorted(df['date'].unique().tolist())
selected_date = st.sidebar.radio("Select Date:", dates)

# Customer filter
customers = ['All'] + sorted(df['customer'].unique().tolist())
selected_customer = st.sidebar.radio("Select Customer:", customers)

# Filter the DataFrame
filtered_df = df.copy()

if selected_category != 'All':
    filtered_df = filtered_df[filtered_df['category'] == selected_category]
if selected_date != 'All':
    filtered_df = filtered_df[filtered_df['date'] == selected_date]
if selected_customer != 'All':
    filtered_df = filtered_df[filtered_df['customer'] == selected_customer]

# Display filtered data
st.subheader("Filtered Data")
st.dataframe(filtered_df)

# Get embeddings for products
unique_products = filtered_df['product'].unique().tolist()

if len(unique_products) > 0:
    # Get embeddings and perform PCA
    word_vectors = get_embeddings(unique_products)
    
    if len(unique_products) > 1:
        pca = PCA(n_components=2)
        word_vectors_2d = pca.fit_transform(word_vectors)
        
        # Create DataFrame for visualization
        viz_df = pd.DataFrame({
            "Product": unique_products,
            "Dimension_1": word_vectors_2d[:, 0],
            "Dimension_2": word_vectors_2d[:, 1]
        })
        
        # Create main visualization
        st.subheader("Product Embedding Visualization")
        
        fig = px.scatter(
            viz_df,
            x="Dimension_1",
            y="Dimension_2",
            text="Product",
            title="Product Embeddings in 2D Space",
            labels={"Dimension_1": "First Principal Component", 
                   "Dimension_2": "Second Principal Component"}
        )
        
        # Update the scatter plot appearance
        fig.update_traces(
            marker=dict(
                size=12,
                color='#636EFA',
                symbol='circle'
            ),
            textposition="top center"
        )
        
        fig.update_layout(
            height=600,
            hovermode='closest',
            paper_bgcolor='white',
            plot_bgcolor='white',
            xaxis=dict(
                gridcolor='lightgray',
                zeroline=True,
                zerolinecolor='lightgray'
            ),
            yaxis=dict(
                gridcolor='lightgray',
                zeroline=True,
                zerolinecolor='lightgray'
            ),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add similarity analysis
        st.subheader("Product Similarity Analysis")
        
        # Calculate pairwise distances
        from scipy.spatial.distance import pdist, squareform
        distances = pdist(word_vectors)
        distance_matrix = squareform(distances)
        
        # Create similarity matrix visualization
        similarity_df = pd.DataFrame(
            distance_matrix,
            columns=unique_products,
            index=unique_products
        )
        
        fig_heatmap = px.imshow(
            similarity_df,
            labels=dict(x="Product", y="Product", color="Distance"),
            title="Product Similarity Matrix"
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
    else:
        st.warning("Need at least 2 products to create visualization.")
else:
    st.warning("No products found with the current filter settings.")

# Add metrics
st.markdown("---")
st.subheader("Summary Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Products", len(filtered_df))
    
with col2:
    st.metric("Unique Products", len(filtered_df['product'].unique()))
    
with col3:
    st.metric("Selected Category", selected_category)

# Add explanation
st.markdown("---")
st.subheader("About this Dashboard")
st.markdown("""
This dashboard provides analysis of product embeddings using BERT:
- Use the sidebar filters to select specific categories, dates, or customers
- The scatter plot shows semantic relationships between products based purely on their names
- Products that are closer together in the visualization are more semantically similar
- The similarity matrix shows pairwise distances between product embeddings
- All embeddings are generated using BERT and reduced to 2D using PCA
""")