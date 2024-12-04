import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
import plotly.express as px
import numpy as np

# Page configuration
st.set_page_config(page_title="Product Embeddings Dashboard", layout="wide")
st.title("Product Embeddings Visualization")

# Initialize BERT model and tokenizer
@st.cache_resource
def load_bert_model():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_bert_model()

# List of products
products = ["milk", "Netflix", "cheese", "butter", "ghee", "apples", 
           "sketches", "Spotify", "Audio", "Radio", "Headphones"]

# Create sidebar for product selection
st.sidebar.header("Product Selection")
selected_product = st.sidebar.selectbox("Choose a product:", products)

# Function to get embeddings
@st.cache_data
def get_embeddings(words):
    word_vectors = []
    with torch.no_grad():
        for word in words:
            inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs)
            # Get the embedding from the last hidden state
            embedding = outputs.last_hidden_state[0][0].numpy()
            word_vectors.append(embedding)
    return np.array(word_vectors)

# Get embeddings and perform PCA
word_vectors = get_embeddings(products)
pca = PCA(n_components=2)
word_vectors_2d = pca.fit_transform(word_vectors)

# Create DataFrame for visualization
import pandas as pd
df = pd.DataFrame({
    "Product": products,
    "Dimension_1": word_vectors_2d[:, 0],
    "Dimension_2": word_vectors_2d[:, 1],
    "Selected": [p == selected_product for p in products]
})

# Create main visualization
st.subheader("2D Product Embedding Visualization")

# Create scatter plot with Plotly
fig = px.scatter(
    df,
    x="Dimension_1",
    y="Dimension_2",
    text="Product",
    color="Selected",
    color_discrete_map={True: "#1f77b4", False: "#7f7f7f"},
    title="Product Embeddings in 2D Space",
    labels={"Dimension_1": "First Principal Component", 
            "Dimension_2": "Second Principal Component"}
)

# Update layout and traces
fig.update_traces(
    textposition="top center",
    marker=dict(size=12),
    showlegend=False
)

fig.update_layout(
    height=600,
    hovermode='closest',
    paper_bgcolor='white',
    plot_bgcolor='white',
    xaxis=dict(gridcolor='lightgray'),
    yaxis=dict(gridcolor='lightgray')
)

# Display the plot
st.plotly_chart(fig, use_container_width=True)

# Add explanation section
st.markdown("---")
st.subheader("About this Visualization")
st.markdown("""
BERT embeddings
""")

# Add metrics and analysis
st.markdown("---")
st.subheader("Selected Product Analysis")

# Calculate distances from selected product to all other products
selected_idx = products.index(selected_product)
selected_vector = word_vectors[selected_idx]

distances = []
for idx, vector in enumerate(word_vectors):
    if idx != selected_idx:
        distance = np.linalg.norm(selected_vector - vector)
        distances.append((products[idx], distance))

# Sort by distance
distances.sort(key=lambda x: x[1])

# Display closest and furthest products
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Most Similar Products")
    for product, dist in distances[:3]:
        st.write(f"- {product} (Distance: {dist:.2f})")

with col2:
    st.markdown("### Least Similar Products")
    for product, dist in distances[-3:]:
        st.write(f"- {product} (Distance: {dist:.2f})")