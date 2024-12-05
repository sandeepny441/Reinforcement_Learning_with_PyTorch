from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.decomposition import PCA
import numpy as np
import plotly.express as px

# Load the tokenizer and model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Create contextual sentences for each product
product_contexts = {
    "milk": "I bought fresh milk from the dairy store",
    "cheese": "The dairy shop sells cheese made from milk",
    "butter": "Butter is a dairy product made from cream",
    "ghee": "Ghee is clarified butter used in cooking",
    "apples": "The store has fresh apples in the produce section",
    "shoes": "I need to buy new shoes from the footwear store",
    "wheat": "The farmer grows wheat in the field",
    "eggs": "Fresh eggs are available in the dairy section",
    "carrot": "Carrots are orange vegetables in the produce section"
}

# Function to get contextual embedding
def get_contextual_embedding(sentence):
    # Tokenize and get model outputs
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the embeddings of the entire sequence
    embeddings = outputs.last_hidden_state[0]
    
    # Calculate attention mask to properly average tokens
    attention_mask = inputs['attention_mask'][0]
    
    # Average all token embeddings (excluding [CLS] and [SEP])
    mask = attention_mask.bool()
    masked_embeddings = embeddings[mask]
    
    # Return mean of all token embeddings
    return torch.mean(masked_embeddings, dim=0).numpy()

# Get embeddings for all products
products_available = ["milk", "cheese", "butter", "ghee", "apples", "shoes"]
products_purchased = ["butter", "wheat", "eggs", "carrot", "apples"]

# Combine all unique products
all_products = list(set(products_available + products_purchased))
categories = ["Both" if (word in products_available and word in products_purchased)
             else "Available" if word in products_available
             else "Purchased" for word in all_products]

# Get contextual embeddings for each product
word_vectors = []
for word in all_products:
    embedding = get_contextual_embedding(product_contexts[word])
    word_vectors.append(embedding)

# Convert to numpy array
word_vectors = np.array(word_vectors)

# Perform PCA to reduce dimensionality to 2D
pca = PCA(n_components=2)
word_vectors_2d = pca.fit_transform(word_vectors)

# Prepare data for visualization
data = {
    "Word": all_products,
    "Category": categories,
    "X": word_vectors_2d[:, 0],
    "Y": word_vectors_2d[:, 1],
}

# # Create a 2D scatter plot using Plotly
# fig = px.scatter(
#     x=data["X"],
#     y=data["Y"],
#     text=data["Word"],
#     color=data["Category"],
#     title="2D Embedding Visualization for Products (Contextual)",
#     labels={"x": "Dimension 1", "y": "Dimension 2"},
# )

# fig.update_traces(textposition="top center")
# fig.update_layout(
#     xaxis=dict(title="Dimension 1"),
#     yaxis=dict(title="Dimension 2"),
#     showlegend=True,
#     width=800,
#     height=600,
# )

# fig.show()


import pandas as pd

# Prepare data for visualization as a DataFrame
data = pd.DataFrame({
    "Word": all_products,
    "Category": categories,
    "X": word_vectors_2d[:, 0],
    "Y": word_vectors_2d[:, 1],
})

# Create a 2D scatter plot using Plotly
fig = px.scatter(
    data,
    x="X",
    y="Y",
    text="Word",
    color="Category",  # Color points by category
    title="2D Embedding Visualization for Products (Contextual)",
    labels={"X": "Dimension 1", "Y": "Dimension 2"},
)

# Add circles around each point
for i, row in data.iterrows():
    fig.add_shape(
        type="circle",
        xref="x",
        yref="y",
        x0=row["X"] - 3,  # Adjust radius as needed
        y0=row["Y"] - 3,
        x1=row["X"] + 3,
        y1=row["Y"] + 3,
        line=dict(color="rgba(0,0,0,0.3)", width=1),  # Light transparent circle
    )

# Update text and layout
fig.update_traces(textposition="top center")
fig.update_layout(
    xaxis=dict(title="Dimension 1"),
    yaxis=dict(title="Dimension 2"),
    showlegend=True,
    width=800,
    height=600,
)

fig.show()