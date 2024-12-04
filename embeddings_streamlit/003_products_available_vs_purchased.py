from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
import plotly.express as px

# Load the tokenizer and model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Words to process
products_available = ["milk", "cheese", "butter", "ghee", "apples", "shoes"]
products_purchased = ["butter", "wheat", "eggs", "carrot", "apples"]

# Combine all unique words and assign categories
all_products = list(set(products_available + products_purchased))
categories = ["Available" if word in products_available else "Purchased" for word in all_products]

# Tokenize and get embeddings for each word
word_vectors = []
for word in all_products:
    inputs = tokenizer(word, return_tensors="pt")
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state[0][0].detach().numpy()  # Shape: (768,)
    word_vectors.append(embedding)

# Perform PCA to reduce dimensionality to 2D
pca = PCA(n_components=2)
word_vectors_2d = pca.fit_transform(word_vectors)  # Shape: (len(all_products), 2)

# Prepare data for visualization
data = {
    "Word": all_products,
    "Category": categories,
    "X": word_vectors_2d[:, 0],
    "Y": word_vectors_2d[:, 1],
}

# Create a 2D scatter plot using Plotly
fig = px.scatter(
    x=data["X"],
    y=data["Y"],
    text=data["Word"],
    color=data["Category"],
    title="2D Embedding Visualization for Products",
    labels={"x": "Dimension 1", "y": "Dimension 2"},
)

fig.update_traces(textposition="top center")
fig.update_layout(
    xaxis=dict(title="Dimension 1"),
    yaxis=dict(title="Dimension 2"),
    showlegend=True,
    width=800,
    height=600,
)

fig.show()
