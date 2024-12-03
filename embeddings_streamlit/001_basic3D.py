from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
import plotly.express as px

# Load the tokenizer and model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Words to process
words = ["milk", "cheese", "butter", "ghee", "apples", "sketches", "Spotify", "Audio", "Radio", "Headphones"]

# Tokenize and get embeddings for each word
word_vectors = []
for word in words:
    inputs = tokenizer(word, return_tensors="pt")
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state[0][0].detach().numpy()  # Shape: (768,)
    word_vectors.append(embedding)

# Perform PCA to reduce dimensionality to 3D
pca = PCA(n_components=3)
word_vectors_3d = pca.fit_transform(word_vectors)  # Shape: (10, 3)

# Prepare data for visualization
data = {
    "Word": words,
    "X": word_vectors_3d[:, 0],
    "Y": word_vectors_3d[:, 1],
    "Z": word_vectors_3d[:, 2],
}

# Create a 3D scatter plot using Plotly
fig = px.scatter_3d(
    x=data["X"],
    y=data["Y"],
    z=data["Z"],
    text=data["Word"],
    title="3D Embedding Visualization for Words",
    labels={"x": "Dimension 1", "y": "Dimension 2", "z": "Dimension 3"},
)

fig.update_traces(marker=dict(size=5), textposition="top center")
fig.update_layout(
    scene=dict(
        xaxis=dict(title="Dimension 1"),
        yaxis=dict(title="Dimension 2"),
        zaxis=dict(title="Dimension 3"),
    ),
    width=800,
    height=800,
)

fig.show()
