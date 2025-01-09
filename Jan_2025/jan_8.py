import torch
from transformers import DPRContextEncoder, DPRQuestionEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer

# Load question and context encoders
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

# Tokenizers
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

# Example query and documents
query = "What is PyTorch?"
contexts = ["PyTorch is an open-source machine learning library.", 
            "TensorFlow is another popular library."]

# Encode query and contexts
query_embedding = question_encoder(**question_tokenizer(query, return_tensors='pt')).pooler_output
context_embeddings = torch.stack([
    context_encoder(**context_tokenizer(ctx, return_tensors='pt')).pooler_output for ctx in contexts
])

# Find most relevant context
scores = torch.matmul(query_embedding, context_embeddings.T)
top_context_idx = scores.argmax().item()
print("Top Context:", contexts[top_context_idx])
