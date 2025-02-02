import boto3
import numpy as np
import json

# Initialize Bedrock client
bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-west-2')

# Sample documents (in a real application, use a proper vector database)
documents = [
    "Amazon Bedrock is a fully managed service that makes base models from AI21 Labs, Anthropic, Cohere, Meta, Mistral AI, and Amazon accessible via an API.",
    "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation.",
    "AWS offers various AI services including Amazon SageMaker for machine learning workflows.",
    "Titan Embeddings model converts text into numerical representations for semantic search."
]

# Generate embeddings for documents using Amazon Titan Embed
def get_embedding(text):
    body = json.dumps({"inputText": text})
    response = bedrock.invoke_model(
        body=body,
        modelId="amazon.titan-embed-text-v1",
        accept="application/json",
        contentType="application/json"
    )
    return json.loads(response['body'].read())['embedding']

# Store document embeddings (in-memory for demo)
document_embeddings = [get_embedding(doc) for doc in documents]

# Calculate cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Retrieve relevant documents
def retrieve_documents(query, top_k=2):
    query_embedding = get_embedding(query)
    similarities = [cosine_similarity(query_embedding, doc_emb) for doc_emb in document_embeddings]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [documents[i] for i in top_indices]

# Generate response using Anthropic Claude
def generate_response(prompt):
    body = json.dumps({
        "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
        "max_tokens_to_sample": 500,
        "temperature": 0.5,
        "top_p": 0.9,
    })
    
    response = bedrock.invoke_model(
        body=body,
        modelId="anthropic.claude-v2",
        accept="application/json",
        contentType="application/json"
    )
    
    return json.loads(response['body'].read())['completion']

# RAG Pipeline
def rag_query(query):
    # Retrieve relevant context
    context = retrieve_documents(query)
    
    # Create augmented prompt
    prompt = f"""Use the following context to answer the question. If you don't know the answer, say so.

Context:
{'\n'.join(context)}

Question: {query}

Answer:"""
    
    # Generate response
    return generate_response(prompt)

# Example usage
if __name__ == "__main__":
    question = "What is Amazon Bedrock?"
    print("Question:", question)
    print("Answer:", rag_query(question))