from langchain_ollama import OllamaEmbeddings


# Create an embedding-model interface.
# Ollama is running locally and serves the model.
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

text = "Employees in Germany receive 30 days of annual leave."

# embed_query() converts one piece of text into one numerical vector.
vector = embeddings.embed_query(text)

print("Original text:")
print(text)

print("\nEmbedding dimension:")
print(len(vector))

print("\nFirst 10 embedding values:")
print(vector[:10])