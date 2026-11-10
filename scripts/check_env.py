import sys

packages = ["langchain", "llama_index", "sentence_transformers", "sklearn", "matplotlib"]
print(f"Python: {sys.version}")
print("Checking core project and baseline dependencies:")
for pkg in packages:
    try:
        __import__(pkg)
        print(f"  {pkg}: AVAILABLE")
    except ImportError as e:
        print(f"  {pkg}: NOT INSTALLED ({e})")

print("Note: chromadb and faiss are not required (LocalVectorStore uses in-memory NumPy cosine similarity).")
