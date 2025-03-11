import fitz  # PyMuPDF
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
from datasets import load_dataset
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Models configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight embedding model
LANGUAGE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # Can be replaced with any HF model

class RAGSystem:
    def __init__(self, embedding_model=EMBEDDING_MODEL, language_model=LANGUAGE_MODEL):
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.tokenizer_embed = AutoTokenizer.from_pretrained(embedding_model)
        self.model_embed = AutoModel.from_pretrained(embedding_model).to(device)
        
        # Initialize language model for generation
        print(f"Loading language model: {language_model}")
        self.generator = pipeline(
            "text-generation",
            model=language_model,
            tokenizer=AutoTokenizer.from_pretrained(language_model),
            device_map="auto",  # Automatically use available devices
            max_length=2048
        )
        
        # Vector database to store embeddings
        self.vector_db = []
    
    def get_embedding(self, text):
        """Generate embeddings for a given text using the embedding model"""
        inputs = self.tokenizer_embed(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = self.model_embed(**inputs)
        
        # Use mean pooling to get a single vector representation
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings[0].cpu().numpy()  # Convert to numpy array
    
    def add_chunk_to_database(self, chunk):
        """Add a text chunk and its embedding to the vector database"""
        try:
            embedding = self.get_embedding(chunk)
            self.vector_db.append((chunk, embedding))
            return True
        except Exception as e:
            print(f"Error embedding chunk: {e}")
            return False
    
    def cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors"""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def retrieve(self, query, top_n=3):
        """Retrieve the most relevant chunks for a given query"""
        try:
            query_embedding = self.get_embedding(query)
            
            similarities = [
                (chunk, self.cosine_similarity(query_embedding, embedding)) 
                for chunk, embedding in self.vector_db
            ]
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_n]
        
        except Exception as e:
            print(f"Retrieval error: {e}")
            return []
    
    def generate_response(self, query, retrieved_chunks):
        """Generate a response based on the query and retrieved chunks"""
        if not retrieved_chunks:
            return "No relevant information found to answer your question."
        
        # Prepare context for language model
        context = "\n".join([f"- {chunk}" for chunk, _ in retrieved_chunks])
        
        prompt = f"""<s>[INST] You are an expert assistant. Use only the following pieces of context to answer the question. 
        If you don't know the answer based on the provided context, say so.
        
        Context:
        {context}
        
        Question: {query} [/INST]"""
        
        try:
            response = self.generator(
                prompt,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
            )
            
            # Extract the generated text
            answer = response[0]['generated_text'].split('[/INST]')[-1].strip()
            return answer
        
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"


# Data loading functions
def load_pdf_dataset(pdf_path, chunk_size=1000):
    """Load and chunk content from a PDF file"""
    dataset = []
    doc = fitz.open(pdf_path)
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        
        # Split long pages into smaller chunks
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        dataset.extend(chunks)
    
    doc.close()
    return dataset

def load_huggingface_dataset(dataset_name, subset=None, text_column="text", max_samples=None):
    """Load a dataset from Hugging Face Hub"""
    try:
        if subset:
            dataset = load_dataset(dataset_name, subset)
        else:
            dataset = load_dataset(dataset_name)
        
        # Get the first split (usually 'train')
        split_name = list(dataset.keys())[0]
        data = dataset[split_name]
        
        # Extract text from the specified column
        if text_column in data.features:
            texts = [item[text_column] for item in data]
            if max_samples:
                texts = texts[:max_samples]
            return texts
        else:
            raise ValueError(f"Column '{text_column}' not found in dataset. Available columns: {data.features.keys()}")
    
    except Exception as e:
        print(f"Error loading dataset from Hugging Face: {e}")
        return []

def main():
    # Initialize RAG system
    rag = RAGSystem()
    
    # Choose data source
    print("\nChoose a data source:")
    print("1. Local PDF file")
    print("2. Hugging Face dataset")
    choice = input("Enter your choice (1 or 2): ")
    
    dataset = []
    
    if choice == "1":
        pdf_path = input("Enter the path to your PDF file: ")
        if os.path.exists(pdf_path):
            dataset = load_pdf_dataset(pdf_path)
            print(f"Loaded {len(dataset)} chunks from PDF")
        else:
            print(f"File not found: {pdf_path}")
            return
    
    elif choice == "2":
        dataset_name = input("Enter Hugging Face dataset name (e.g., 'squad', 'glue'): ")
        subset = input("Enter subset name (optional, press Enter to skip): ") or None
        text_column = input("Enter text column name (default: 'text'): ") or "text"
        max_samples = input("Enter maximum number of samples to load (optional, press Enter for all): ")
        max_samples = int(max_samples) if max_samples.isdigit() else None
        
        dataset = load_huggingface_dataset(dataset_name, subset, text_column, max_samples)
        print(f"Loaded {len(dataset)} samples from Hugging Face")
    
    else:
        print("Invalid choice")
        return
    
    # Populate vector database
    print("\nBuilding vector database...")
    for i, chunk in enumerate(dataset):
        if i % 10 == 0:  # Progress update every 10 chunks
            print(f"Processing chunk {i+1}/{len(dataset)}")
        rag.add_chunk_to_database(chunk)
    
    print(f"Vector database built with {len(rag.vector_db)} entries")
    
    # Interactive Retrieval and Chatbot
    print("\n=== RAG System Ready ===")
    print("You can now ask questions about the loaded content")
    
    while True:
        query = input('\nEnter your question (or type "exit" to quit): ')
        
        if query.lower() == 'exit':
            break
        
        # Retrieve relevant chunks
        print("Retrieving relevant information...")
        retrieved_chunks = rag.retrieve(query)
        
        # Generate response
        print("Generating response...")
        response = rag.generate_response(query, retrieved_chunks)
        
        print("\nResponse:")
        print(response)
        
        # Optionally show retrieved chunks
        show_sources = input("\nShow sources? (y/n): ")
        if show_sources.lower() == 'y':
            print("\nRetrieved sources:")
            for i, (chunk, similarity) in enumerate(retrieved_chunks):
                print(f"\n--- Source {i+1} (Similarity: {similarity:.4f}) ---")
                print(chunk[:300] + "..." if len(chunk) > 300 else chunk)

if __name__ == "__main__":
    main()