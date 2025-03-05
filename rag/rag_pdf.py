import fitz  # PyMuPDF
import ollama
import numpy as np

# Load the dataset from PDF
def load_pdf_dataset(pdf_path):
    dataset = []
    doc = fitz.open(pdf_path)
    
    # Different strategies for breaking down PDF content
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        
        # Optional: Split long pages into smaller chunks
        # This helps with more granular embedding and retrieval
        chunk_size = 1000  # characters per chunk
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        dataset.extend(chunks)
    
    doc.close()
    return dataset

# Embedding and Retrieval Setup
EMBEDDING_MODEL = 'nomic-embed-text'
LANGUAGE_MODEL = 'mistral'

# Vector Database
VECTOR_DB = []

def add_chunk_to_database(chunk):
    try:
        # Generate embedding for each chunk
        embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
        VECTOR_DB.append((chunk, embedding))
    except Exception as e:
        print(f"Error embedding chunk: {e}")

def cosine_similarity(a, b):
    # More robust cosine similarity using NumPy
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(query, top_n=3):
    try:
        # Get query embedding
        query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
        
        # Calculate similarities
        similarities = [
            (chunk, cosine_similarity(query_embedding, embedding)) 
            for chunk, embedding in VECTOR_DB
        ]
        
        # Sort by similarity, descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]
    
    except Exception as e:
        print(f"Retrieval error: {e}")
        return []

def main():
    # Path to your PDF about smart contract vulnerabilities
    pdf_path = r"" #PATH
    
    # Load dataset from PDF
    dataset = load_pdf_dataset(pdf_path)
    print(f'Loaded {len(dataset)} chunks')
    
    # Populate vector database
    for i, chunk in enumerate(dataset):
        add_chunk_to_database(chunk)
        print(f'Added chunk {i+1}/{len(dataset)} to the database')
    
    # Interactive Retrieval and Chatbot
    while True:
        input_query = input('Ask a question about smart contract vulnerabilities (or type "exit"): ')
        
        if input_query.lower() == 'exit':
            break
        
        # Retrieve relevant chunks
        retrieved_knowledge = retrieve(input_query)
        
        if not retrieved_knowledge:
            print("No relevant information found.")
            continue
        
        # Prepare context for language model
        instruction_prompt = f'''You are an expert in smart contract security. 
        Use only the following pieces of context to answer the question. 
        Do not make up any new information:
        
        {'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
        '''
        
        # Generate response
        try:
            stream = ollama.chat(
                model=LANGUAGE_MODEL,
                messages=[
                    {'role': 'system', 'content': instruction_prompt},
                    {'role': 'user', 'content': input_query},
                ],
                stream=True,
            )
            
            print('\nChatbot response:')
            for chunk in stream:
                print(chunk['message']['content'], end='', flush=True)
            print('\n')
        
        except Exception as e:
            print(f"Error generating response: {e}")

if __name__ == "__main__":
    main()