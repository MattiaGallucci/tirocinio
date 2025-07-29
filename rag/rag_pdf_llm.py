# RAG System with Hugging Face Models
# For Google Colab with Hugging Face login

import fitz  # PyMuPDF
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from torch.nn.functional import cosine_similarity
import os
from tqdm.notebook import tqdm
from huggingface_hub import login
import getpass

# Login to Hugging Face
def hf_login():
    print("Please enter your Hugging Face token to access gated models.")
    print("You can find your token at https://huggingface.co/settings/tokens")
    token = getpass.getpass("Hugging Face Token: ")
    login(token=token)
    print("Successfully logged in to Hugging Face!")

# Load the dataset from PDF
def load_pdf_dataset(pdf_path):
    dataset = []
    doc = fitz.open(pdf_path)
    # Different strategies for breaking down PDF content
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        # Split long pages into smaller chunks
        chunk_size = 1000  # characters per chunk
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        dataset.extend(chunks)
    doc.close()
    return dataset

# Vector Database
VECTOR_DB = []

# Load models from Hugging Face - optimized for Colab
def load_models():
    print("Loading embedding model...")
    # Load embedding model (sentence-transformers model for embeddings)
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = AutoModel.from_pretrained(embedding_model_name)
    embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    
    print("Loading language model...")
    # Choose a smaller model that will fit in Colab's memory
    llm_model_name = "FraChiacc99/deep_seek_algorand"  # or try "google/flan-t5-base" for a smaller model #mistralai/Mistral-7B-Instruct-v0.2 #FraChiacc99/deep_seek_solana #FraChiacc99/deep_seek_algorand
    
    # Load tokenizer first
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    if llm_tokenizer.pad_token is None and llm_tokenizer.eos_token is not None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    
    # Import BitsAndBytesConfig
    from transformers import BitsAndBytesConfig
    
    # Load the model with optimizations
    llm_model = None
    try:
        # Try loading with 8-bit quantization using BitsAndBytesConfig
        print("Attempting to load model with 8-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=torch.float16
        )
    except Exception as e:
        print(f"8-bit loading failed: {e}")
        try:
            # Try with 4-bit quantization using BitsAndBytesConfig
            print("Attempting to load model with 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            
            llm_model = AutoModelForCausalLM.from_pretrained(
                llm_model_name,
                device_map="auto",
                quantization_config=quantization_config,
                torch_dtype=torch.float16
            )
        except Exception as e:
            print(f"4-bit loading failed: {e}")
            try:
                # Try without quantization but with half precision
                print("Attempting to load model with half precision...")
                llm_model = AutoModelForCausalLM.from_pretrained(
                    llm_model_name,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
            except Exception as e:
                print(f"Half precision loading failed: {e}")
                # Fall back to a much smaller model if all else fails
                print("Falling back to a smaller model...")
                llm_model_name = "gpt2"
                llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
                llm_model = AutoModelForCausalLM.from_pretrained(
                    llm_model_name,
                    device_map="auto"
                )
    
    print(f"Successfully loaded models. LLM: {llm_model_name}")
    return {
        "embedding_model": embedding_model,
        "embedding_tokenizer": embedding_tokenizer,
        "llm_model": llm_model,
        "llm_tokenizer": llm_tokenizer
    }
# Function to try different models if one fails
def try_alternative_models():
    # List of models to try in order of preference
    model_options = [
        "mistralai/Mistral-7B-Instruct-v0.2",
        "meta-llama/Llama-2-7b-chat-hf",
        "facebook/opt-2.7b",
        "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",  # Quantized version
        "gpt2-medium",  # Very small fallback
        "gpt2"  # Smallest fallback
    ]
    
    from transformers import BitsAndBytesConfig
    
    for model_name in model_options:
        print(f"Trying to load {model_name}...")
        try:
            
            if "t5" in model_name.lower():
                from transformers import T5ForConditionalGeneration
                model = T5ForConditionalGeneration.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
                return model, tokenizer, model_name
                
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                
            # Try different loading configurations
            try:
                # Use BitsAndBytesConfig instead of deprecated parameters
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16
                )
            except Exception as e:
                print(f"8-bit loading failed: {e}")
                try:
                    # Try with 4-bit quantization
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16
                    )
                    
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map="auto",
                        quantization_config=quantization_config,
                        torch_dtype=torch.float16
                    )
                except Exception as e:
                    print(f"4-bit loading failed: {e}")
                    # Try without quantization
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map="auto",
                        torch_dtype=torch.float16
                    )
            
            print(f"Successfully loaded {model_name}")
            return model, tokenizer, model_name
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
    
    # If all else fails, try the smallest model with no quantization
    try:
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print(f"Loaded fallback model {model_name}")
        return model, tokenizer, model_name
    except Exception as e:
        raise Exception(f"Could not load any language model: {e}")

# Generate embeddings using Hugging Face model
def generate_embedding(text, models):
    # Tokenize and get embedding
    tokenizer = models["embedding_tokenizer"]
    model = models["embedding_model"]

    # Add padding and truncation
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        # Use the mean of the last hidden state as the sentence embedding
        embeddings = outputs.last_hidden_state.mean(dim=1)

    return embeddings.squeeze().cpu().numpy()

def add_chunk_to_database(chunk, models):
    try:
        # Generate embedding for each chunk
        embedding = generate_embedding(chunk, models)
        VECTOR_DB.append((chunk, embedding))
        return True
    except Exception as e:
        print(f"Error embedding chunk: {e}")
        return False

def calculate_similarity(query_embedding, doc_embedding):
    query_embedding = torch.tensor(query_embedding)
    doc_embedding = torch.tensor(doc_embedding)
    return cosine_similarity(query_embedding.unsqueeze(0), doc_embedding.unsqueeze(0)).item()

def retrieve(query, models, top_n=3):
    try:
        # Get query embedding
        query_embedding = generate_embedding(query, models)

        # Calculate similarities
        similarities = [
            (chunk, calculate_similarity(query_embedding, embedding))
            for chunk, embedding in VECTOR_DB
        ]

        # Sort by similarity, descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]
    except Exception as e:
        print(f"Retrieval error: {e}")
        return []

def generate_response(query, context, models):
    try:
        tokenizer = models["llm_tokenizer"]
        model = models["llm_model"]

        # Prepare prompt with context and question
        prompt = f"""You are an expert in smart contract security.
Use only the following pieces of context to answer the question.
Do not make up any new information:

{context}

Question: {query}
Answer:"""

        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate response
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1000, #aumenta se necessario
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
        )

        # Decode the response, removing the input prompt
        input_length = inputs.input_ids.shape[1]
        generated_text = tokenizer.decode(generated_ids[0][input_length:], skip_special_tokens=True)
        return generated_text

    except Exception as e:
        print(f"Error generating response: {e}")
        return "I encountered an error while generating a response."

def main():
    print("Checking for GPU availability...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Login to Hugging Face
    hf_login()

    # Try loading models - first try standard approach, then fallback to alternatives if needed
    try:
        print("Loading models from Hugging Face...")
        models = load_models()
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models with standard approach: {e}")
        print("Trying alternative models...")
        llm_model, llm_tokenizer, llm_model_name = try_alternative_models()

        # Load embedding model separately
        embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embedding_model = AutoModel.from_pretrained(embedding_model_name)
        embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)

        models = {
            "embedding_model": embedding_model,
            "embedding_tokenizer": embedding_tokenizer,
            "llm_model": llm_model,
            "llm_tokenizer": llm_tokenizer
        }
        print(f"Successfully loaded alternative models. Using {llm_model_name}")

    # Upload and load the PDF file
    pdf_path = "./algorand_vulnerabilities.pdf"

    # Load dataset from PDF
    print("Loading dataset from PDF...")
    dataset = load_pdf_dataset(pdf_path)
    print(f'Loaded {len(dataset)} chunks')

    # Populate vector database
    print("Building vector database...")
    successful_chunks = 0
    for i, chunk in tqdm(enumerate(dataset), total=len(dataset), desc="Processing chunks"):
        if add_chunk_to_database(chunk, models):
            successful_chunks += 1

    print(f"Vector database built successfully with {successful_chunks} chunks!")

    # Interactive Retrieval and Chatbot
    print("\n=== Smart Contract Security Assistant ===")
    while True:
        input_query = input('\nAsk a question about smart contract vulnerabilities (or type "exit"): ')
        if input_query.lower() == 'exit':
            break

        # Retrieve relevant chunks
        print("Retrieving relevant information...")
        retrieved_knowledge = retrieve(input_query, models)
        if not retrieved_knowledge:
            print("No relevant information found.")
            continue

        # Prepare context for language model
        context = '\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])

        # Generate and print response
        print('Generating response...')
        response = generate_response(input_query, context, models)
        print(f"\nResponse:\n{response}")

if __name__ == "__main__":
    main()