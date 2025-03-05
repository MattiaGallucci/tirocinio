import pandas as pd
import json

# Carica il dataset CSV, assicurandoti che non ci siano colonne indesiderate
df = pd.read_csv("output.csv", usecols=[0], names=["text"], skiprows=1)  # Ignora l'intestazione se presente

# Lista per memorizzare i dati formattati
data = []

# Itera sulle righe del DataFrame
for text_content in df["text"]:
    # Dividere il testo in blocchi tra [INST] e [/INST]
    parts = text_content.split("[INST]")
    
    for part in parts[1:]:  # Ignora la prima parte se non contiene istruzioni
        segments = part.split("[/INST]")
        if len(segments) < 2:
            continue
        
        instruction = segments[0].strip()  # Estrarre il prompt dell'utente
        response = segments[1].strip()  # Estrarre la risposta del modello

        data.append({
            "instruction": instruction,
            "input": "",
            "output": response
        })

# Salva il dataset formattato in JSONL
with open("dataset.jsonl", "w", encoding="utf-8") as f:
    for entry in data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print("Conversione completata! File salvato come dataset.jsonl")
