import pandas as pd
import json

# Carica il dataset CSV
df = pd.read_csv("output.csv", usecols=[0], names=["text"], skiprows=1)

data = []
current_entry = None

for text_content in df["text"]:
    parts = text_content.split("[INST]")
    
    for part in parts[1:]:  # Ignora la prima parte se non contiene istruzioni
        segments = part.split("[/INST]")
        if len(segments) < 2:
            continue
        
        instruction = segments[0].strip()  # Estrarre il prompt dell'utente
        response = segments[1].strip()  # Estrarre la risposta del modello

        # Se è una continuazione di una richiesta precedente, la uniamo
        if current_entry and not instruction.startswith("Can you check if"):
            current_entry["output"] += " " + response
        else:
            # Salviamo il vecchio entry e ne iniziamo uno nuovo
            if current_entry:
                data.append(current_entry)
            current_entry = {
                "instruction": instruction,
                "input": "",
                "output": response
            }

# Salva anche l'ultima entry
if current_entry:
    data.append(current_entry)

# Salva il dataset formattato in JSONL
with open("dataset.jsonl", "w", encoding="utf-8") as f:
    for entry in data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print("Conversione completata! File salvato come dataset.jsonl")
