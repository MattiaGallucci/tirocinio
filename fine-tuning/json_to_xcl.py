import pandas as pd

# Carica il dataset JSONL (ogni riga è un oggetto JSON)
df = pd.read_json("dataset.jsonl", lines=True)

# Visualizza le prime righe (opzionale)
print(df.head())

# Salva il DataFrame in un file Excel
df.to_excel("dataset.xlsx", index=False)

print("Dataset salvato in 'dataset.xlsx'")