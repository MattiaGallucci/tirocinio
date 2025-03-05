from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# 1. Carica il dataset JSONL
dataset = load_dataset("json", data_files="dataset.jsonl", split="train")

# 2. Scegli un modello pre-addestrato (qui usiamo GPT-2, ma puoi usare un altro modello)
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Aggiungi il token di fine riga se necessario (per GPT-2 solitamente non serve)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3. Funzione di preprocessamento
def preprocess_function(examples):
    texts = []
    for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
        # Costruisci il prompt in un formato chiaro
        # Puoi personalizzare questo formato in base alle tue necessità
        prompt = f"Instruction: {instruction}\n"
        if input_text.strip():
            prompt += f"Input: {input_text}\n"
        prompt += f"Output: {output}\n"
        texts.append(prompt)
    tokenized = tokenizer(texts, truncation=True, max_length=512, padding="max_length")
    return tokenized

# Applica il preprocessamento al dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

# 4. Configura il data collator (per il language modeling causale, mlm=False)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 5. Imposta gli argomenti di training
training_args = TrainingArguments(
    output_dir="./finetuned_model",
    overwrite_output_dir=True,
    per_device_train_batch_size=2,  # regola in base alla memoria disponibile
    num_train_epochs=3,             # numero di epoche
    logging_steps=10,
    save_steps=500,
    fp16=True,                      # se usi GPU con supporto per mixed precision
)

# 6. Inizializza il Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# 7. Avvia il fine-tuning
trainer.train()

# 8. Salva il modello fine-tuned
trainer.save_model("./finetuned_model")
