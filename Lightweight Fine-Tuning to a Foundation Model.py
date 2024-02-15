#dataset URL:https://huggingface.co/datasets/poem_sentiment


import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM
from datasets import load_dataset
from peft import get_peft_model, AutoPeftModelForCausalLM, LoraConfig
from sklearn.model_selection import train_test_split

def tokenize_batch(tokenizer, batch, text_key='verse_text', label_key='label'):
    inputs = tokenizer(batch[text_key], return_tensors="pt", padding=True, truncation=True)
    inputs['labels'] = batch[label_key]
    return inputs

# Load an appropriate dataset
dataset_name = "poem_sentiment"  # Replaced with the actual dataset name
dataset = load_dataset(dataset_name)

# Assuming that the dataset has a 'train' key, update this if needed
train_dataset = dataset['train']

# Check the structure of your dataset and find the correct key for the text
print("Dataset Structure:", train_dataset.features)

# Use the correct key for the text in your dataset; tokenizing here
text_key = 'verse_text'  # Replace with the actual key for the text
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Set padding token to EOS token
tokenizer.pad_token = tokenizer.eos_token
train_inputs = tokenize_batch(tokenizer, train_dataset, text_key=text_key)

# Split the dataset into train and test sets
train_dataset, test_dataset = train_test_split(train_dataset, test_size=0.1, random_state=42)

# Tokenize and preprocess the datasets for training
train_inputs = tokenize_batch(tokenizer, train_dataset, text_key=text_key, label_key='label')  # Specify text_key and label_key here
test_inputs = tokenize_batch(tokenizer, test_dataset, text_key=text_key, label_key='label')  # Specify text_key and label_key here

# Load the Lora model
pretrained_model_name = "facebook/opt-350m"  # You can choose another model suitable for your task
lora_config = LoraConfig(
    r=16,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
    lora_alpha=32,
    lora_dropout=0.05,
    base_model_name_or_path=pretrained_model_name
)

# Load the pre-trained model
pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name)

# Use get_peft_model to apply LoRA
lora_model = get_peft_model(pretrained_model, lora_config)

# Print trainable parameters
print("Trainable Parameters Before Fine-Tuning:")
lora_model.print_trainable_parameters()

# Define training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    per_device_train_batch_size=8,
    num_train_epochs=1,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    save_total_limit=1,
    save_steps=500,
)

# Fine-tune the LoRA model
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_inputs,
    eval_dataset=test_inputs,
)
trainer.train()

# Save the fine-tuned LoRA model
lora_model.save_pretrained("./fine_tuned_model")

# Load the fine-tuned LoRA model for inference
fine_tuned_model = AutoPeftModelForCausalLM.from_pretrained("./fine_tuned_model")

# Print trainable parameters after fine-tuning
print("\nTrainable Parameters After Fine-Tuning:")
fine_tuned_model.print_trainable_parameters()

# Tokenize and evaluate with the fine-tuned model
input_text = test_dataset[0][text_key]  # Use any text example from your test dataset
inputs = tokenizer(input_text, return_tensors="pt")
outputs_fine_tuned = fine_tuned_model(**inputs)

# Printing fine tuning model
print("Fine-Tuned Model Output:", outputs_fine_tuned)
