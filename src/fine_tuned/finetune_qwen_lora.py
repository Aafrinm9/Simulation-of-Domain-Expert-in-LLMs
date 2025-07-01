import logging
import os
import torch
import boto3
import json
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, pipeline
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, PeftModel

# Suppress warnings
logging.getLogger("agentic_modeling_classes").disabled = True
logging.getLogger("boto3").setLevel(logging.CRITICAL)
logging.getLogger("botocore").setLevel(logging.CRITICAL)
logging.getLogger("s3transfer").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Configure logging with reduced verbosity
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set specific loggers to WARNING or CRITICAL
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.CRITICAL)
logging.getLogger("botocore").setLevel(logging.CRITICAL)
logging.getLogger("s3transfer").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Set device - optimized for CPU
device = "cpu"
logger.info(f"Using device: {device}")

# Load JSON files from S3
bucket_name = 'finetune-train'
prefix = ''

logger.info(f"Loading data from S3 bucket: {bucket_name}")
s3_client = boto3.client('s3')
response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

# Load and normalize data
all_examples = []

for item in response.get('Contents', []):
    if item['Key'].endswith('.json'):
        logger.info(f"Processing file: {item['Key']}")
        try:
            # Download the JSON file
            file_obj = s3_client.get_object(Bucket=bucket_name, Key=item['Key'])
            file_content = file_obj['Body'].read().decode('utf-8')
            data = json.loads(file_content)
            
            # Extract the document and conversation fields
            document = data.get("document", "")
            conversations = data.get("conversation", [])
            
            all_examples.append({
                "document": document,
                "conversation": conversations
            })
            
            logger.info(f"Successfully processed file: {item['Key']}")
        except Exception as e:
            logger.error(f"Error processing file {item['Key']}: {e}")

logger.info(f"Loaded {len(all_examples)} examples from JSON files")

# Create a Dataset object
dataset = Dataset.from_list(all_examples)

# Enhanced preprocessing
def preprocess_data(example):
    document_context = example.get("document", "")
    conversations = example["conversation"]
    
    # Find question-response pairs
    pairs = []
    for i in range(0, len(conversations) - 1):
        if conversations[i]["role"] == "interviewer" and conversations[i+1]["role"] == "employee":
            question = conversations[i]
            response = conversations[i+1]
            
            question_quality = question.get("quality", "")
            response_quality = response.get("quality", "")
            follow_up_needed = response.get("follow_up_needed", False)
            
            pairs.append({
                "question": question["content"],
                "question_quality": question_quality,
                "response": response["content"],
                "response_quality": response_quality,
                "follow_up_needed": follow_up_needed,
                "follow_up_strategy": response.get("follow_up_strategy", {})
            })
    
    # Create training examples
    examples = []
    for pair in pairs:
        if pair["question_quality"] == "high":
            examples.append({
                "context": document_context,
                "question": pair["question"],
                "is_good_question": True,
                "follow_up_needed": pair["follow_up_needed"]
            })
    
    return examples

# Process the dataset
processed_examples = []
for example in dataset:
    processed = preprocess_data(example)
    processed_examples.extend(processed)

# Create dataset from processed examples
processed_dataset = Dataset.from_list(processed_examples)
processed_dataset = processed_dataset.filter(lambda example: example["context"] != "")

# Log dataset statistics
logger.info(f"Processed dataset contains {len(processed_dataset)} examples")
logger.info(f"Sample processed example: {processed_dataset[0] if len(processed_dataset) > 0 else 'No examples found'}")

# Create train/validation split
split_dataset = processed_dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]
logger.info(f"Split dataset: {len(train_dataset)} training examples, {len(val_dataset)} validation examples")

# Initialize tokenizer
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
logger.info(f"Loading tokenizer from {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    # The data has 'question' column, not 'conversation'
    text = example["question"]
    
    # Tokenize the question text
    tokenized = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512
    )
    
    # Create labels (copy of input_ids for language modeling)
    tokenized['labels'] = tokenized['input_ids'].copy()
    
    return tokenized

# Tokenize datasets (removed duplicate code)
logger.info("Tokenizing datasets...")
train_tokenized = train_dataset.map(tokenize, remove_columns=train_dataset.column_names)
val_tokenized = val_dataset.map(tokenize, remove_columns=val_dataset.column_names)

# Set format for PyTorch
train_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Define evaluation metrics
def compute_metrics(eval_preds):
    loss_array = eval_preds[0]
    loss_value = float(np.mean(loss_array))
    return {"loss": loss_value}

# Standard LoRA configuration
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # Using full precision for CPU
    trust_remote_code=True
)
# Make sure the base model is in training mode
base_model.train()

lora_config = LoraConfig(
    r=8,  # Reduced rank for CPU efficiency
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Get PEFT model
logger.info("Applying LoRA adapter to base model")
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# Make sure the model is in training mode
model.train()

# Define output directories
output_dir = "./qwen-0.5b-lora-finetuned-5epochs"
final_model_dir = "./final-qwen-0.5b-lora-5epochs"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(final_model_dir, exist_ok=True)

# Set training arguments for CPU optimization
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,  # Small batch size for CPU
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,  # Simulate larger batch size
    evaluation_strategy="steps",
    eval_steps=50,
    num_train_epochs=5,
    learning_rate=2e-5,  # Slightly higher for better convergence
    warmup_steps=50,
    weight_decay=0.01,
    save_steps=100,
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    gradient_checkpointing=False,
    report_to="none",
    optim="adamw_hf", # CPU-optimized optimizer
    remove_unused_columns=False,  # Important for PEFT
    gradient_checkpointing_kwargs={"use_reentrant": False} 
)


# Initialize trainer
logger.info("Initializing Trainer")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    compute_metrics=compute_metrics,
)

# Fine-tune the model
logger.info("Starting training...")
trainer.train()

# Save training results
training_results = {
    "final_loss": trainer.state.log_history[-1].get("loss", None),
    "best_eval_loss": trainer.state.best_metric,
    "total_steps": trainer.state.global_step,
    "optimizer": "LoRA"
}

with open(os.path.join(output_dir, "training_results.json"), "w") as f:
    json.dump(training_results, f, indent=4)

# Save the adapter weights
logger.info("Saving LoRA adapter weights...")
adapter_save_path = "./qwen-0.5b-lora-adapter-5epochs"
model.save_pretrained(adapter_save_path)
tokenizer.save_pretrained(adapter_save_path)

# Save the complete merged model
logger.info("Saving the merged model (base + adapter)...")

# Load fresh base model for merging
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": device},
    torch_dtype=torch.float32,
    trust_remote_code=True
)

# Load the saved adapter
merged_model = PeftModel.from_pretrained(base_model, adapter_save_path)

# Merge weights
merged_model = merged_model.merge_and_unload()

# Save the complete merged model
merged_model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)

# Save model state dict
logger.info("Saving model state dict for agentic modeling...")
torch.save(merged_model.state_dict(), f"{final_model_dir}/model_weights.pt")

# Test the model
logger.info("Testing model with example prompts...")

# Ensure the model is on the correct device (MPS for Apple Silicon)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
merged_model = merged_model.to(device)

test_contexts = [
    "This document outlines the Data Analyst role in Telecommunications, focusing on data strategy, architecture, and stewardship responsibilities.",
    "This document describes the handover process for the Customer Journey Analytics Enhancement project led by Jessica Miller."
]

for context in test_contexts:
    messages = [
        {"role": "user", "content": f"Based on the following document, generate a high-quality question to extract missing information:\n\n{context}"}
    ]
    
    # Apply chat template and immediately put on correct device
    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    
    # Create attention mask for the tokenized input
    attention_mask = torch.ones_like(tokenized)
    
    # Move both to the correct device
    tokenized = tokenized.to(device)
    attention_mask = attention_mask.to(device)
    
    logger.info(f"\nTesting with context:\n{context}")
    
    # Generate response with proper attention mask
    outputs = merged_model.generate(
        input_ids=tokenized,
        attention_mask=attention_mask,  # Include attention mask
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2
    )
    
    # Decode and extract only the generated part
    generated = tokenizer.decode(outputs[0][tokenized.shape[1]:], skip_special_tokens=True)
    
    # Validate the generated question
    if not generated.strip().endswith("?"):
        logger.warning(f"Generated output is not a proper question: {generated}")
    
    logger.info(f"Generated question: {generated}\n")

logger.info("LoRA fine-tuning and testing complete!")
logger.info(f"Training results saved to: {output_dir}/training_results.json")
logger.info(f"LoRA adapter: {adapter_save_path}")
logger.info(f"Complete merged model: {final_model_dir}")
logger.info(f"Model state dict: {final_model_dir}/model_weights.pt")