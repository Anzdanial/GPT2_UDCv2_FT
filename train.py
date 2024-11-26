import os
import math
import torch
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

def load_responses(responses_file):
    """
    Load responses from responses.txt into a dictionary.
    """
    responses = {}
    with open(responses_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                idx, response = parts
                responses[int(idx)] = response.strip()
    return responses

def preprocess_conversation(conversation):
    """
    Replace special tokens with appropriate separators.
    """
    conversation = conversation.replace('__eou__', '\n')
    conversation = conversation.replace('__eot__', '')
    conversation = conversation.strip()
    return conversation

def process_dataset(data_file, responses, output_file):
    """
    Process train.txt, valid.txt, or test.txt to extract conversation contexts and responses.
    """
    with open(data_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                index = parts[0]
                conversation = parts[1]
                first_entry = parts[2]
                second_entry = parts[3]

                # Determine if it's a positive sample
                if first_entry != 'NA':
                    response_idx = int(first_entry)
                elif second_entry != 'NA':
                    response_idx = int(second_entry)
                else:
                    continue  # Skip if no valid response index

                # Get the response text using the response index
                response_text = responses.get(response_idx, '')
                if not response_text:
                    continue  # Skip if response text is not found

                # Preprocess conversation and response
                conversation_text = preprocess_conversation(conversation)
                response_text = response_text.replace('__eou__', '\n').strip()

                # Combine conversation and response for GPT-2
                combined_text = f"{conversation_text}\n{response_text}\n<|endoftext|>\n"

                # Write the combined text to the output file
                f_out.write(combined_text)

def main():
    # Paths to the dataset files
    data_dir = 'C:\\Anas\'s Data\\Ubuntu_Corpus_V2'  # Update this path
    responses_file = os.path.join(data_dir, 'responses.txt')
    train_file = os.path.join(data_dir, 'train.txt')
    valid_file = os.path.join(data_dir, 'valid.txt')
    test_file = os.path.join(data_dir, 'test.txt')

    # Output files
    processed_train_file = os.path.join(data_dir, 'train_processed.txt')
    processed_valid_file = os.path.join(data_dir, 'valid_processed.txt')
    processed_test_file = os.path.join(data_dir, 'test_processed.txt')

    # Load responses into a dictionary
    print("Loading responses...")
    responses = load_responses(responses_file)
    print(f"Loaded {len(responses)} responses.")

    # Process train.txt
    print("Processing train.txt...")
    process_dataset(train_file, responses, processed_train_file)
    print(f"Preprocessed training data saved to {processed_train_file}")

    # Process valid.txt
    print("Processing valid.txt...")
    process_dataset(valid_file, responses, processed_valid_file)
    print(f"Preprocessed validation data saved to {processed_valid_file}")

    # Process test.txt
    print("Processing test.txt...")
    process_dataset(test_file, responses, processed_test_file)
    print(f"Preprocessed test data saved to {processed_test_file}")

    # Load the dataset
    data_files = {
        'train': processed_train_file,
        'validation': processed_valid_file
    }
    print("Loading datasets...")
    datasets = load_dataset('text', data_files=data_files)

    # Initialize the tokenizer
    print("Initializing tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

    # Initialize the model
    print("Initializing model...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))

    # Tokenize the datasets
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=512,
            padding='max_length'
        )

    print("Tokenizing datasets...")
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=['text']
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,  # Adjust based on your GPU memory
        per_device_eval_batch_size=2,
        evaluation_strategy='steps',
        eval_steps=500,
        save_steps=500,
        warmup_steps=200,
        logging_dir='./logs',
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        fp16=torch.cuda.is_available(),  # Enable FP16 if GPU is available
        dataloader_num_workers=4
    )

    # Initialize Trainer
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        data_collator=data_collator,
    )

    # Start training
    print("Starting training...")
    trainer.train()

    # Save the model
    print("Saving the model...")
    trainer.save_model('./fine_tuned_gpt2')
    tokenizer.save_pretrained('./fine_tuned_gpt2')
    print("Model saved to ./fine_tuned_gpt2")

    # Evaluate the model
    print("Evaluating the model...")
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

if __name__ == '__main__':
    main()
