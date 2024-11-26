import os
import math
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer

def evaluate_model(model, tokenizer, test_dataset):
    # Tokenize the test dataset
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=512,
            padding='max_length'
        )

    print("Tokenizing test dataset...")
    tokenized_test_dataset = test_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=['text']
    )

    # Set up DataLoader
    test_dataloader = DataLoader(
        tokenized_test_dataset,
        batch_size=1,
        shuffle=False
    )

    # Generate responses
    print("Generating responses...")
    model.eval()
    generated_responses = []
    reference_responses = []

    for batch in tqdm(test_dataloader, desc="Generating Responses"):
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=512,
                num_beams=5,
                no_repeat_ngram_size=3,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_responses.append(generated_text)

        # Extract reference response
        original_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        reference_response = original_text.strip().split('\n')[-1]
        reference_responses.append(reference_response)

    # Compute BLEU Score
    print("Computing BLEU score...")
    nltk.download('punkt', quiet=True)
    references = [[nltk.word_tokenize(resp.lower())] for resp in reference_responses]
    hypotheses = [nltk.word_tokenize(resp.lower()) for resp in generated_responses]
    bleu_score = corpus_bleu(references, hypotheses)
    print(f"BLEU Score: {bleu_score:.4f}")

    # Compute ROUGE Scores
    print("Computing ROUGE scores...")
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for ref, hyp in zip(reference_responses, generated_responses):
        scores = scorer.score(ref, hyp)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)

    print(f"ROUGE-1 F1 Score: {avg_rouge1:.4f}")
    print(f"ROUGE-2 F1 Score: {avg_rouge2:.4f}")
    print(f"ROUGE-L F1 Score: {avg_rougeL:.4f}")

    # Compute Perplexity
    print("Calculating Perplexity...")
    from torch.nn import CrossEntropyLoss
    loss_fct = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='sum')
    total_loss = 0
    total_tokens = 0

    for batch in tqdm(test_dataloader, desc="Calculating Perplexity"):
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

        batch_loss = loss.item() * input_ids.size(1)
        total_loss += batch_loss
        total_tokens += attention_mask.sum().item()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    print(f"Perplexity: {perplexity:.2f}")

def main():
    # Paths to the dataset files
    data_dir = 'C:\\Anas\'s Data\\Ubuntu_Corpus_V2'  # Update this path
    processed_test_file = os.path.join(data_dir, 'test_processed.txt')

    # Load the tokenizer and model
    print("Loading tokenizer and model...")
    model_path = './fine_tuned_gpt2'  # Path to your fine-tuned model
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path).to('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the test dataset
    print("Loading test dataset...")
    test_dataset = load_dataset('text', data_files={'test': processed_test_file})['test']

    # Evaluate the model
    evaluate_model(model, tokenizer, test_dataset)

if __name__ == '__main__':
    main()
