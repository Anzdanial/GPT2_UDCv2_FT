# **Fine-Tuning GPT-2 on the Ubuntu Dialogue Corpus v2 (UDCv2)**

---

## **Table of Contents**

1. [Introduction](#introduction)
2. [Dataset Overview](#dataset-overview)
    - [Dataset Structure](#dataset-structure)
3. [Environment Setup](#environment-setup)
4. [Data Preprocessing](#data-preprocessing)
    - [Loading Responses](#loading-responses)
    - [Processing Conversation Data](#processing-conversation-data)
    - [Preprocessing Script](#preprocessing-script)
5. [Fine-Tuning GPT-2](#fine-tuning-gpt-2)
    - [Loading the Dataset](#loading-the-dataset)
    - [Tokenization](#tokenization)
    - [Model Configuration](#model-configuration)
    - [Training Arguments](#training-arguments)
    - [Training the Model](#training-the-model)
6. [Evaluating the Model](#evaluating-the-model)
    - [Evaluation Metrics](#evaluation-metrics)
    - [Generating Predictions](#generating-predictions)
    - [Calculating Metrics](#calculating-metrics)
7. [Conclusion](#conclusion)
8. [References](#references)

---

## **Introduction**

This guide provides a comprehensive walkthrough of fine-tuning the GPT-2 language model on the **Ubuntu Dialogue Corpus v2 (UDCv2)**. We'll cover everything from dataset understanding and preprocessing to model training and evaluation.

---
## **Dataset Overview**

The **Ubuntu Dialogue Corpus v2** is a large dataset of multi-turn dialogues extracted from Ubuntu chat logs. It's ideal for training conversational models, especially in technical support contexts.

### **Dataset Structure**

Your UDCv2 directory should contain the following files:

```
Ubuntu_Corpus_V2/
├── responses.txt
├── train.txt
├── valid.txt
└── test.txt

```

- **`responses.txt`**: Contains all unique responses.
- **`train.txt`**, **`valid.txt`**, **`test.txt`**: Contain conversation contexts and indices referencing `responses.txt`.

---

## **Environment Setup**

### **1. Install Required Libraries**

```
pip install transformers datasets torch nltk rouge_score
```

- **`transformers`**: For the GPT-2 model and tokenizers.
- **`datasets`**: For loading and processing datasets.
- **`torch`**: PyTorch for model training.
- **`nltk`**, **`rouge_score`**: For evaluation metrics.

### **2. Verify GPU Availability**

```
import torch

print(torch.cuda.is_available())

```

---

### **3. Update FilePath in evaluate.py and train.py**

```
data_dir = 'C:\\Anas\'s Data\\Ubuntu_Corpus_V2'  # Update this path

```

## **Data Preprocessing**

To fine-tune GPT-2, we need to preprocess the dataset appropriately.

### **Loading Responses**

Create a mapping from response indices to response texts.

```
def load_responses(responses_file):
    responses = {}
    with open(responses_file, 'r', encoding='utf-8') as f:
        for line in f:
            idx, response = line.strip().split('\t', 1)
            responses[int(idx)] = response.strip()
    return responses
```


### **Processing Conversation Data**

- **Replace Special Tokens**:
    - `__eou__` (End of Utterance) ➔ `\n` (newline)
    - `__eot__` (End of Turn) ➔ (optional handling)
- **Combine Conversation and Response**:
    - Concatenate conversation context and the correct response.
    - Add `<|endoftext|>` token at the end.

### **Preprocessing Script**

Below is the complete preprocessing script:


```
import os

def load_responses(responses_file):
    responses = {}
    with open(responses_file, 'r', encoding='utf-8') as f:
        for line in f:
            idx, response = line.strip().split('\t', 1)
            responses[int(idx)] = response.strip()
    return responses

def preprocess_conversation(conversation):
    # Replace '__eou__' with newline
    conversation = conversation.replace('__eou__', '\n')
    # Optionally remove '__eot__'
    conversation = conversation.replace('__eot__', '')
    return conversation.strip()

def process_dataset(data_file, responses, output_file):
    with open(data_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                index = parts[0]
                conversation = parts[1]
                first_entry = parts[2]
                second_entry = parts[3]

                # Identify positive samples
                if first_entry != 'NA':
                    response_idx = int(first_entry)
                elif second_entry != 'NA':
                    response_idx = int(second_entry)
                else:
                    continue

                response_text = responses.get(response_idx, '')
                if not response_text:
                    continue

                # Preprocess conversation and response
                conversation_text = preprocess_conversation(conversation)
                response_text = response_text.replace('__eou__', '\n').strip()

                # Combine and write to output
                combined_text = f"{conversation_text}\n{response_text}\n<|endoftext|>\n"
                f_out.write(combined_text)

```

---

## **Fine-Tuning GPT-2**

### **Loading the Dataset**

```
from datasets import load_dataset

data_files = {
    'train': 'train_processed.txt',
    'validation': 'valid_processed.txt'
}
datasets = load_dataset('text', data_files=data_files)

```

### **Tokenization**

```
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=512,
        padding='max_length'
    )

tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    num_proc=4,
    remove_columns=['text']
)

```

### **Model Configuration**

```
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

```

### **Training Arguments**

```
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
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
    fp16=True,  # Set to False if not using a compatible GPU
    dataloader_num_workers=4
)

```

### **Training the Model**

```
from transformers import Trainer, DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
)

trainer.train()

```

- **Monitoring Training**: Use TensorBoard to monitor training metrics.

---

## **Evaluating the Model**

### **Evaluation Metrics**

- **Perplexity**
- **BLEU Score**
- **ROUGE Scores**

### **Generating Predictions**

```
from transformers import Trainer, DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
)

trainer.train()

```

### **Calculating Metrics**

#### **BLEU Score**

```
import nltk
from nltk.translate.bleu_score import corpus_bleu

nltk.download('punkt')

references = [[nltk.word_tokenize(resp.lower())] for resp in reference_responses]
hypotheses = [nltk.word_tokenize(resp.lower()) for resp in generated_responses]

bleu_score = corpus_bleu(references, hypotheses)
print(f"BLEU Score: {bleu_score:.4f}")

```

#### **ROUGE Scores**

```
from rouge_score import rouge_scorer

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

```

#### **Perplexity**


```
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

loss_fct = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='sum')
total_loss = 0
total_tokens = 0

for batch in tqdm(test_dataloader):
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

```


---

## **Conclusion**

By following this guide, you've:

- Understood the UDCv2 dataset structure.
- Preprocessed the data for GPT-2 fine-tuning.
- Fine-tuned GPT-2 on the conversation data.
- 
---

## **References**

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/model_doc/gpt2)
- [PyTorch Cuda Documentation](https://pytorch.org/get-started/locally/)
- [Ubuntu Dialogue Corpus](https://github.com/rkadlec/ubuntu-ranking-dataset-creator)
