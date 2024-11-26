import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline

def main():
    # Load the fine-tuned model and tokenizer
    model_path = './fine_tuned_gpt2'  # Update this path if your model is saved elsewhere

    print("Loading the fine-tuned model...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up the text generation pipeline
    generator = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1  # Use GPU if available
    )

    print("\nWelcome to the GPT-2 Chatbot!")
    print("Type 'exit', 'quit', or 'q' to end the conversation.\n")

    # Initialize conversation history (optional)
    conversation_history = ""

    while True:
        # Get user input
        user_input = input("You: ")
        if user_input.strip().lower() in ['exit', 'quit', 'q']:
            print("Exiting the chatbot. Goodbye!")
            break

        # Append user input to the conversation history
        conversation_history += f"User: {user_input}\n"

        # Generate a response
        response = generator(
            conversation_history,
            max_length=512,
            num_return_sequences=1,
            no_repeat_ngram_size=3,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )

        # Extract and print the generated response
        generated_text = response[0]['generated_text']

        # Remove the conversation history from the generated text
        bot_response = generated_text[len(conversation_history):].strip()

        # Append the bot response to the conversation history
        conversation_history += f"Bot: {bot_response}\n"

        print(f"Bot: {bot_response}\n")

if __name__ == '__main__':
    main()
