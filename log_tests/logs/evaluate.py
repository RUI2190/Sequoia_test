import os
import json
from openai import OpenAI
from transformers import AutoTokenizer

# Initialize OpenAI client
client = OpenAI()

# Define target model and tokenizer
target_model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(target_model_name)

# Load the JSON file
with open("Spec_tree_logs.json", "r") as file:
    data = json.load(file)

# Define the evaluation function
def evaluate_draft_tokens(draft_tokens):
    # Decode the tokens into text
    decoded_text = tokenizer.decode(draft_tokens, skip_special_tokens=True)

    # Create the prompt asking GPT-4 to split the text into 10-token chunks and evaluate
    prompt = f"""
    You are an expert LLM evaluator. Your task is to assess whether each 10-token chunk in the following text makes sense.
    Return a list of 1s and 0s, where 1 means the chunk makes sense and 0 means it doesn't.

    Here is the text:
    {decoded_text}
    
    Please split the text into chunks of 10 tokens and evaluate each chunk.
    Provide your response as a list of integers, e.g., [1, 0, 1, 1, 0, ...]
    """

    # Make the API call
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert LLM evaluator."},
            {"role": "user", "content": prompt}
        ]
    )
    print(completion.choices[0].message)
    # Extract the response, which should be a list of integers
    return completion.choices[0].message


results = []  # Store the evaluation results for each draft
for draft_tokens in data["draft_generated_tokens"]:
    draft_evaluations = evaluate_draft_tokens(draft_tokens)
    results.append(draft_evaluations)

# Update the JSON file with the evaluation results
data["evaluation_results"] = results  

# Save the updated JSON file
with open("Spec_tree_logs_evaluated.json", "w") as file:
    json.dump(data, file, indent=4)

print("Evaluation complete. Results saved to SpecTree_logs_evaluated.json.")
