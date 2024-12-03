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
def evaluate_chunk(Accept,chunk):
    prompt = f"""
    You are an expert LLM evaluator. I will give you a prefix and 10 or less draft token text. Your task is to assess whether each of the given tokens makes sense as a continuation of the prefix. 
    Return a list of 10 values: use 1 if the token logically follows the given prefix, and 0 if it does not. The format should be: [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]. No explaniation need.

    Prefix: 
    {Accept}
    Draft Text:
    {chunk}
    """
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert LLM evaluator."},
            {"role": "user", "content": prompt}
        ]
    )
    # Extract the response content and return the evaluation
    print(completion.choices[0].message)
    return completion.choices[0].message


results = []  # Store the evaluation results for each draft
for i in range(len(data["draft_generated_tokens"])):
    # Gather all previous accept tokens as the prefix
    if i == 0:
        accept_tokens_prefix = []  # No accepted tokens at the beginning
    else:
        accept_tokens_prefix = sum(data["accepted_path"][:i], [])
    
    draft_tokens = data["draft_generated_tokens"][i]
    accept_text = tokenizer.decode(accept_tokens_prefix, skip_special_tokens=True)

    for i in draft_tokens:
        decoded_text = tokenizer.decode(i, skip_special_tokens=True)
        # Split text into chunks of 10 tokens
        token_chunks = decoded_text.split()  # Split into words
        chunks_of_10 = [" ".join(token_chunks[j:j+10]) for j in range(0, len(token_chunks), 10)]
        draft_evaluations = []
        for chunk in chunks_of_10:
            evaluation = evaluate_chunk(accept_text, chunk)
            draft_evaluations.append(evaluation)
        
        # Store evaluations for this draft
        results.append(draft_evaluations)

# Update the JSON file with the evaluation results
data["evaluation_results"] = results  # Add the evaluations to the data

# Save the updated JSON file
with open("Spec_tree_logs_evaluated.json", "w") as file:
    json.dump(data, file, indent=4)

print("Evaluation complete. Results saved to SpecTree_logs_evaluated.json.")
