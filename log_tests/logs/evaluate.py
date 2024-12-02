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

for i in data["accepted_path"]:
    print(i)
    decoded_text = tokenizer.decode(i, skip_special_tokens=True)
    print(decoded_text)

# # Define the evaluation function
# def evaluate_chunk(chunk):
#     prompt = f"""
#     You are an expert LLM evaluator. Your task is to assess whether the following text makes sense. 
#     Return 1 if it makes sense and 0 if it doesn't.

#     Text:
#     {chunk}
#     """
#     completion = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[
#             {"role": "system", "content": "You are an expert LLM evaluator."},
#             {"role": "user", "content": prompt}
#         ]
#     )
#     # Extract the response content and return the evaluation
#     print(completion.choices[0].message)[[[[[[]]]]]]
#     return completion.choices[0].message



# results = []  # Store the evaluation results for each draft
# for draft_tokens in data["draft_generated_tokens"]:
#     # Decode the token IDs into text
#     decoded_text = tokenizer.decode(draft_tokens, skip_special_tokens=True)
#     # Split text into chunks of 10 tokens
#     token_chunks = decoded_text.split()  # Split into words
#     chunks_of_10 = [" ".join(token_chunks[i:i+10]) for i in range(0, len(token_chunks), 10)]
#     draft_evaluations = []
#     for chunk in chunks_of_10:
#         evaluation = evaluate_chunk(chunk)
#         draft_evaluations.append(evaluation)
    
#     # Store evaluations for this draft
#     results.append(draft_evaluations)

# # Update the JSON file with the evaluation results
# data["evaluation_results"] = results  # Add the evaluations to the data

# # Save the updated JSON file
# with open("Spec_tree_logs_evaluated.json", "w") as file:
#     json.dump(data, file, indent=4)

# print("Evaluation complete. Results saved to SpecTree_logs_evaluated.json.")
