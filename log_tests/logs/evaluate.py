import os
from openai import OpenAI
import json

client = OpenAI()

# Load the JSON file
with open("SpecTree_logs.json", "r") as file:
    data = json.load(file)

prompt1 = """
You are an professional LLM evaluator. Your task is to score the model's output trace against the expected output based on the following criteria:
1. Correctness: Is the trace logically correct? (Score 1-10)
2. Completeness: Does the trace fully satisfy the expected requirements? (Score 1-10)
3. Clarity: Is the trace clear and easy to understand? (Score 1-10)

Log Trace:
{data_trace}

Provide a score for each criterion and a brief explanation for the score.
"""

def evaluate_trace(data_trace):
    prompt = prompt1.format(data_trace=data_trace)
    response = client.chat.completions.create(
        model="gpt-4",  # Use GPT-4 or any available model
        messages=[
            {"role": "system", "content": "You are an professional LLM evaluator."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0
    )
    # Correctly access the text of the first completion
    return response.choices[0].message

evaluation = evaluate_trace(data)
print(evaluation)
