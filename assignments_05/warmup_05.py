import os
from dotenv import load_dotenv
import json


load_dotenv()
print("API key loaded:", bool(os.getenv("OPENAI_API_KEY")))

from openai import OpenAI
# --- Completion API ---
# API Q1
client = OpenAI()

response = client.chat.completions. create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "What is one thing that makes Python a good language for beginners?"
        }
    ]
)
print("Response:", response.choices[0].message.content)
print("Model:", response.model)
print("Total tokens:", response.usage.total_tokens)


# API Q2
prompt = "Suggest a creative name for a data engineering consultancy."

for temperature in [0, 0.7, 1.5]:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature

    )
    print(f"\nTemperature {temperature}:")
    print(response.choices[0].message.content)


# The names differed at each temperature.
# In this run, 1.5 gave a more playful name: Data Dynamo.
# I would choose temperature 0 for more consistent output,
# although it does not guarantee identical answers every time.

# API Q3
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Give me a one-sentence fun fact about pandas (the animal, not the library). "

        }
    ],
    n=3,
    temperature=1.0
)
for number, choice in enumerate(response.choices, start=1):
    print(f"\nPanda fact {number}:")
    print(choice.message.content)

# API Q4 
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Explain how neural networks work."

        }
    ],
    max_tokens=15
)

print("\nAPI Q4 - max_tokens=15:")
print(response.choices[0].message.content)


# --- Reponses
# The response stopped in the middle of a sentence.
# max_tokens=15 limited the output to 15 tokens.
# This helps control output length and cost, but the answer may be incomplete.

# --- System Messages and Conversation ---
# System Q1

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "You are a patient, encouraging Python tutor. Explain things simply and end with encouragement."

        },
        {
            "role": "user",
            "content": "I dont understand what a list comprehension is"
        }
    ]
)
print("\nPatient tutor:")
print(response.choices[0].message.content)

# System Q1 - Second persona

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "You are a formal Python reference manual. Give a brief, technical explanation without encouragement."
        },
        {
            "role": "user",
            "content": "I dont understand what a list comprehension is"
        }
    ]
)

print("\nPython reference manual:")
print(response.choices[0].message.content)


# The patient tutor sounded friendly and encouraging.
# It used a comparison and explained the example step by step.
# The reference manual was more direct and technical.
# Changing the system message changed the style of the answer,
# even though the user question stayed the same.

# System Q2

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": "My name is Jordan. I'm learning Python."
    },
    {
        "role": "assistant",
        "content": "Nice to meet you, Jordan! Python is a great language to learn. What are you working on?"
    },
    {
        "role": "user",
        "content": "Can you remind me what my name is?"
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("\nSystem Q2 - Conversation history:")
print(response.choices[0].message.content)

# The model answered Jordan because the name was in the messages list.
# Each API request is separate, so we provide earlier messages
# when we want the model to use conversation history.

# --- Prompt Engineering ---
# Prompt Q1 - Zero-shot

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

for review in reviews:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"Classify this review as positive, negative, or mixed:\n{review}"
            }
        ]
    )

    print("\nReview:", review)
    print("Zero-shot sentiment:", response.choices[0].message.content)

    # Prompt Q2 - One-shot

for review in reviews:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"""Classify the review as positive, negative, or mixed.
Follow the format of this example.

Review: Fast shipping but the item arrived damaged.
Sentiment: mixed

Review: {review}
Sentiment:"""
            }
        ]
    )

    print("\nReview:", review)
    print("One-shot sentiment:", response.choices[0].message.content)

    # Both approaches classified the three reviews the same way.
# Zero-shot included extra explanations.
# With one example, one-shot returned just the sentiment labels.

# Prompt Q3 - Few-shot

for review in reviews:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"""Classify the review as positive, negative, or mixed.
Follow the format of these examples.

Review: The service was excellent.
Sentiment: positive

Review: The product broke on the first day.
Sentiment: negative

Review: Fast shipping but the item arrived damaged.
Sentiment: mixed

Review: {review}
Sentiment:"""
            }
        ]
    )

    print("\nReview:", review)
    print("Few-shot sentiment:", response.choices[0].message.content)

    # One-shot and few-shot returned the same labels and format in this run.
# I would use zero-shot for a simple task with clear instructions.
# I would use one-shot to demonstrate the answer format.
# I would use few-shot when several examples help explain the categories.

# Prompt Q4 - Step-by-step calculation

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": """An employee earns an annual salary of $85,000.
They receive a 12% raise. Six months later, their new annual
salary increases by another $7,500.
What is their final annual salary?
Show the calculation steps and clearly state the final answer."""
        }
    ]
)

print("\nPrompt Q4 - Salary calculation:")
print(response.choices[0].message.content)

# The model calculated the percentage raise first, then added $7,500.
# Showing the steps made the calculations easier to check.
# Breaking a problem into smaller steps can help avoid mistakes,
# but I still need to verify the answer.

# Prompt Q5 - Structured output with JSON

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": """Analyze this review:
"The tool handles large datasets well, but the interface is clunky
and the export options are limited."

Return only a valid JSON object with these keys:
"sentiment": "positive", "negative", or "mixed"
"confidence": a number between 0 and 1
"reason": a one-sentence explanation

Do not include Markdown or any text outside the JSON object."""
        }
    ]
)

raw_response = response.choices[0].message.content
print("\nPrompt Q5 - Raw JSON:")
print(raw_response)

try:
    result = json.loads(raw_response)

    print("Sentiment:", result["sentiment"])
    print("Confidence:", result["confidence"])
    print("Reason:", result["reason"])
except json.JSONDecodeError:
    print("Could not parse the response as JSON.")
    print("Raw response:", raw_response)


# Prompt Q6 - Delimiters

texts = [
    "Boil water, add salt, cook the pasta for 8–10 minutes, drain it, and add sauce.",
    "The afternoon sunlight filled the quiet room."
]

for text in texts:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"""Read the text between the triple backticks.
If it contains instructions, rewrite them as numbered steps.
Otherwise, respond with exactly: No steps provided.

```
{text}
```"""
            }
        ]
    )

    print("\nPrompt Q6 - Input:", text)
    print("Output:", response.choices[0].message.content)

# The model turned the pasta instructions into numbered steps.
# For the descriptive sentence, it returned "No steps provided."
# The triple backticks separate the text being analyzed
# from the instructions telling the model what to do.

# --- Local Models with Ollama ---
# Ollama Q1

ollama_response = """A large language model is an advanced AI system that can understand and generate human-like text, such as writing stories, solving problems, or answering questions. It relies on vast amounts of training data and learns to recognize patterns and context to produce coherent, meaningful responses."""

print("\nOllama response:")
print(ollama_response)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Explain what a large language model is in two sentences."
        }
    ]
)

print("\nOpenAI response:")
print(response.choices[0].message.content)

# Both models answered in two sentences and described learning
# patterns from training data to generate text.
# Ollama mentioned writing stories, while OpenAI mentioned summarizing.
# An advantage of running locally is that prompts stay on my computer.
# A disadvantage is that performance depends on my computer's resources.

