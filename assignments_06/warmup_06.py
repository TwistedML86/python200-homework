from dotenv import load_dotenv
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

# --- RAG Concepts ---
# Concepts Q1
# Scenario A: RAG.
# It can look up information in the policy PDFs before answering.
# When policies change, the documents can be updated.

# Scenario B: Fine-tuning.
# The company has 3,000 examples of its writing style.
# These examples can help train the model to write in that style.

# Scenario C: Prompt engineering.
# She can include the two-page report in the prompt and ask questions.
# She only needs help with this one report.

# Concepts Q2
# A confidently wrong answer can make users believe it without checking.
# Saying "I am not sure" lets the user know they should verify it.
# My chatbot added experience analyzing student data that was not provided.
# An employer could believe this and think the applicant has that experience.
# The confident tone makes the claim sound true even when it is invented.

# Concepts Q3
# 1. Extract text from source documents.
# Get the text out of the source files.
# 2. Split text into chunks.
# Break the text into smaller pieces.
# 3. Convert text chunks into embeddings.
# Turn each chunk into numbers that represent its meaning.
# 4. Receive the user's query.
# Get the question the user wants answered.
# 5. Embed the user's query.
# Turn the question into numbers that represent its meaning.
# 6. Retrieve the most relevant chunks.
# Find the chunks with meanings closest to the question.
# 7. Inject retrieved chunks into the prompt.
# Include those chunks with the question in the prompt.
# 8. Generate a response from the LLM.
# Have the model use the retrieved text to answer the question.

# --- Keyword RAG ---

import string


def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]

# Keyword Q1
query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

result = simple_keyword_retrieval(query, documents, verbose=True)
print("Selected document:", result[0][0])

# The function selected loyalty.txt, but hours.txt has the answer.
# Three documents tied with one matching word each.
# "Your" matched in hiring.txt and loyalty.txt.
# The reverse sorting put loyalty.txt first because of its filename.

# Keyword Q2
query = "Do you have anything without caffeine?"

result = simple_keyword_retrieval(query, documents, verbose=True)
print("Selected document:", result[0][0])

# The function returned "None found" because no keywords matched.
# It missed that the question is related to the menu.
# Semantic retrieval could do better by matching the meaning.
# But the menu does not say which items are caffeine-free.

# Keyword Q3
# I predict "None found" because none of the remaining words
# match the words in the documents.

query = "How do I sign up for rewards?"

result = simple_keyword_retrieval(query, documents, verbose=True)
print("Selected document:", result[0][0])

# My prediction was correct. It returned "None found".
# The loyalty document is related, but the words do not match.
# The function cannot connect "rewards" with "loyalty program".

# --- Semantic RAG Concepts ---
# Semantic Q1
# 1. A vector embedding is a list of numbers that represents
# the meaning of text. Similar meanings have similar vectors.

# 2. The chunk with 0.85 is more relevant because it is closer to 1.
# Its meaning is more similar to the question, but that does not
# guarantee it has the correct answer.

# 3. Semantic search compares meaning instead of only exact words.
# "Rewards" and "loyalty program" are related, so it can find a match.

# Semantic Q2
# | Feature         | Keyword RAG                    | Semantic RAG                 |
# |-----------------|--------------------------------|------------------------------|
# | What is compared? | Exact word overlap           | Text embeddings (meaning)    |
# | What is retrieved? | Full document               | Relevant text chunks         |
# | Can it handle synonyms? | No                     | Yes                          |
# | Storage format  | Plain text dictionary          | Vector store or index        |
# | Relevance score | Number of overlapping keywords | Cosine similarity            |


# --- LlamaIndex ---
# LlamaIndex Q1

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini")

documents = SimpleDirectoryReader(
    "../python-200-v1/lessons/06_AI_augmentation/resources/brightleaf_pdfs"
).load_data()

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for question in questions:
    print("\nQuestion:", question)
    response = query_engine.query(question)
    print("Answer:", response)

    for source in response.source_nodes:
        print("Similarity score:", source.score)
        print("Chunk:", source.node.get_content()[:150])

# Benefits question:
# The first chunk was about employee benefits and looked relevant.
# The answer sounded confident and specific, without saying it was unsure.
# I did not expect the partnership and mission chunks to appear.

# Security question:
# The first chunk was about security and looked relevant.
# The answer sounded confident and gave specific policy details.
# The benefits and partnership chunks seemed less related.

# LlamaIndex Q2

question = "What employee benefits does BrightLeaf offer?"

for top_k in [1, 5]:
    print("\nsimilarity_top_k:", top_k)
    print("Question:", question)

    engine = index.as_query_engine(similarity_top_k=top_k)
    response = engine.query(question)
    print("Answer:", response)

    for source in response.source_nodes:
        print("Similarity score:", source.score)

# Both settings gave similar information about employee benefits.
# With top_k=1, the answer used a clear numbered list.
# In one run, top_k=5 left out the $600 wellness amount.
# In another run, it included that amount.
# More context did not clearly improve the answer.
# Extra chunks may be less relevant.

# LlamaIndex Q3
# I expect this question to be hard because I have not seen
# information about student loan payments in the results so far.

question = "Does BrightLeaf help employees pay off student loans?"

response = query_engine.query(question)
print("\nQuestion:", question)
print("Answer:", response)

for source in response.source_nodes:
    print("\nSimilarity score:", source.score)
    print("Chunk:", source.node.get_content())


# I expected the question to be hard because student loan help
# was not mentioned in the results I had seen.
# The model said the provided information did not mention this benefit.
# It did not make up an answer.
# The benefits chunk was related, but the other chunks were less useful.
# I would add a clear student loan policy so the system could answer.

# LlamaIndex Q4

from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

judge_llm = OpenAI(model="gpt-4o-mini")

faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llm)
relevancy_evaluator = RelevancyEvaluator(llm=judge_llm)

evaluation_questions = [
    "What employee benefits does BrightLeaf offer?",
    "What is BrightLeaf's recipe for chocolate cake?",
]

for question in evaluation_questions:
    response = query_engine.query(question)

    faithfulness_result = faithfulness_evaluator.evaluate_response(
        response=response
    )
    relevancy_result = relevancy_evaluator.evaluate_response(
        query=question,
        response=response
    )

    print("\nEvaluation question:", question)
    print("Answer:", response)
    print("Faithfulness score:", faithfulness_result.score)
    print("Relevancy score:", relevancy_result.score)


# A faithfulness score of 1.0 means the judge found the answer
# supported by the retrieved text.
# A score of 0.0 means the judge found unsupported information.
# This does not guarantee the answer is true outside the documents.

# Relevancy checks whether the answer addresses the question.
# Faithfulness checks whether it stays supported by the source text.

# The benefits answer scored 1.0 for both.
# The cake answer scored 1.0 for faithfulness and 0.0 for relevancy.
# It did not invent a recipe, but it could not give the recipe asked for.

# LLM-as-a-judge means using another model to evaluate an answer.
# Answers can use different words and still be good, so checking
# for one exact answer is not enough.
# The judge can also make mistakes.

