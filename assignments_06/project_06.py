from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# --- Step 1: Setup ---

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

docs_dir = Path(
    "../python-200-v1/lessons/06_AI_augmentation/resources/groundwork_docs"
)

assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

# --- Step 2: Load the Documents ---

documents = SimpleDirectoryReader(str(docs_dir)).load_data()

print("\nDocuments loaded:", len(documents))

for document in documents:
    print("File:", document.metadata["file_name"])

# --- Step 3: Build the Index and Query Engine ---

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini")

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)

print("\nIndex built successfully. Ready to answer questions.")

# --- Step 4: Query the Assistant ---

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for question in questions:
    print("\nQuestion:", question)

    response = query_engine.query(question)
    print("Answer:", response)

    source = response.source_nodes[0]
    print("Document:", source.node.metadata["file_name"])
    print("Similarity score:", source.score)
    print("Chunk:", source.node.get_content()[:200])

# The answers sounded confident and specific.
# The weekend hours matched the FAQ.
# I was surprised that our_story.txt was the top source for hours.
# The dairy-free question also ranked seasonal_specials.txt first.
# The model received three chunks, so the first source may not
# contain all the information used in the answer.
# I still need to compare the answers with the documents.

# --- Step 5: Find a Failure ---

question = "Does Groundwork offer a student discount, and how much is it?"

response = query_engine.query(question)

print("\nQuestion:", question)
print("Answer:", response)

for source in response.source_nodes:
    print("\nDocument:", source.node.metadata["file_name"])
    print("Similarity score:", source.score)
    print("Chunk:", source.node.get_content()[:200])

# I asked about a student discount because I expected that information
# might be missing from the documents.
# The retrieved sources did not provide a student discount policy.
# The model said there was no mention of it instead of guessing an amount.
# Its answer was more cautious than the earlier detailed answers.
# This shows why I should check the sources before trusting an answer.
# I would add a clear discount policy so the system could answer.


# --- Step 6: Reflection ---
# My setup and query engine took four statements.
# LlamaIndex handles chunking, embedding, and indexing for me.
# The keyword function needed more code just to match words.
# Writing all those steps myself would take more code and work.
# A framework helps me build the system with less code.

# Question 2
# A career center could use this to help students find scholarships
# and improve their resumes.
# The assistant could answer questions from scholarship documents
# and show the source for deadlines and requirements.
# It could also use an uploaded resume and job description
# to suggest changes without inventing experience.
# This could help students find information faster.


# Question 3
# RAG can still give a misleading or exaggerated answer
# even when it retrieves the right text.
# The model might misunderstand the source or add unsupported details.
# I would tell it not to invent information and to show its sources.
# If the sources do not support an answer, it should say
# "I could not find enough information in the provided documents."
# These instructions can help, but users still need to check the answer.


# --- Optional B: Add a New Document ---
# I added student_discount.txt with a sample student discount policy.
# It explains the 10% discount, student ID requirement, and restrictions.
# I tested: "Does Groundwork offer a student discount, and how much is it?"
# Before adding the document, the assistant could not find the information.
# After rebuilding the index, it retrieved student_discount.txt first
# and answered with the discount and ID requirement.
# RAG let me add new information without retraining the model.
# The Step 5 comments describe my test before adding this document.
