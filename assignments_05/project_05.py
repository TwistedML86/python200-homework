from dotenv import load_dotenv
from openai import OpenAI
import json 
import ollama



# --- Task 1: Setup and helper function ---

load_dotenv()
client = OpenAI()
SYSTEM_PROMPT = """
You are a supportive job application coach for people changing careers.
Help users improve resume bullets, write cover letters, and discuss
job applications. Keep the conversation focused on these topics.

Do not invent experience, qualifications, achievements, or numbers.
Remind users to review and edit generated text before using it
in a job application.

Explain that your advice reflects general hiring practices and may
not fit every industry or situation. Encourage users to use their
own judgment.
"""

# I chose a supportive coach to help people changing careers.
# The instructions discourage invented qualifications and remind
# users to review the suggestions before using them.

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

# --- Task 2: Bullet Point Rewriter ---

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    bullet_text = "\n".join(bullets)

    prompt = f"""Rewrite these resume bullets using clear action verbs.
Make them more specific using only the information provided.
Do not invent achievements, numbers, or experience.

Return ONLY a valid JSON list.
Each item must have two keys:
"original": the original bullet
"improved": your rewritten version

Bullet points:
```
{bullet_text}
```
"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    answer = get_completion(messages).strip()

    if answer.startswith("```"):
        answer = "\n".join(answer.splitlines()[1:-1])

    return json.loads(answer)

sample_bullets = [
    "Helped customers with questions.",
    "Made reports for management.",
    "Worked with a team to meet deadlines."
]

rewritten_bullets = rewrite_bullets(sample_bullets)

for bullet in rewritten_bullets:
    print("\nOriginal:", bullet["original"])
    print("Improved:", bullet["improved"])
    
# The original bullets were vague and used general verbs.
# The model replaced them with assisted, compiled, and collaborated.
# The rewrites sound more professional, but I still need to check
# that every statement accurately describes the person's work.

# --- Task 3: Cover Letter Opening ---

def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""Write a cover letter opening of 3–5 sentences.
Use a confident, specific tone like the examples below.
Avoid generic openings such as "I am writing to apply."
Use only the applicant's actual background. Do not invent facts.

Example 1:
Role: Junior Software Developer
Background: Seven years as a nurse, then completed a coding bootcamp.
Opening: Working as a nurse for seven years taught me to solve
problems carefully and communicate under pressure. Completing a
coding bootcamp gave me a new way to apply those strengths.
I would bring both experiences to a junior software developer role.

Example 2:
Role: Junior Software Engineer
Background: Ten years in retail banking operations, then two years
of self-directed Python study.
Opening: After ten years in retail banking operations, I became
curious about the software behind our daily work. That curiosity
led me to spend two years learning Python independently.
I would bring practical banking experience and a commitment to
learning to a junior software engineer role.

Now write an opening for this applicant:
Job title: {job_title}
Background: {background}

Return only the opening paragraph."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    return get_completion(messages)

sample_opening = generate_cover_letter(
    job_title="Junior Data Engineer",
    background=(
        "Five years as a teacher. Learning Python and building "
        "data pipelines through coursework."
    )
)

print("\nCover letter opening:")
print(sample_opening)

# The examples show how to connect previous work to a new career.
# They guide the model toward a confident, specific opening
# instead of a generic introduction.


# --- Task 4: Moderation ---

def is_safe(text: str) -> bool:
    response = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )

    if response.results[0].flagged:
        print("Please rephrase your message respectfully.")
        return False

    return True


print("\nModeration test 1:")
print(is_safe("Can you help me improve my resume?"))

print("\nModeration test 2:")
print(is_safe("You are worthless and everyone would be better off without you."))


# --- Task 5: Chatbot Loop ---

def run_chatbot():
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print("\nJob Application Helper")
    print("Ask about job applications, resume bullets, or cover letters.")
    print("Type quit or exit to stop.")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        if not user_input:
            continue

        if not is_safe(user_input):
            continue

        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("Enter one resume bullet per line.")
            print("Type DONE when finished.")

            bullets = []

            while True:
                bullet = input("Bullet: ").strip()

                if bullet.upper() == "DONE":
                    break

                if not bullet:
                    continue

                if not is_safe(bullet):
                    continue

                bullets.append(bullet)

            if bullets:
                rewritten = rewrite_bullets(bullets)
                messages.append({
                    "role": "user",
                    "content": user_input + "\n" + "\n".join(bullets)
                })
                messages.append({
                    "role": "assistant",
                    "content": json.dumps(rewritten)
                })

                for item in rewritten:
                    print("\nOriginal:", item["original"])
                    print("Improved:", item["improved"])

            continue
        if "cover letter" in user_input.lower():
            job_title = input("Job title: ").strip()

            if not is_safe(job_title):
                continue

            background = input("Your background: ").strip()

            if not is_safe(background):
                continue

            opening = generate_cover_letter(job_title, background)
            print("\nCover letter opening:", opening)
            messages.append({
                "role": "user",
                "content": user_input + "\nJob title: " + job_title
                           + "\nBackground: " + background
            })
            messages.append({
                "role": "assistant",
                "content": opening
            })



            continue

        messages.append({"role": "user", "content": user_input})

        response = ollama.chat(model="qwen3:0.6b", messages=messages)
        answer = response.message.content
        print("\nCoach:", answer)
        messages.append({"role": "assistant", "content": answer})
    
if __name__ == "__main__":
    run_chatbot()

# --- Task 6: Reflection ---

# The AI added experience analyzing student data that was not
# included in the background we provided.
# An employer could believe this claim and consider hiring someone
# for experience they do not actually have.
# This could damage trust when the person cannot explain or
# demonstrate that experience.
# Users should review every claim before submitting an application.
# Before using this tool in production, I would require users to
# confirm that the generated statements match their real experience.


# ---Optional --- 

# Optional Ollama comparison
# I asked both models what entry-level roles I could explore.
# GPT-4o-mini listed more roles and explained what each one does.
# Qwen gave fewer roles but suggested specific tools for each.
# Qwen also claimed Google requires Hadoop, which I could not verify.
# Qwen generated replies on my laptop without using OpenAI for
# generation, but the first reply took a while. Moderation still
# used OpenAI, so the chatbot was not fully local.