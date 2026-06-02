import pandas as pd
from sentence_transformers import SentenceTransformer, util
import os
from dotenv import load_dotenv

# -----------------------------
# Load ENV
# -----------------------------
load_dotenv()

INPUT_EXCEL = os.getenv("INPUT_EXCEL_PATH")
QUESTIONS_FILE = os.getenv("QUESTIONS_FILE_PATH")
OUTPUT_FILE = os.getenv("OUTPUT_FILE_PATH")

if not INPUT_EXCEL or not os.path.exists(INPUT_EXCEL):
    raise ValueError("Invalid INPUT_EXCEL_PATH in .env")

if not QUESTIONS_FILE or not os.path.exists(QUESTIONS_FILE):
    raise ValueError("Invalid QUESTIONS_FILE_PATH in .env")

if not OUTPUT_FILE:
    raise ValueError("OUTPUT_FILE_PATH not set in .env")

# -----------------------------
# Load Topic Data
# -----------------------------
df = pd.read_excel(INPUT_EXCEL)

df["Full_Text"] = df["Full_Text"].fillna("")
df["Topic Label"] = df["Topic Label"].fillna("")

topics = []
topic_labels = []
topic_texts = []

for _, row in df.iterrows():
    topics.append(row["Topic"])
    topic_labels.append(row["Topic Label"])
    topic_texts.append(row["Full_Text"])

print(f"Loaded {len(topic_labels)} topic summaries")

# -----------------------------
# Load Questions
# -----------------------------
with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
    questions = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(questions)} questions")

# -----------------------------
# Load SciBERT
# -----------------------------
model = SentenceTransformer("allenai/scibert_scivocab_uncased")

print("Loaded SciBERT model")

# -----------------------------
# Encode Topics
# -----------------------------
topic_embeddings = model.encode(
    topic_texts,
    convert_to_tensor=True,
    show_progress_bar=True
)

# -----------------------------
# Encode Questions
# -----------------------------
question_embeddings = model.encode(
    questions,
    convert_to_tensor=True,
    show_progress_bar=True
)

# -----------------------------
# Topic Matching
# -----------------------------
TOP_K = 5
results = []

for i, question in enumerate(questions):

    print(f"\n🔎 Question: {question}")

    cos_scores = util.cos_sim(question_embeddings[i], topic_embeddings)[0]

    top_results = cos_scores.argsort(descending=True)[:TOP_K]

    for rank, idx in enumerate(top_results, start=1):

        topic = topics[idx]
        label = topic_labels[idx]
        score = cos_scores[idx].item()
        summary = topic_texts[idx]

        print(f"  Rank {rank} | {label} | {score:.4f}")

        results.append({
            "Question": question,
            "Rank": rank,
            "Topic": topic,
            "Topic_Label": label,
            "Similarity": round(score, 4),
            "Mapped_Summary": summary
        })

# -----------------------------
# Save Output
# -----------------------------
results_df = pd.DataFrame(results)

results_df.to_excel(OUTPUT_FILE, index=False)

print(f"\n✅ Results saved to: {OUTPUT_FILE}")