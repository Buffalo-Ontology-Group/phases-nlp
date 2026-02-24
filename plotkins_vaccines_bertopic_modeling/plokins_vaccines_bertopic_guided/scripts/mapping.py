#!/usr/bin/env python3
"""
Check whether topics.txt terms exist in vo_terms.tsv (?label column).
Save topics that are NOT present in VO.
"""

import csv
from pathlib import Path


# --------------------------------------------------
# Resolve paths relative to this script
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

VO_TSV = BASE_DIR / "vo_terms.tsv"     # ?ID  ?label  ?type
TOPICS_TXT = BASE_DIR / "topics.txt"   # one topic per line
OUTPUT_FILE = BASE_DIR / "topics_not_in_vo.txt"


# --------------------------------------------------
# Safety checks
# --------------------------------------------------
if not VO_TSV.exists():
    raise FileNotFoundError(f"VO TSV not found: {VO_TSV}")

if not TOPICS_TXT.exists():
    raise FileNotFoundError(f"Topics file not found: {TOPICS_TXT}")


# --------------------------------------------------
# Load VO labels (middle column)
# --------------------------------------------------
vo_labels = set()

with VO_TSV.open("r", encoding="utf-8", errors="ignore") as f:
    reader = csv.reader(f, delimiter="\t")
    next(reader, None)  # skip header

    for row in reader:
        if len(row) >= 2:
            label = row[1].strip().lower()
            if label:
                vo_labels.add(label)


# --------------------------------------------------
# Load topics
# --------------------------------------------------
topics = [
    line.strip()
    for line in TOPICS_TXT.read_text(encoding="utf-8", errors="ignore").splitlines()
    if line.strip()
]


# --------------------------------------------------
# Find topics NOT in VO
# --------------------------------------------------
missing_topics = sorted(
    topic for topic in topics
    if topic.lower() not in vo_labels
)


# --------------------------------------------------
# Output
# --------------------------------------------------
print(f"Total topics:              {len(topics)}")
print(f"VO labels:                 {len(vo_labels)}")
print(f"Topics NOT present in VO:  {len(missing_topics)}\n")

print("Topics missing from VO:")
for t in missing_topics:
    print(t)


# --------------------------------------------------
# Save results
# --------------------------------------------------
OUTPUT_FILE.write_text(
    "\n".join(missing_topics) + ("\n" if missing_topics else ""),
    encoding="utf-8"
)

print(f"\nSaved missing topics to: {OUTPUT_FILE}")
