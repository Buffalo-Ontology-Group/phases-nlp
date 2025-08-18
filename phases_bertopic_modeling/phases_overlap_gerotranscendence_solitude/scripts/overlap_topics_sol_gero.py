import os
from dotenv import load_dotenv
import re
import json
import requests
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import plotly.graph_objs as go
import networkx as nx


load_dotenv()

# Paths from env or set manually
txt_path_1 = os.getenv("BERTopic_TXT_SOLITUDE")
txt_path_2 = os.getenv("BERTopic_TXT_GEROTRANSCENDENCE") 
output_dir = os.getenv("RESULTS_DIRECTORY") 
os.makedirs(output_dir, exist_ok=True)

# CoreNLP OpenIE server URL
CORENLP_URL = "http://localhost:9000"

def parse_bertopic_txt(file_path):
    """
    Parses a BERTopic-generated TXT file to extract topics, labels, keywords, and summaries.

    Args:
        file_path (str): Path to the BERTopic output text file.

    Returns:
        dict: Dictionary mapping topic IDs to a dictionary with keys
            'label', 'keywords', and 'summary'.
    """
    topics = {}
    current_topic = None
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if match := re.match(r"Topic (\d+) \((.*?)\): (.*)", line):
                current_topic = int(match.group(1))
                label = match.group(2)
                keywords = match.group(3)
                topics[current_topic] = {
                    "label": label,
                    "keywords": keywords,
                    "summary": ""
                }
            elif line.strip().startswith("Summary:") and current_topic is not None:
                topics[current_topic]["summary"] = line.replace("Summary:", "").strip()
    return topics

def openie_extract_relations(text):
    """
    Extrcats relational phrases from input text using the OpenIE annotator from Stanford CoreNLP.

    Args:
        text (str): Input text for relaiton extraction.

    Returns:
        list: List of extracted relaitons (strings).
    """
    if not text.strip():
        return []
    props = {
        'annotators': 'tokenize,ssplit,pos,lemma,depparse,natlog,openie',
        'outputFormat': 'json',
        'timeout': 30000
    }
    try:
        response = requests.post(
            CORENLP_URL,
            params=props,
            data=text.encode('utf-8'),
            headers={'Content-Type': 'application/octet-stream'}
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Error in OpenIE request: {e}")
        return []
    relations = []
    for sentence in data.get("sentences", []):
        for triple in sentence.get("openie", []):
            rel = triple.get("relation", "").strip()
            if rel:
                relations.append(rel)
    return relations

# Load topics
topics_1 = parse_bertopic_txt(txt_path_1)
topics_2 = parse_bertopic_txt(txt_path_2)

df1 = pd.DataFrame([
    {"id": tid, "label": t["label"], "keywords": t["keywords"], "summary": t["summary"], "source": "solitude"}
    for tid, t in topics_1.items()
])
df2 = pd.DataFrame([
    {"id": tid, "label": t["label"], "keywords": t["keywords"], "summary": t["summary"], "source": "gerotranscendence"}
    for tid, t in topics_2.items()
])

# Prepare combined text for embeddings
df1["full_text"] = df1["label"] + " " + df1["keywords"] + " " + df1["summary"]
df2["full_text"] = df2["label"] + " " + df2["keywords"] + " " + df2["summary"]

# Compute embeddings and cosine similarity
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings1 = model.encode(df1["full_text"].tolist(), convert_to_tensor=True)
embeddings2 = model.encode(df2["full_text"].tolist(), convert_to_tensor=True)
cosine_scores = util.cos_sim(embeddings1, embeddings2).cpu().numpy()

# Find matching topic pairs above threshold
threshold = 0.6
matches = []
for i, row in df1.iterrows():
    for j, row2 in df2.iterrows():
        score = cosine_scores[i, j]
        if score >= threshold:
            matches.append({
                "solitude_id": row["id"],
                "solitude_label": row["label"],
                "gerotranscendence_id": row2["id"],
                "gerotranscendence_label": row2["label"],
                "similarity": float(score)
            })

merged_df = pd.DataFrame(matches)
merged_df["relationship"] = merged_df["similarity"].apply(lambda x: "strong_match" if x > 0.75 else "related")
merged_df["merged_topic"] = merged_df.apply(
    lambda r: f"{r['solitude_label']} → {r['gerotranscendence_label']}", axis=1
)

# Extract raw OpenIE relations for each merged topic
print("Extracting relations with OpenIE... this may take some time")
extracted_relations_list = []
for idx, row in merged_df.iterrows():
    combined_text = (
        f"{row['solitude_label']}. {row['gerotranscendence_label']}. "
    )
    # Add summaries if available
    s_summary = df1.loc[df1["id"] == row["solitude_id"], "summary"].values
    g_summary = df2.loc[df2["id"] == row["gerotranscendence_id"], "summary"].values
    if s_summary.size > 0:
        combined_text += s_summary[0] + " "
    if g_summary.size > 0:
        combined_text += g_summary[0]
    
    relations = openie_extract_relations(combined_text)

    # Deduplicate while preserving order
    seen = set()
    unique_relations = []
    for rel in relations:
        if rel not in seen:
            seen.add(rel)
            unique_relations.append(rel)
            
    extracted_relations_list.append(unique_relations)

merged_df["extracted_relations"] = extracted_relations_list

# Save CSV
column_order = [
    "solitude_id", "solitude_label",
    "gerotranscendence_id", "gerotranscendence_label",
    "similarity", "relationship", "merged_topic", "extracted_relations"
]
merged_csv = os.path.join(output_dir, "merged_topics_with_relations.csv")
merged_df[column_order].to_csv(merged_csv, index=False)

# Interactive Graph visualization (same as before)
G = nx.Graph()
for _, row in merged_df.iterrows():
    sid = f"S{row['solitude_id']}: {row['solitude_label']}"
    gid = f"G{row['gerotranscendence_id']}: {row['gerotranscendence_label']}"
    G.add_node(sid, group="solitude")
    G.add_node(gid, group="gerotranscendence")
    G.add_edge(sid, gid, weight=row["similarity"])

solitude_nodes = [n for n, d in G.nodes(data=True) if d["group"] == "solitude"]
gero_nodes = [n for n, d in G.nodes(data=True) if d["group"] == "gerotranscendence"]

pos = {}
y_spacing_solitude = 1 / (len(solitude_nodes) + 1)
for i, node in enumerate(solitude_nodes, start=1):
    pos[node] = (0, 1 - i * y_spacing_solitude)

y_spacing_gero = 1 / (len(gero_nodes) + 1)
for i, node in enumerate(gero_nodes, start=1):
    pos[node] = (1, 1 - i * y_spacing_gero)

edge_x = []
edge_y = []
edge_colors = []
for edge in G.edges(data=True):
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])
    
    weight = edge[2].get('weight', 0)
    edge_colors.append('red' if weight > 0.75 else 'orange')

edge_traces = []
for color in set(edge_colors):
    xs = []
    ys = []
    for i, c in enumerate(edge_colors):
        if c == color:
            xs.extend(edge_x[i*3:i*3+3])
            ys.extend(edge_y[i*3:i*3+3])
    edge_traces.append(go.Scatter(
        x=xs, y=ys,
        line=dict(width=1, color=color),
        hoverinfo='none',
        mode='lines'
    ))

node_x, node_y, node_text, node_color = [], [], [], []
for node, data in G.nodes(data=True):
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_text.append(node)
    node_color.append("blue" if data["group"] == "solitude" else "green")

node_trace = go.Scatter(
    x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="top center",
    marker=dict(color=node_color, size=10, line_width=2),
    hoverinfo='text'
)

fig = go.Figure(data=edge_traces + [node_trace],
                layout=go.Layout(
                    title=dict(text='Merged Topic Label Graph with OpenIE Relations', font=dict(size=16)),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                ))

html_output = os.path.join(output_dir, "interactive_topic_graph_with_relations.html")
fig.write_html(html_output)

print(f"""
Process complete!
- Merged topics + raw OpenIE relations saved to: {merged_csv}
- Interactive graph saved to: {html_output}
""")
