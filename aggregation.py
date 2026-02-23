import json
import csv
import re
from collections import defaultdict
from rapidfuzz import fuzz
from sklearn.cluster import DBSCAN
import numpy as np
from openai import OpenAI

# --- Configuration ---
EMBEDDING_MODEL = "text-embedding-3-small"
API_KEY = "YOUR_API_KEY"  # Replace with your actual API key1
client = OpenAI(api_key=API_KEY)

def normalize_string(s):
    return re.sub(r'\s+', ' ', s.strip().lower())

def extract_wo_fails(data):
    skills = []
    for entry in data:
        for result in entry.get("match_results", []):
            if result.get("status") == "wo-fail":
                mismatch_skill = result.get("mismatch_skill", "").strip()
                if mismatch_skill:
                    norm_skill = normalize_string(mismatch_skill)
                    skills.append(norm_skill)
    return skills

def group_similar_skills(skills, threshold=95):
    grouped = []
    used = set()
    for skill in skills:
        if skill in used:
            continue
        group = [skill]
        used.add(skill)
        for other in skills:
            if other not in used and fuzz.token_sort_ratio(skill, other) >= threshold:
                group.append(other)
                used.add(other)
        grouped.append(group)
    return grouped

def count_and_expand_grouped_skills(skills, threshold=95):
    grouped_skills = group_similar_skills(skills, threshold)
    result = []
    for rank, group in enumerate(grouped_skills, start=1):
        representative = group[0]
        count = sum(skills.count(s) for s in group)
        members = sorted(set(group))
        result.append((rank, count, representative, members))
    return result

def write_csv_with_group_members(grouped_data, output_path):
    grouped_data.sort(key=lambda x: x[1], reverse=True)
    with open(output_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'count', 'representative', 'group_members'])
        for rank, (count, rep, members) in enumerate(
            [(count, rep, members) for _, count, rep, members in grouped_data], start=1
        ):
            writer.writerow([rank, count, rep, ", ".join(members)])

def load_group_members_from_csv(csv_path, count_threshold):
    skills_set = set()
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                count = int(row.get('count', 0))
            except ValueError:
                continue
            if count < count_threshold:
                members_raw = row.get('group_members', '')
                members = [normalize_string(s) for s in members_raw.split(',')]
                skills_set.update(members)
    return sorted(skills_set)

def embed_skills(skills, model=EMBEDDING_MODEL):
    embeddings = []
    for skill in skills:
        response = client.embeddings.create(input=skill, model=model)
        embedding = response.data[0].embedding
        embeddings.append((skill, embedding))
    return embeddings

def cluster_embeddings(vectors, eps=0.73, min_samples=3):
    print("Running DBSCAN...")
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(vectors)
    return labels

def save_clusters_to_csv(skills, labels, output_csv_path):
    clusters = defaultdict(list)
    for skill, label in zip(skills, labels):
        if label != -1:
            clusters[label].append(skill)

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["cluster_id", "skills"])
        for cluster_id in sorted(clusters):
            writer.writerow([cluster_id, clusters[cluster_id]])
    print(f"Clustered results saved to {output_csv_path}")

def full_aggregation_pipeline(json_input, intermediate_csv, final_csv, count_threshold=3, eps=0.73, min_samples=3, grouping_threshold=85):
    # Step 1: Load JSON and extract skills
    with open(json_input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    skills = extract_wo_fails(data)

    # Step 2: Group similar skills and write to intermediate CSV
    grouped_data = count_and_expand_grouped_skills(skills, threshold=grouping_threshold)
    write_csv_with_group_members(grouped_data, intermediate_csv)
    print(f"Intermediate aggregation saved to {intermediate_csv}")

    # Step 3: Load long-tail group members for clustering
    long_tail_skills = load_group_members_from_csv(intermediate_csv, count_threshold)
    print(f"Loaded {len(long_tail_skills)} long-tail skills for clustering.")

    # Step 4: Embed skills and cluster
    embeddings = embed_skills(long_tail_skills)
    skill_texts = [s for s, _ in embeddings]
    vectors = np.array([e for _, e in embeddings])
    labels = cluster_embeddings(vectors, eps=eps, min_samples=min_samples)

    # Step 5: Save final clustered results
    save_clusters_to_csv(skill_texts, labels, final_csv)

if __name__ == "__main__":
    input_json = 'data/turbo_output/processed_1.3_novelty_skills_30_gemini-2.0-flash_combined_branchen_after_2024.json'
    intermediate_csv = 'data/aggregated_output/token_aggregated_1.3_novelty_skills_1000_gemini-2.0-flash_branchen_after_2024.csv'
    final_csv = 'data/aggregated_output/long_tail_skill_clusters_dbscan_1.3_30_gemini-2.0-flash.csv'

    full_aggregation_pipeline(
        json_input=input_json,
        intermediate_csv=intermediate_csv,
        final_csv=final_csv,
        count_threshold=3,
        eps=0.73,
        min_samples=1,
        grouping_threshold=85
    )
