from openai import OpenAI
import json
import os
import time

# Set up the OpenAI client for the Gemini API
client = OpenAI(api_key="YOUR_API_KEY", base_url="https://generativelanguage.googleapis.com/v1beta/openai/")


# System prompt for context
system_prompt = (
    "Du bist ein Experte in der Extraktion von Skills aus Textzonen von Stelleninseraten, welche Fähigkeiten und Tätigkeiten beinhalten. Du bist spezialisiert auf Stelleninserate aus der Schweiz."
)

prompt_template = (
    "Datenherkunft: Du erhältst die folgenden Daten aus einer Pipeline, die Skills aus Stelleninseraten extrahiert. "
    "Die Stelleninserate werden in Zonen unterteilt. "
    "Die relevanten Zonen 'Fähigkeiten und Tätigkeiten' sowie 'Erfahrung' werden einem LLM übergeben."
    "Das LLM generiert daraus einen Raw Skill, in dem der für den Skill relevante Textteil genommen wird und daraus ein Skill erstellt wird. "
    "Anschliessend wird dieser Raw Skill durch ein Embedding basiertes Matching auf einen Ontologie Skill gematcht. "
    "Das Matching auf die Ontologie erfolgt, um die Skills in ein bevorzugtes standardisiertes Format zu bringen.\n\n"
    "Die Ontologie Skills sind dabei immer aus einem Topic und einem Prädikat aufgebaut z.B. 'Analysen durchführen' oder 'Python programmieren'. "
    "Aufgabe: Deine Aufgabe ist es, die inhaltlich fehlenden Skills der Ontologie Matches zu identifizieren. "
    "Dazu vergleichst du die bereits gefunden Ontologie Skills mit der Liste der Raw Skills und prüfst, ob unter den Ontologie Skills einer der Raw Skills inhaltlich fehlt. "
    "Achte dabei darauf, dass die Skills nicht wörtlich vorkommen müssen, sondern in ihrem Inhalt. "
    "Fokussiere dich auf neuartige Skills, welche noch nicht in der Ontologie vorkommen, da sie eine Neuheit in der Arbeitswelt darstellen.\n\n"
    "Die vollständigen Stelleninserate dienen als Kontext, sollen aber nicht direkt zur Skill-Identifikation genutzt werden.\n\n"
    "Output: Der Output soll ein JSON File mit den fehlenden Skills im Ontologie Format sein. Es soll kein anderer Text zurückgegeben werden, "
    "nur die fehlenden Skills, als JSON Objekt mit dem Schlüssel 'mismatch_skills' und den fehlenden Skills als Wert.\n\n"
    "Daten: Die folgenden Skills wurden bereits gefunden: {skills_name}\n\n"
    "Die folgenden Raw Skills wurden in den Zonen gefunden: {skills_phrase}\n\n"
    "Das Stelleninserat, der Jobtitel und der Zeitstempel, welche als Kontext helfen können: Job Titel: {job_title} Text: {content} Zeitstempel: {timestamp}\n\n"
)

# Set model
model_name = "gemini-2.0-flash"


# Load input data from JSON file
num_entries = 30
input_dir = "data/input"
input_file_name = f"combined_input_{num_entries}_branchen_after_2024.json"
input_file = os.path.join(input_dir, input_file_name)

with open(input_file, "r", encoding="utf-8") as file:
    data = json.load(file)

# Initialize results
results = []

# Process each job posting separately
for item in data:
    try:
        start_time = time.time()
        ad_id = item["id"]
        content = item["content"]
        language = item["language"]
        timestamp = item["timestamp"]
        job_title = item["job_title"]
        skills_name_output = item.get("skills_name", [])
        skills_phrase_output = item.get("skills_phrase", [])
        skills_name = ", ".join(item.get("skills_name", []))  # Convert list to comma-separated string
        skills_phrase = ", ".join(item.get("skills_phrase", []))  # Convert list to comma-separated string

        # Print notification
        print(f"Processing job advertisement ID: {ad_id}")

        # JSON adding hyperparameter
        # Make API request
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_template.format(
                    content=content, skills_name=skills_name, skills_phrase=skills_phrase, job_title=job_title, timestamp=timestamp
                )}
            ],
            response_format={ "type": "json_object" }
        )

        # Extract mismatch skills from the response JSON
        response_content = response.choices[0].message.content

        # Parse the content which is a JSON string
        try:
            mismatch_skills_data = json.loads(response_content)
            mismatch_skills = mismatch_skills_data.get("mismatch_skills", ["Error extracting skills"])
        except json.JSONDecodeError:
            mismatch_skills = ["Error decoding response"]
        
        # Store results
        results.append({
            "id": ad_id,
            "existing_skills": skills_name_output,
            "existing_skill_phrases": skills_phrase_output,
            "mismatch_skills": mismatch_skills
        })

        # Timing
        elapsed_time = time.time() - start_time
        print(f"Successfully processed ID: {ad_id} | Duration: {elapsed_time:.2f} seconds")

    except Exception as e:
        print(f"Error for ID {ad_id}: {e}")
        results.append({
            "id": ad_id,
            "content": content,
            "mismatch_skills": ["Error"]
        })

# Define output directory and file
output_dir = "data/prompt_output"
os.makedirs(output_dir, exist_ok=True)
output_file_name = f"novelty_skills_{num_entries}_{model_name}.json"
output_file = os.path.join(output_dir, output_file_name)

# Save results to JSON file
try:
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)
    print(f"Output saved to {output_file} successfully.")
except PermissionError:
    print(f"Error: PermissionError for {output_file} - Please close the file and try again.")