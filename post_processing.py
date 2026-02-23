import json
import requests

API_URL = 'http://localhost:8000/get_match'
API_KEY = 'YOUR_API_KEY'  # Replace with your actual API key
DISTANCE_THRESHOLD = 1.3


# Function to get match from the API
def get_match(raw_skills):
    try:
        response = requests.post(API_URL, json={'skills': raw_skills}, headers={'Authorization': f'Bearer {API_KEY}'})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f'Error during API request: {e}')
        return None


# Logic of the turbo
def process_mismatch_skills(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    for entry in data:
        mismatch_skills = entry.get('mismatch_skills', [])
        existing_skills = entry.get('existing_skills', [])

        match_results = []

        if mismatch_skills != []:
            results = get_match(mismatch_skills)
            for skill in results:
                if skill:
                    matched_skill = skill.get('name')
                    raw_skill = skill.get('phrase')
                    distance = skill.get('distance')
                    distance = round(distance, 2)

                    if distance > DISTANCE_THRESHOLD:
                        match_results.append({'mismatch_skill': raw_skill, 'x28_skill': '', 'status': 'wo-fail', 'distance': distance})
                    else:
                        if matched_skill not in existing_skills:
                            match_results.append({'mismatch_skill': raw_skill, 'x28_skill': matched_skill, 'status': 'match-fail', 'distance': distance})
                        else:
                            match_results.append({'mismatch_skill': raw_skill, 'x28_skill': matched_skill, 'status': 'novelty-detection-fail', 'distance': distance})
        else:
            match_results.append({'mismatch_skill': '', 'x28_skill': '', 'status': 'no-skill', 'distance': 0})

        entry['match_results'] = match_results

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        print(f'Processed data has been saved to {output_file}')



if __name__ == '__main__':
    input_dir = 'data/prompt_output/'
    input_file_name = 'novelty_skills_30_gemini-2.0-flash_branchen_after_2024.json'
    input_file = input_dir + input_file_name
    output_dir = 'data/turbo_output/'
    output_file = output_dir + 'processed_' + str(DISTANCE_THRESHOLD) + '_' + input_file_name
    print(f'Processing {input_file}...')
    process_mismatch_skills(input_file, output_file)