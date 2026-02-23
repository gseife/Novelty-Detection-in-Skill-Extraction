# A Scalable Pipeline for Novelty Detection in Skill Extraction Using Large Language Models
Repository for LREC Paper 2026

## Abstract
The rapid evolution of the labour market requires skill ontologies to be continuously updated, but manually identifying
emerging skills in job advertisements is highly labour-intensive. This paper presents a scalable, multi-stage pipeline
for automated novelty detection in skill extraction. The system combines Large Language Models (LLMs) for
candidate generation, a re-matching and threshold-based filtering module (“Turbo”), and a two-step aggregation
process that merges string-based and embedding-based clustering. Experiments on Swiss job advertisement
datasets using GPT-4o, Gemini-2.0-flash, and DeepSeek-V3 show that the pipeline effectively reduces noise and
manual curation effort: Turbo filtering lowered false positives by 82%, and aggregation reduced the number of items
requiring review by 97%. Among the tested models, Gemini-2.0-flash achieved the highest precision, detecting
genuine novelties with up to 73% accuracy. These findings demonstrate the pipeline’s potential as an efficient tool for
maintaining dynamic skill ontologies.

## Setup 
A requirments.txt file is provided which contains all the needed libaries for the execution of the programm.

## Usage

### Prompt(prompt.py)
To run the prompt you will need an **API key** for the LLM service. 

### Turbo(post_processing.py)
The turbo can't be executed directly because the connecting API is an internal Program of x28 which isn't accesible to the public. To execute the turbo a matching api call needs to be implemented and the barrier needs to be adjusted accordingly. 

### Aggregation(aggregation.py)
To run the aggregation you will need an **API key** for the embedding service.

You can customize the behavior of the aggregation by adjusting the parameters in the `full_aggregation_pipeline` function:

- **`count_threshold`**: Controls how many low-frequency (long-tail) entries are grouped together.  
  A higher value results in more entries being considered part of the long tail.

- **`eps`**: Sets the maximum distance between embeddings for them to be considered semantically similar.  
  A higher value allows for looser semantic grouping.

- **`min_samples`**: Defines the minimum number of similar skills required to form a cluster.  
  A higher value results in fewer but more robust clusters.

- **`grouping_threshold`**: Determines how syntactically similar skills must be to be grouped.  
  A higher value enforces stricter similarity based on wording.