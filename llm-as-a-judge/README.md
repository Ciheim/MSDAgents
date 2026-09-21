# LLM-as-a-Judge Evaluation Script

This directory contains a Python script (`llmjudge.py`) that uses a locally running Large Language Model (LLM) to act as an automated "judge". The script evaluates a chatbot's responses against human-generated ground truth answers using [LangChain](https://python.langchain.com/) and [Ollama](https://ollama.com/).

## Overview

`llmjudge.py` performs the following tasks:
1. **Reads Test Data**: Loads the chatbot's output log (`catalog_bot_output_log.yaml`), automatically cleaning out irrelevant HTTP request lines.
2. **Reads Ground Truth**: Loads the baseline dataset (`groundtruth.yaml`).
3. **Evaluates**: Passes the user query, ground truth answer, and chatbot answer to a local LLM (default: `nemotron-3-nano`, but can be changed) and asks it to score the response (1-10) and provide reasoning.
4. **Outputs Results**: Parses the LLM's YAML-formatted feedback and compiles it into a final output file (`llm_as_a_judge_results.yaml`).

## Prerequisites

### 1. Install Ollama & Pull the Model
Ensure you have [Ollama](https://ollama.com/) installed and running locally. You will need to pull the model you intend to use. For example:
```bash
ollama pull nemotron-3-nano
```

### 2. Install Python Dependencies
`llmjudge.py` requires Python 3.8+ and the following packages:
```bash
pip install langchain-ollama langchain-core
```
These packages are included in this repository's pyproject.toml

## Required Input Files

`llmjudge.py` expects two YAML files to be present in the same directory:

### `groundtruth.yaml`
A simple key-value mapping of user queries to their correct, human-approved answers.
```yaml
"What is the capital of France?": "The capital of France is Paris."
"How do I reset my password?": "Click on 'Forgot Password' on the login screen."
```

### `catalog_bot_output_log.yaml`
The output from Ciheim Brown's chatbot logger. It should be a list of dictionaries containing `user_query` and `ai_response`.
```yaml
HTTP Request: POST http://127.0.0.1:11434/api/embed "HTTP/1.1 200 OK"
HTTP Request: POST http://127.0.0.1:11434/api/embed "HTTP/1.1 200 OK"
HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
- timestamp: '2026-05-19T09:11:42.471930'
  llm_model: llama3.2
  system_prompt: 'You are a chatbot who answers questions

    Context:

    {context}'
  user_query: What is the capital of France?
  ai_response: 'Paris is the capital.'
  retrieved_files:
  - source: ../dir/doc/Paris.rst
    similarity_score: 0.4691466188389781
- timestamp: '2026-05-19T10:11:42.471930'
  llm_model: llama3.2
  system_prompt: 'You are a chatbot who answers questions

    Context:

    {context}'
  user_query: How do I reset my password?
  ai_response: 'I'm not sure how to do that'
  retrieved_files:
  - source: ../dir/doc/pass.rst
    similarity_score: 0.5691466188389781

```
*(Note: `llmjudge.py` automatically filters out lines starting with `HTTP Request:` before parsing.)*

## Usage

1. Open `llmjudge.py` and verify that the `MODEL_NAME` variable matches the model you have pulled via Ollama (e.g., `MODEL_NAME = "nemotron-3-nano"`).
2. Run the script:
```bash
python llmjudge.py
```

## Output

`llmjudge.py` generates a file named `llm_as_a_judge_results.yaml` containing the score and reasoning for each evaluated query.

Example Output:
```yaml
What is the capital of France?:
  score: 10
  reasoning: The AI correctly identified Paris as the capital, which perfectly matches the ground truth.
How do I reset my password?:
  score: 1
  reasoning: The AI failed to provide the required steps to reset the password as detailed in the ground truth.
```

## Troubleshooting
* **YAML Parsing Errors:** If the LLM judge hallucinates formatting and fails to return valid YAML, the script will catch the error and dump the raw output into the results file.
* **Missing Ground Truth:** If a query in the test file is not found in the ground truth file, the script will log a warning to the console and skip evaluating that query.
