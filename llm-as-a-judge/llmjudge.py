"""
LLM-as-a-judge Evaluation Script

This script acts as an LLM judge evaluating the responses of a chatbot against
ground truth answers.  It interacts with a locally running Ollama server.
"""

import re
from pathlib import Path
from typing import Dict
import yaml
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# File paths
# BASELINE_FILE: evaluation dataset used as the "ground truth"
# TEST_FILE: chatbot dataset for the LLM judge to evaluate
BASELINE_FILE = Path("groundtruth.yaml")
TEST_FILE = Path("catalog_bot_output_log.yaml")
OUTPUT_FILE = Path("llm_as_a_judge_results.yaml")
# The name of the model as the backend to this LLM judge
MODEL_NAME = "nemotron-3-nano"

def clean_logger_file(file_path: Path) -> str:
    """Reads a file and keeps only the lines that do NOT start with 'HTTP Request:'"""
    with file_path.open("r", encoding="utf-8") as f:
        clean_lines = [
            line for line in f
            if not line.startswith("HTTP Request:")
        ]
    return "".join(clean_lines)


def main():
    """Main execution block for the LLM-as-a-judge evaluation.

    This will open the BASELINE_FILE and TEST_FILE and output the results of the llm-as-a-judge evaluation
    in the OUTPUT FILE.

    The TEST_FILE yaml is expected to be produced by Ciheim Brown's Chatbot Logger which is referred
    in this documentation simply as "the logger".
    """


    # Step 1: Parse the TEST_FILE yaml and store as a dictionary called logger_output_yaml
    # The logger captures some lines that start with "HTTP Request" that need to be cleaned out
    clean_yaml_string = clean_logger_file(TEST_FILE)
    data = yaml.safe_load(clean_yaml_string)

    # data is a list
    # data[n] is a dictionary

    # The logger yaml contains one item per query, so a loop through the list will loop over the queries
    logger_output_yaml: Dict[str, str] = {
        item.get("user_query", ""): item.get("ai_response", "")
        for item in data
        if item.get("user_query")
    }

    # Step 2: Parse the BASELINE_FILE yaml and store as a dictionary called groundtruth_yaml
    with BASELINE_FILE.open("r", encoding="utf-8") as f:
        groundtruth_yaml: Dict[str, str] = yaml.safe_load(f)

    #Step 3: LLM

    # Initialize ChatOllama client
    llm = ChatOllama(model=MODEL_NAME, temperature=0)

    # Set up LLM-as-a-judge prompt
    prompt_template = ChatPromptTemplate.from_template(
        "You are an expert evaluator scoring an AI assistant's accuracy.\n\n"
        "Query to the AI Assistant: {query}\n"
        "Ground Truth Answer: {correct_answer}\n"
        "AI Assistant's Answer: {response_text}\n\n"
        "Score the AI Assistant's answer on a scale from 1 to 10 (1=totally incorrect/hallucinated, "
        "10=perfectly accurate (word for word)). Provide your reasoning for giving the score.\n"
        "Format your response EXACTLY as the following YAML (do not include markdown backticks):\n"
        "score: <the score>\n"
        "reasoning: <the reasoning>"
    )
 
    # Create the chain
    chain = prompt_template | llm

    # Dictionary to store Judge evaluations
    judge_responses: Dict[str, str]= {}

    # Loop over the queries and evaluate
    for query, response_text in logger_output_yaml.items():
        correct_answer = groundtruth_yaml.get(query.strip())

        if not correct_answer:
            print(f"Warning: No ground truth for query '{query}'. Skipping.")
            continue

        # Ask the LLM / invoke the chain of prompt and Ollama
        eval_res = chain.invoke(
            {
                "query": query,
                "correct_answer": correct_answer,
                "response_text": response_text,
            }
        )

        content = eval_res.content.strip()
        # Remove potential markdown formatting from the LLM's response
        if content.startswith("```"):
            # Use regex to strip ```yaml and ``` from the start and end
            content = re.sub(r"^```(?:yaml)?\n|```$", "", content, flags=re.MULTILINE).strip()

        # Print the LLM's output
        print(f"Query: {query}\nEvaluation:\n{content}\n" + "-" * 40)

        # Save the query and evaluation in a dictionary
        try:
            judge_responses[query] = yaml.safe_load(content)
        except yaml/YAMLError as e:
            print(f"YAML Parsing Errror for query '{query}': {e}")
            judge_responses[query] = content

    # Outside of the LLM loop, aggregate evaluation results
    # This should be replaced by the logger
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        yaml.dump(
            judge_responses,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    print(f"Evaluation complete.  Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
