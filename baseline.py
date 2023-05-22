import argparse
import csv
import json
import logging
import requests
import random

from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import List

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Read jsonl file containing LM-KBC data
def read_lm_kbc_jsonl(file_path: str):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Disambiguation baseline
def disambiguation_baseline(item):
    try:
        # If item can be converted to an integer, return it directly
        return int(item)
    except ValueError:
        # If not, proceed with the Wikidata search
        try:
            url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={item}&language=en&format=json"
            data = requests.get(url).json()
            # Return the first id (Could upgrade this in the future)
            return data['search'][0]['id']
        except:
            return item


# Read prompt templates from a CSV file
def read_prompt_templates_from_csv(file_path: str):
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        prompt_templates = {row['Relation']: row['PromptTemplate'] for row in reader}
    return prompt_templates

# Read train data from a CSV file
def read_train_data_from_csv(file_path: str):
    with open(file_path, "r") as file:
        train_data = [json.loads(line) for line in file]
    return train_data

# Create a prompt using the provided data
def create_prompt(subject_entity: str, relation: str, prompt_templates: dict, instantiated_templates: List[str], tokenizer, few_shot: int = 0, task: str = "fill-mask") -> str:
    prompt_template = prompt_templates[relation]
    if task == "text-generation":
        if few_shot > 0:
            random_examples = random.sample(instantiated_templates, min(few_shot, len(instantiated_templates)))
        else:
            random_examples = []
        few_shot_examples = "\n".join(random_examples)
        prompt = f"{few_shot_examples}\n{prompt_template.format(subject_entity=subject_entity)}"
    else:
        prompt = prompt_template.format(subject_entity=subject_entity, mask_token=tokenizer.mask_token)
    return prompt

def run(args):
    # Load the model
    model_type = args.model
    logger.info(f"Loading the model \"{model_type}\"...")
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = AutoModelForMaskedLM.from_pretrained(model_type)  if "bert" in model_type.lower()  else AutoModelForCausalLM.from_pretrained(model_type)
    task = "fill-mask" if "bert" in model_type.lower() else "text-generation"    
     
    
    # Read the prompt templates and train data from CSV files
    if task == "text-generation":
        pipe = pipeline(task=task, model=model, tokenizer=tokenizer, top_k=args.top_k, device=args.gpu, fp16=args.fp16) 
        logger.info(f"Reading question prompt templates from \"{args.question_prompts}\"...")
        prompt_templates = read_prompt_templates_from_csv(args.question_prompts)
    else:
        pipe = pipeline(task=task, model=model, tokenizer=tokenizer, top_k=args.top_k, device=args.gpu) 
        logger.info(f"Reading fill-mask prompt templates from \"{args.fill_mask_prompts}\"...")
        prompt_templates = read_prompt_templates_from_csv(args.fill_mask_prompts)
    # Instantiate templates with train data
    instantiated_templates = []
    if task == "text-generation":
        logger.info(f"Reading train data from \"{args.train_data}\"...")
        train_data = read_train_data_from_csv(args.train_data)
        logger.info("Instantiating templates with train data...")
        for row in train_data:
            relation = row['Relation']
            prompt_template = prompt_templates[relation]
            object_entities = row['ObjectEntities']
            answers = ', '.join(object_entities)
            instantiated_example = prompt_template.format(subject_entity=row["SubjectEntity"]) + f" {answers}"
            instantiated_templates.append(instantiated_example)


    # Load the input file
    logger.info(f"Loading the input file \"{args.input}\"...")
    input_rows = [json.loads(line) for line in open(args.input, "r")]
    logger.info(f"Loaded {len(input_rows):,} rows.")

    # Create prompts
    logger.info(f"Creating prompts...")
    prompts = [create_prompt(
    subject_entity=row["SubjectEntity"],
    relation=row["Relation"],
    prompt_templates=prompt_templates,
    instantiated_templates=instantiated_templates,
    tokenizer=tokenizer,
    few_shot=args.few_shot,
    task=task,
    ) for row in input_rows]

    # Run the model
    logger.info(f"Running the model...")
    if task == 'fill-mask':
        outputs = pipe(prompts, batch_size=args.batch_size)
    else:
        outputs = pipe(prompts, batch_size=args.batch_size, max_length=256)

    results = []
    for row, output, prompt in zip(input_rows, outputs, prompts):
        object_entities_with_wikidata_id = []
        if task == "fill-mask":
            for seq in output:
                if seq["score"] > args.threshold:
                    wikidata_id = disambiguation_baseline(seq["token_str"])
                    object_entities_with_wikidata_id.append(wikidata_id)
        else:
            # Remove the original prompt from the generated text
            qa_answer = output[0]['generated_text'].split(prompt)[-1].strip()
            qa_entities = qa_answer.split(", ")
            for entity in qa_entities:
                wikidata_id = disambiguation_baseline(entity)
                object_entities_with_wikidata_id.append(wikidata_id)


        result_row = {
            "SubjectEntityID": row["SubjectEntityID"],
            "SubjectEntity": row["SubjectEntity"],
            "ObjectEntitiesID": object_entities_with_wikidata_id,
            "Relation": row["Relation"],
        }
        results.append(result_row)

    # Save the results
    logger.info(f"Saving the results to \"{args.output}\"...")
    with open(args.output, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Model with Question and Fill-Mask Prompts")
    parser.add_argument("-m", "--model", type=str, default="bert-base-cased", help="HuggingFace model name (default: bert-base-cased)")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input test file (required)")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output file (required)")
    parser.add_argument("-k", "--top_k", type=int, default=10, help="Top k prompt outputs (default: 100)")
    parser.add_argument("-t", "--threshold", type=float, default=0.1, help="Probability threshold (default: 0.1)")
    parser.add_argument("-g", "--gpu", type=int, default=-1, help="GPU ID, (default: -1, i.e., using CPU)")
    parser.add_argument("-qp", "--question_prompts", type=str, required=True, help="CSV file containing question prompt templates (required)")
    parser.add_argument("-fp", "--fill_mask_prompts", type=str, required=True, help="CSV file containing fill-mask prompt templates (required)")
    parser.add_argument("-f", "--few_shot", type=int, default=5, help="Number of few-shot examples (default: 5)")
    parser.add_argument("--train_data", type=str, required=True, help="CSV file containing train data for few-shot examples (required)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for the model. (default:32)")
    parser.add_argument("--fp16", action="store_true", help="Enable 16-bit model (default: False). This is ignored for BERT.")

    args = parser.parse_args()

    run(args)
