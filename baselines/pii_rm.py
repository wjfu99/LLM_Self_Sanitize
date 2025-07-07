from typing import List, Tuple
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import datasets
import argparse
from tqdm import tqdm
from copy import deepcopy

parser = argparse.ArgumentParser(description="Anonymize text using Presidio")
parser.add_argument("--output_dir", type=str, default="./results/response", help="The output directory for the dataset")
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-13b-chat-hf", help="The model name to use")
parser.add_argument("--azure_endpoint", type=str, default="https://self-sanitize.cognitiveservices.azure.com/", help="Azure Text Analytics endpoint")
parser.add_argument("--azure_key", type=str, default="CcnQfd0PlZIj5m0EMQdJ4XwkDJkAVBpwQTAIPSStpR4TeqeXI741JQQJ99BFACYeBjFXJ3w3AAAaACOG66Mr", help="Azure Text Analytics key")
parser.add_argument("--azure_threshhold", type=float, default=0.4, help="Confidence threshold for Azure PII detection")


args = parser.parse_args()

def anonymize_presidio(
    to_anon: str,
    remove_entities: List[str] = [
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "LOCATION",
        "NRP",
    ],
) -> str:
    res = presidio_analyzer.analyze(text=to_anon, entities=remove_entities, language="en")
    anon_text = presidio_anonymizer.anonymize(
        text=to_anon,
        analyzer_results=res,
    )
    return anon_text.text

def splice(
    text: str, start: int, end: int, offset: int, replacement: str
) -> Tuple[str, int]:
    new_offset = offset + len(replacement) - (end - start)

    return text[: start + offset] + replacement + text[end + offset :], new_offset

def anonymize_str(
    input_str: str,
    entities: List[Tuple[str, int, int, str]] = [],
):
    """Anonymizes a string by replacing all entities with <ENTITY_TYPE> tags.

    Args:
        input_str (str): The string to anonymize
        include (bool, optional): Sorted list of entities to remove


    """

    text = input_str
    copy_str = text
    offset = 0
    last_end = -1
    for ent in entities:
        start = ent[1] + offset
        end = ent[2] + offset

        relevant_section = copy_str[start:end]
        # assert relevant_section == ent[0], "Entity does not match text"
        copy_str, offset = splice(
            copy_str,
            ent[1],
            ent[2],
            offset,
            ent[3],
        )

    return copy_str

def anonymize_azure(
    text: str,
    remove_entities: List[str] = [
        "Person",
        "PersonType",
        "Location",
        "Organization",
        "Event",
        "Address",
        "PhoneNumber",
        "Email",
        "URL",
        "IP",
        "DateTime",
        ("Quantity", ["Age", "Currency", "Number"]),
    ],
    use_entities: bool = True,
) -> str:
    res = azure_anonymizer.recognize_entities([text])[0]
    
    anonymization_requests = []
    for entity in res.entities:
        need_removal = False
    
        for rme in remove_entities:
            if isinstance(rme, tuple):
                if entity.category == rme[0] and entity.subcategory in rme[1]:
                    need_removal = True
                    break
            else:
                if entity.category == rme:
                    need_removal = True
                    break
    
        if not need_removal:
            continue
    
        if entity.confidence_score < args.azure_threshhold:
            continue
    
        if use_entities:
            repl_text = (
                f"<{entity.category.upper()}>"
                if not entity.subcategory
                else f"<{entity.subcategory.upper()}>"
            )
        else:
            repl_text = "*" * len(entity.text)
    
        anonymization_requests.append(
            (
                entity.text,
                entity.offset,
                entity.offset + entity.length,
                repl_text,
            )
        )
    
    anon_text = anonymize_str(
        text,
        anonymization_requests,
    )
    return anon_text

presidio_analyzer = AnalyzerEngine()
presidio_anonymizer = AnonymizerEngine()
azure_anonymizer = TextAnalyticsClient(
    endpoint=args.azure_endpoint,
    credential=AzureKeyCredential(args.azure_key)
)

original_dataset = datasets.load_from_disk(f"{args.output_dir}/original/{args.model_name}")
presidio_dataset = deepcopy(original_dataset)
azure_dataset = deepcopy(original_dataset)

for type in original_dataset.keys():
    dataset = original_dataset[type]
    presidio_anonymied = []
    azure_anonymied = []
    for entry in tqdm(dataset):
        presidio_anonymied.append(anonymize_presidio(entry["response"]))
        azure_anonymied.append(anonymize_azure(entry["response"]))
    presidio_dataset[type] = presidio_dataset[type].remove_columns("response")
    presidio_dataset[type] = presidio_dataset[type].add_column("response", presidio_anonymied)
    azure_dataset[type] = azure_dataset[type].remove_columns("response")
    azure_dataset[type] = azure_dataset[type].add_column("response", azure_anonymied)
        
presidio_dataset.save_to_disk(f"{args.output_dir}/anonymization_presidio/{args.model_name}")
azure_dataset.save_to_disk(f"{args.output_dir}/anonymization_azure/{args.model_name}")