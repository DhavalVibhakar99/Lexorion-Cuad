"""
Download and extract the CUAD dataset.
Run: python -m src.data_pipeline.download_cuad
"""

import os
import json
import zipfile
import urllib.request
from pathlib import Path
from datasets import Dataset, DatasetDict


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Original source from the CUAD GitHub repo (found in the HuggingFace loading script)
DATA_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"


def download_cuad():
    """Download CUAD dataset from the original GitHub source."""
    
    print("=" * 60)
    print("Downloading CUAD dataset...")
    print("=" * 60)
    
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download the zip file
    zip_path = RAW_DIR / "data.zip"
    if not zip_path.exists():
        print(f"Downloading from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, str(zip_path))
        print("Download complete.")
    else:
        print("Zip file already exists, skipping download.")
    
    # Extract
    extract_dir = RAW_DIR / "cuad_extracted"
    if not extract_dir.exists():
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(str(extract_dir))
        print("Extraction complete.")
    
    # Find the JSON files inside the zip
    train_json = None
    test_json = None
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            fpath = os.path.join(root, f)
            if f == "train_separate_questions.json":
                train_json = fpath
            elif f == "test.json":
                test_json = fpath
            if f.endswith(".json"):
                print(f"  Found: {f}")
    
    if train_json is None:
        raise FileNotFoundError("Could not find train_separate_questions.json in the zip")
    
    # Parse SQuAD-format JSON into records
    def parse_squad_json(filepath):
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        
        records = []
        for article in data["data"]:
            title = article.get("title", "").strip()
            for para in article["paragraphs"]:
                context = para["context"].strip()
                for qa in para["qas"]:
                    records.append({
                        "id": qa["id"],
                        "title": title,
                        "context": context,
                        "question": qa["question"].strip(),
                        "answers": {
                            "text": [a["text"].strip() for a in qa["answers"]],
                            "answer_start": [a["answer_start"] for a in qa["answers"]],
                        },
                    })
        return records
    
    print("\nParsing train data...")
    train_records = parse_squad_json(train_json)
    
    splits = {"train": Dataset.from_list(train_records)}
    
    if test_json:
        print("Parsing test data...")
        test_records = parse_squad_json(test_json)
        splits["test"] = Dataset.from_list(test_records)
    
    dataset = DatasetDict(splits)
    
    print(f"\nDataset loaded successfully!")
    print(f"  Train split: {len(dataset['train'])} examples")
    if "test" in dataset:
        print(f"  Test split:  {len(dataset['test'])} examples")
    
    # Save to disk for offline use
    dataset.save_to_disk(str(RAW_DIR / "cuad_hf"))
    print(f"\nSaved HuggingFace dataset to {RAW_DIR / 'cuad_hf'}")
    
    # Group by contract
    train_data = [example for example in dataset["train"]]
    
    contracts = {}
    for example in train_data:
        title = example["title"]
        if title not in contracts:
            contracts[title] = []
        contracts[title].append({
            "question": example["question"],
            "context": example["context"],
            "answers": {
                "text": example["answers"]["text"],
                "answer_start": example["answers"]["answer_start"],
            },
            "is_impossible": len(example["answers"]["text"]) == 0,
        })
    
    print(f"\nExtracted {len(contracts)} unique contracts")
    
    # Save contract index
    contract_index = {
        title: {
            "num_questions": len(questions),
            "num_with_answers": sum(1 for q in questions if not q["is_impossible"]),
            "num_impossible": sum(1 for q in questions if q["is_impossible"]),
        }
        for title, questions in contracts.items()
    }
    
    with open(RAW_DIR / "contract_index.json", "w") as f:
        json.dump(contract_index, f, indent=2)
    
    print(f"Saved contract index to {RAW_DIR / 'contract_index.json'}")
    
    # Print summary stats
    total_with_answers = sum(v["num_with_answers"] for v in contract_index.values())
    total_impossible = sum(v["num_impossible"] for v in contract_index.values())
    
    print(f"\n{'=' * 60}")
    print(f"CUAD Dataset Summary")
    print(f"{'=' * 60}")
    print(f"  Total contracts:        {len(contracts)}")
    print(f"  Total QA pairs:         {len(train_data)}")
    print(f"  Pairs with answers:     {total_with_answers}")
    print(f"  Pairs without answers:  {total_impossible}")
    print(f"  Answer density:         {total_with_answers / len(train_data):.1%}")
    print(f"{'=' * 60}")
    
    return dataset


if __name__ == "__main__":
    download_cuad()