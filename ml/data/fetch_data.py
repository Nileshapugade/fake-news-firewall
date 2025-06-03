from datasets import load_dataset
import pandas as pd
import os

def fetch_data():
    try:
        print("Loading LIAR dataset from Hugging Face...")
        dataset = load_dataset("liar", trust_remote_code=True)
        print("Dataset loaded successfully.")
        
        # Verify dataset structure
        if "train" not in dataset:
            raise ValueError("Dataset does not contain 'train' split")
        if "statement" not in dataset["train"].features or "label" not in dataset["train"].features:
            raise ValueError("Dataset missing 'statement' or 'label' fields")
        
        print("Creating DataFrame...")
        df = pd.DataFrame({
            "text": dataset["train"]["statement"],
            "label": dataset["train"]["label"]
        })
        
        # Ensure output directory exists
        os.makedirs("ml/data", exist_ok=True)
        output_path = "ml/data/liar_dataset.csv"
        
        print(f"Saving DataFrame to {output_path}...")
        df.to_csv(output_path, index=False)
        print(f"Successfully saved {output_path}")
        
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to fetch or save LIAR dataset: {str(e)}")

if _name_ == "_main_":
    try:
        df = fetch_data()
        print("Dataset fetching and saving completed.")
    except Exception as e:
        print(f"Error: {str(e)}")