import pandas as pd
from api.models.preprocess import preprocess_text

def preprocess_data(input_file, output_file):
    df = pd.read_csv(input_file)
    df["text"] = df["text"].apply(preprocess_text)
    # Map LIAR labels (0-5) to numerical labels: 0=fake, 1=misleading, 2=credible
    label_map = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
    df["label"] = df["label"].map(label_map)
    df.to_csv(output_file, index=False)

if _name_ == "_main_":
    preprocess_data("ml/data/liar_dataset.csv", "ml/data/processed_data.csv")
