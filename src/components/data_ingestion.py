import os
import pandas as pd


class DataIngestion:
    def __init__(self):
        self.input_file = "data/online_retail_data.xlsx"
        self.output_path = "artifacts/raw_data.csv"

    def initiate_data_ingestion(self):
        # Load both sheets
        df1 = pd.read_excel(self.input_file, sheet_name=0)
        df2 = pd.read_excel(self.input_file, sheet_name=1)

        # Combine the sheets
        df = pd.concat([df1, df2], ignore_index=True)

        # Create artifacts folder if it doesn't exist
        os.makedirs("artifacts", exist_ok=True)

        # Save combined dataset
        df.to_csv(self.output_path, index=False)

        print("Data Ingestion Completed")
        print("Dataset Shape:", df.shape)

        return self.output_path