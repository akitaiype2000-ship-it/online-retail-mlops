import os
import pandas as pd


class DataValidation:

    def __init__(self):
        self.input_path = "artifacts/raw_data.csv"
        self.output_path = "artifacts/validated_data.csv"

    def initiate_data_validation(self):

        print("Reading raw dataset...")
        df = pd.read_csv(self.input_path)

        # Rename columns
        df.rename(columns={
            "Invoice": "InvoiceNo",
            "Price": "UnitPrice",
            "Customer ID": "CustomerID"
        }, inplace=True)

        # Remove missing CustomerID
        print("Removing missing CustomerID...")
        df = df.dropna(subset=["CustomerID"])

        # Convert CustomerID to integer
        df["CustomerID"] = df["CustomerID"].astype(int)

        # Remove cancelled invoices
        print("Removing cancelled invoices...")
        df = df[
            (df["Quantity"] > 0) &
            (~df["InvoiceNo"].astype(str).str.startswith("C"))
        ]

        # Remove duplicate rows
        print("Removing duplicate rows...")
        df = df.drop_duplicates()

        # Remove invalid prices
        print("Removing invalid prices...")
        df = df[df["UnitPrice"] > 0]

        os.makedirs("artifacts", exist_ok=True)

        df.to_csv(self.output_path, index=False)

        print("Data Validation Completed")
        print("Final Shape:", df.shape)

        return self.output_path