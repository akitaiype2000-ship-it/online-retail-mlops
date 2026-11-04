import os
import pandas as pd


class DataTransformation:

    def __init__(self):
        self.input_path = "artifacts/validated_data.csv"
        self.output_path = "artifacts/customer_data.csv"

    def initiate_data_transformation(self):

        print("Reading validated dataset...")

        df = pd.read_csv(self.input_path)

        # -----------------------------
        # Create Revenue column
        # -----------------------------
        print("Creating Revenue feature...")
        df["Revenue"] = df["Quantity"] * df["UnitPrice"]

        # -----------------------------
        # Total Revenue
        # -----------------------------
        print("Calculating total revenue...")

        total_revenue = (
            df.groupby("CustomerID")["Revenue"]
            .sum()
            .reset_index(name="total_revenue")
        )

        # -----------------------------
        # Customer Behaviour
        # -----------------------------
        print("Calculating customer behaviour...")

        customer_behavior = (
            df.groupby("CustomerID")
            .agg(
                number_of_purchases=("InvoiceNo", "nunique"),
                purchase_frequency=("InvoiceNo", "count"),
                average_order_value=("Revenue", "mean")
            )
            .reset_index()
        )

        # -----------------------------
        # Recency
        # -----------------------------
        print("Calculating recency...")

        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

        last_purchase = (
            df.groupby("CustomerID")["InvoiceDate"]
            .max()
            .reset_index()
        )

        last_purchase["recency_days"] = (
            snapshot_date - last_purchase["InvoiceDate"]
        ).dt.days

        # -----------------------------
        # Merge Features
        # -----------------------------
        print("Merging customer features...")

        customer_df = total_revenue.merge(
            customer_behavior,
            on="CustomerID"
        )

        customer_df = customer_df.merge(
            last_purchase[["CustomerID", "recency_days"]],
            on="CustomerID"
        )

        # -----------------------------
        # Create Target Variable
        # -----------------------------
        print("Creating target variable...")

        revenue_threshold = customer_df["total_revenue"].quantile(0.75)

        customer_df["high_value_repeat"] = (
            (customer_df["number_of_purchases"] > 1) &
            (customer_df["total_revenue"] > revenue_threshold)
        ).astype(int)

        # -----------------------------
        # Save Transformed Dataset
        # -----------------------------
        os.makedirs("artifacts", exist_ok=True)

        customer_df.to_csv(self.output_path, index=False)

        print("Data Transformation Completed")
        print("Final Shape:", customer_df.shape)

        return self.output_path