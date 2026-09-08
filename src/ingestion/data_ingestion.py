import pandas as pd


class DataIngestion:

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def load_data(self):
        df = pd.read_csv(self.input_path)
        return df

    def save_data(self, df):
        df.to_csv(self.output_path, index=False)

    def run(self):
        df = self.load_data()
        self.save_data(df)
        return df


if __name__ == "__main__":

    ingestion = DataIngestion(
        "data/raw/Telco-Customer-Churn.csv",
        "data/processed/processed_telco.csv"
    )

    df = ingestion.run()

    print("Data ingestion completed successfully!")
    print("Dataset shape:", df.shape)