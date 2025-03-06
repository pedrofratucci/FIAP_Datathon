import pandas as pd
import glob
import os


class ItemLoader:
    """
    A class to load and process item data from CSV files stored in a specified directory.

    Attributes:
    -----------
    data_dir : str
        The directory where the raw item CSV files are stored.

    Methods:
    --------
    extract_category(url: str) -> str
        Extracts the category from a given URL.

    load_items() -> pd.DataFrame
        Loads all the CSV files from the specified directory, extracts categories from URLs,
        drops duplicates based on the 'page' column, and returns a DataFrame with the cleaned data.

    save_data(df: pd.DataFrame)
        Saves the processed DataFrame to the refined items data directory.
    """

    def __init__(self, data_dir: str):
        """
        Initializes the ItemLoader class with the directory containing the CSV files.

        Parameters:
        -----------
        data_dir : str
            The directory where the raw item CSV files are stored.
        """
        self.data_dir = data_dir
        # Ensure the directory exists
        os.makedirs(self.data_dir, exist_ok=True)

    @staticmethod
    def extract_category(url: str) -> str:
        """
        Extracts the category from a given URL.

        Parameters:
        -----------
        url : str
            The URL to extract the category from.

        Returns:
        --------
        str
            The category (e.g., 'politica', 'economia', etc.) or 'geral' if no category is found.
        """
        possible_categories = [
            "politica",
            "economia",
            "esporte",
            "ciencia",
            "mundo",
            "tecnologia",
            "entretenimento",
        ]
        for category in possible_categories:
            if f"/{category}/" in url:
                return category
        return "geral"

    def load_items(self) -> pd.DataFrame:
        """
        Loads all CSV files from the specified directory, extracts categories from URLs,
        drops duplicates based on the 'page' column, and returns a cleaned DataFrame.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the combined data from all CSV files, with an added 'category' column.
        """
        # Get all CSV files from the specified directory
        item_files = glob.glob(f"{self.data_dir}/*.csv")

        if not item_files:
            raise FileNotFoundError(
                f"No CSV files found in the directory: {self.data_dir}"
            )

        # Load and concatenate all CSV files
        df_items = pd.concat(
            [pd.read_csv(f, encoding="utf-8") for f in item_files], ignore_index=True
        )

        # Extract categories from URLs and create a new column 'category'
        df_items["category"] = df_items["url"].astype(str).apply(self.extract_category)

        # Remove duplicate entries based on the 'page' column
        df_items.drop_duplicates(subset=["page"], inplace=True)

        return df_items

    def save_data(self, df: pd.DataFrame):
        """
        Saves the processed DataFrame to the refined items data directory.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to save to a CSV file.
        """
        items_output_directory = "../data/refined_itens_data"
        os.makedirs(items_output_directory, exist_ok=True)
        output_file_path = os.path.join(
            items_output_directory, "refined_itens_data.csv"
        )
        df.to_csv(output_file_path, index=False)
        print(f"Processed data saved to {output_file_path}")


if __name__ == "__main__":
    # Specify the directory containing the raw item data
    input_directory = "../data/raw_itens_data"

    # Create an instance of ItemLoader
    item_loader = ItemLoader(input_directory)

    # Load the items into a DataFrame
    df_items = item_loader.load_items()

    # Display the first few rows of the DataFrame
    print(df_items.head())

    # Save the processed data to a CSV file
    item_loader.save_data(df_items)
