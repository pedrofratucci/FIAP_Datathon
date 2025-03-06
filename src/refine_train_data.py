import pandas as pd
import glob
import os


class DataProcessor:
    """
    A class to process and refine training data by reading, cleaning, and exploding certain columns.

    Attributes:
    -----------
    input_directory : str
        The directory where the raw data files are stored.

    output_directory : str
        The directory where the processed data will be saved.

    columns_to_explode : list
        A list of column names that need to be exploded.

    Methods:
    --------
    read_all_files(directory: str) -> pd.DataFrame
        Reads all CSV files in the specified directory and returns a combined DataFrame.

    split_column_values(series: pd.Series) -> pd.Series
        Splits a comma-separated string into a list of strings.

    process_data() -> pd.DataFrame
        Loads the data, splits the necessary columns, explodes the values, cleans, and returns the processed DataFrame.

    save_data(df: pd.DataFrame)
        Saves the processed DataFrame to the output directory.

    save_data_as_items(df: pd.DataFrame)
        Saves the processed DataFrame to the refined items data directory.
    """

    def __init__(
        self, input_directory: str, output_directory: str, columns_to_explode: list
    ):
        """
        Initializes the DataProcessor class with the input and output directories and columns to explode.

        Parameters:
        -----------
        input_directory : str
            The directory where the raw data files are stored.

        output_directory : str
            The directory where the processed data will be saved.

        columns_to_explode : list
            A list of column names that need to be exploded.
        """
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.columns_to_explode = columns_to_explode

        # Create output directory if it doesn't exist
        os.makedirs(self.output_directory, exist_ok=True)

    def read_all_files(self) -> pd.DataFrame:
        """
        Reads all CSV files in the specified directory and returns a combined DataFrame.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing all data from the CSV files.
        """
        all_files = glob.glob(os.path.join(self.input_directory, "*.csv"))

        if not all_files:
            raise FileNotFoundError(
                f"No CSV files found in the directory: {self.input_directory}"
            )

        # Read each file and combine them into a single DataFrame
        df_list = [pd.read_csv(file) for file in all_files]
        combined_df = pd.concat(df_list, ignore_index=True)

        return combined_df

    @staticmethod
    def split_column_values(series: pd.Series) -> pd.Series:
        """
        Splits a comma-separated string into a list of strings.

        Parameters:
        -----------
        series : pd.Series
            The column to split.

        Returns:
        --------
        pd.Series
            A series where each value is a list of strings.
        """
        return series.str.split(",")

    def process_data(self) -> pd.DataFrame:
        """
        Loads the data, splits the necessary columns, explodes the values, cleans,
        and returns the processed DataFrame.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the processed and exploded data.
        """
        # Read all data from the input directory
        df_users = self.read_all_files()

        # Loop through the columns to explode and apply the split function to each one
        for column in self.columns_to_explode:
            df_users[column] = self.split_column_values(df_users[column])

        # Explode each of these columns, creating new rows for each item
        df_exploded = df_users.explode(self.columns_to_explode)

        # Remove any leading/trailing spaces from the column values
        for column in self.columns_to_explode:
            df_exploded[column] = df_exploded[column].str.strip()

        return df_exploded

    def save_data(self, df: pd.DataFrame):
        """
        Saves the processed DataFrame to the output directory.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to save to a CSV file.
        """
        output_file_path = os.path.join(self.output_directory, "refined_train_data.csv")
        df.to_csv(output_file_path, index=False)
        print(f"Processed data saved to {output_file_path}")

    def save_data_as_items(self, df: pd.DataFrame):
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
        print(df.head)
        print(f"Processed data saved to {output_file_path}")


if __name__ == "__main__":
    # Define input and output directories
    input_directory = "../data/raw_train_data"
    output_directory = "../data/refined_train_data"

    # Define the columns to be exploded
    columns_to_explode = [
        "history",
        "timestampHistory",
        "numberOfClicksHistory",
        "timeOnPageHistory",
        "scrollPercentageHistory",
        "pageVisitsCountHistory",
    ]

    # Create an instance of DataProcessor
    data_processor = DataProcessor(
        input_directory, output_directory, columns_to_explode
    )

    # Process the data
    processed_data = data_processor.process_data()

    # Save the processed data to a CSV file
    data_processor.save_data(processed_data)

    # Save the processed data to the refined items data directory
    data_processor.save_data_as_items(processed_data)
