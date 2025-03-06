import pandas as pd
from sentence_transformers import SentenceTransformer
import joblib
import os
import numpy as np


class NewsEmbeddingsProcessor:
    """
    A class to process news articles, generate embeddings using a pre-trained Sentence Transformer model,
    and save the embeddings into a pickle file.

    Attributes:
    -----------
    input_file : str
        The path to the CSV file containing processed news articles data.

    output_dir : str
        The directory where the embeddings file will be saved.

    model_name : str
        The name of the pre-trained Sentence Transformer model to use.

    model : SentenceTransformer
        The Sentence Transformer model instance.

    Methods:
    --------
    load_data() -> pd.DataFrame
        Loads the processed news articles from the CSV file.

    preprocess_data(df: pd.DataFrame) -> pd.DataFrame
        Prepares the data for embedding generation by filling missing values and combining text.

    generate_embeddings(df: pd.DataFrame) -> pd.DataFrame
        Generates embeddings for the news articles using the Sentence Transformer model.

    save_embeddings(df: pd.DataFrame)
        Saves the embeddings DataFrame to a pickle file.

    process_and_save_embeddings()
        Loads, processes, generates embeddings, and saves the embeddings to a file.
    """

    def __init__(
        self, input_file: str, output_dir: str, model_name: str = "all-MiniLM-L12-v2"
    ):
        """
        Initializes the NewsEmbeddingsProcessor class.

        Parameters:
        -----------
        input_file : str
            Path to the CSV file with processed news data.

        output_dir : str
            Directory where the output embeddings will be saved.

        model_name : str, optional
            The name of the pre-trained Sentence Transformer model (default is "all-MiniLM-L6-v2").
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.model_name = model_name
        self.model = self._load_model()

        os.makedirs(self.output_dir, exist_ok=True)

    def _load_model(self) -> SentenceTransformer:
        """
        Loads the Sentence Transformer model.

        Returns:
        --------
        SentenceTransformer
            The pre-trained model instance.
        """
        try:
            print(f"Loading model: {self.model_name}...")
            model = SentenceTransformer(self.model_name)
            return model
        except Exception as e:
            raise RuntimeError(f"Error loading the embedding model: {str(e)}")

    def load_data(self) -> pd.DataFrame:
        """
        Loads the processed news articles data from a CSV file.

        Returns:
        --------
        pd.DataFrame
            The DataFrame containing the processed news data.
        """
        print(f"Loading data from {self.input_file}...")
        return pd.read_csv(self.input_file)

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares the data for embedding generation by filling missing values and combining text.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing news articles.

        Returns:
        --------
        pd.DataFrame
            The preprocessed DataFrame with 'full_text' column containing combined title and body.
        """
        print("Preprocessing data...")
        df["category"] = df["category"].fillna("unknown")
        df["title"] = df["title"].fillna("")
        df["body"] = df["body"].fillna("")

        # Create the full text for embedding
        df["full_text"] = (
            df["title"] + " " + df["body"].apply(lambda x: " ".join(x.split()[:50]))
        )

        return df

    def generate_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates embeddings for the news articles using the Sentence Transformer model.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the preprocessed news articles.

        Returns:
        --------
        pd.DataFrame
            The DataFrame with a new column 'embeddings' containing the generated embeddings.
        """
        print("Generating embeddings for the news articles...")
        df["embeddings"] = df["full_text"].apply(
            lambda x: self.model.encode(str(x)) if isinstance(x, str) else np.zeros(384)
        )

        # Convert embeddings to lists to store them in the dataframe
        df["embeddings"] = df["embeddings"].apply(lambda x: x.tolist())

        # Drop rows where embeddings are NaN or empty
        df = df.dropna(subset=["embeddings"])

        return df

    def save_embeddings(self, df: pd.DataFrame):
        """
        Saves the embeddings DataFrame to a pickle file.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame with the embeddings to save.

        """
        output_file_path = os.path.join(self.output_dir, "embeddings_itens.pkl")
        print(f"Saving embeddings to {output_file_path}...")
        joblib.dump(df[["page", "title", "category", "embeddings"]], output_file_path)

    def process_and_save_embeddings(self):
        """
        Loads, processes, generates embeddings, and saves the embeddings to a file.
        """
        df = self.load_data()
        df = self.preprocess_data(df)
        df = self.generate_embeddings(df)
        print(df.head())
        self.save_embeddings(df)
        print(f"Embeddings generated and saved to {self.output_dir}")


# Usage Example:

if __name__ == "__main__":
    # Specify the input and output directories
    input_file = "../data/refined_itens_data/refined_itens_data.csv"
    output_dir = "../data/processed_itens_data"

    # Create an instance of NewsEmbeddingsProcessor
    embeddings_processor = NewsEmbeddingsProcessor(input_file, output_dir)

    # Process and save the embeddings
    embeddings_processor.process_and_save_embeddings()

oi = pd.read_csv("../data/refined_itens_data/refined_itens_data.csv")
oi.head()
