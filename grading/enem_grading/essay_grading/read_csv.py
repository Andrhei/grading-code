import csv
import pandas as pd

class CSVReader:
    def __init__(self, file_path):
        """
        Initialize the CSVReader with the path to the CSV file.
        :param file_path: Path to the CSV file.
        """
        self.file_path = file_path

    def read_data(self):
        """
        Reads the CSV file and returns the data as a list of dictionaries.
        Each dictionary represents a row, with keys as column headers.
        :return: List of dictionaries containing the CSV data.
        """
        data = []
        try:
            with open(self.file_path, mode='r', encoding='utf-8', errors='replace') as file:
                reader = csv.DictReader(file)
                for row in reader: 
                    data.append(row)
        except FileNotFoundError:
            print(f"Error: File not found at {self.file_path}")
        except Exception as e:
            print(f"An error occurred: {e}")
        return data

    def get_structure(self):
        """
        Reads the CSV file and returns the structure of the data (column headers).
        :return: List of column headers.
        """
        try:
            with open(self.file_path, mode='r', encoding='iso-8859-1') as file:
                reader = csv.DictReader(file)
                return reader.fieldnames
        except FileNotFoundError:
            print(f"Error: File not found at {self.file_path}")
        except Exception as e:
            print(f"An error occurred: {e}")
        return None
    
# Example usage of the CSVReader class
if __name__ == "__main__":
    # Specify the path to your CSV file
    file_path = "grading/enem_grading/text_grading/MICRODADOS_ENEM_2023.csv"

    # Create an instance of CSVReader
    csv_reader = CSVReader(file_path)

    # Read the data from the CSV file
    data = csv_reader.read_data()
    print("Data:")
    pd.DataFrame(data[:100]).to_csv("output.csv", index=False)  # Save to CSV for verification

    # Get the structure (column headers) of the CSV file
    structure = csv_reader.get_structure()
    print("Structure:")
    print(structure)