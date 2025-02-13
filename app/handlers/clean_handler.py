import pandas as pd
import io

def clean_train_request_csv(csv_content: str) -> str:
    try:
        df = pd.read_csv(io.StringIO(csv_content))

        column_averages = df.mean(numeric_only=True)
        df.fillna(column_averages, inplace=True)

        cleaned_csv = io.StringIO()
        df.to_csv(cleaned_csv, index=False)
        
        return cleaned_csv.getvalue()

    except Exception as e:
        print(f"Error during CSV cleaning: {e}")
        return None
