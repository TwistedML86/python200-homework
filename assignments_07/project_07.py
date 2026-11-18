from pathlib import Path
from scipy.stats import pearsonr

import pandas as pd
from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIServerModel, tool

DATA_DIR = Path("happiness_project")

COLUMN_NAMES = {
    "Country": "country",
    "Regional indicator": "region",
    "Happiness score": "happiness_score",
    "Ladder score": "happiness_score",
    "GDP per capita": "gdp_per_capita",
}

df = None


@tool
def load_happiness_data() -> dict:
    """Load and combine the yearly World Happiness CSV files."""
    global df

    frames = []
    for path in sorted(DATA_DIR.glob("world_happiness_*.csv")):
        year_df = pd.read_csv(path, sep=";", decimal=",")
        year_df = year_df.rename(columns=COLUMN_NAMES)
        year_df["year"] = int(path.stem.split("_")[-1])
        frames.append(year_df)

    df = pd.concat(frames, ignore_index=True)
    return {"shape": df.shape, "columns": df.columns.tolist()}

@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for one column in the loaded dataset.

    Args:
        column: Name of the column to summarize.
    """
    if df is None:
        return {"error": "Load the dataset first."}

    if column not in df.columns:
        return {"error": f"Column not found: {column}"}

    return df[column].describe().to_dict()

@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Return the Pearson correlation and p-value for two numeric columns.

    Args:
        col1: Name of the first numeric column.
        col2: Name of the second numeric column.
    """
    if df is None:
        return {"error": "Load the dataset first."}

    for column in (col1, col2):
        if column not in df.columns:
            return {"error": f"Column not found: {column}"}
        if not pd.api.types.is_numeric_dtype(df[column]):
            return {"error": f"Column must be numeric: {column}"}

    pairs = df[[col1, col2]].dropna()
    if len(pairs) < 2:
        return {"error": "At least two complete pairs are required."}
    if pairs[col1].nunique() < 2 or pairs[col2].nunique() < 2:
        return {"error": "Both columns need more than one distinct value."}

    r, p = pearsonr(pairs[col1], pairs[col2])
    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(float(r), 4),
        "p_value": round(float(p), 4),
    }
@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Rank countries by a numeric column for a specific year.

    Args:
        column: Numeric column to rank countries by.
        year: Year to use for the ranking.
        n: Number of countries to return.

    Returns:
        A dictionary containing the ranked countries or an error message.
    """
    if df is None:
        return {"error": "Load the dataset first."}
    if column not in df.columns:
        return {"error": f"Column not found: {column}"}
    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"error": f"Column must be numeric: {column}"}
    if n < 1:
        return {"error": "n must be at least 1."}

    year_rows = df.loc[df["year"] == year]
    if year_rows.empty:
        return {"error": f"No data found for {year}."}

    ranked = (
        year_rows[["country", column]]
        .dropna(subset=[column])
        .sort_values(by=column, ascending=False)
        .head(n)
    )
    return {"top_countries": ranked.to_dict(orient="records")}

if __name__ == "__main__":
    load_dotenv()
    Path("outputs").mkdir(exist_ok=True)

    model = OpenAIServerModel(model_id="gpt-4o-mini")

    agent = CodeAgent(
        tools=[
            load_happiness_data,
            summarize_column,
            compute_correlation,
            get_top_n_countries,
        ],
        model=model,
        instructions=(
            "You are a data analyst assistant for the World Happiness dataset. "
            "Use the available tools when they fit the question. "
            "Write Python when the tools are not enough, such as for custom plots. "
            "The yearly CSVs are in happiness_project/ relative to this script's "
            "working directory. Save plots to outputs/ without opening a window."
        ),
        additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
        max_steps=8,
    )

    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        (
            "Read the yearly happiness_project/world_happiness_YYYY.csv files "
            "using pandas with sep=';' and decimal=','. Extract the year from "
            "each filename. Use 'Regional indicator' for the region and "
            "'Happiness score' for the score, or 'Ladder score' in 2024. "
            "Average the scores by year and region. Plot one line per region "
            "and save it to outputs/happiness_by_region.png without calling "
            "plt.show(). The load_happiness_data tool returns only metadata."
        )
    ]

    for query in queries:
        print(f"\n--- Query: {query} ---")
        print(agent.run(query, reset=False))

        # Task 4: My own questions
    custom_queries = [
        "Which three countries had the highest happiness scores in 2024?",
        (
            "How did Mexico's happiness score change from 2015 to 2024? "
            "Read those two CSVs with pandas using sep=';' and decimal=','. "
            "Use 'Happiness score' in 2015 and 'Ladder score' in 2024. "
            "Report both scores and subtract the 2015 score from the 2024 score."
        ),
    ]

    for question in custom_queries:
        print(f"\n--- My question: {question} ---")
        print(agent.run(question, reset=False))

# --- Task 4: Custom query observations ---
# 1. The agent tried get_top_n_countries with the raw column name
#    "Ladder score". It then read the 2024 CSV directly and ranked
#    Finland, Denmark, and Iceland as the top three.
# 2. The agent wrote pandas code to read the 2015 and 2024 files,
#    find Mexico's scores, and calculate the change: -0.512 points.

# --- Task 5: Reflection ---
# 1. The agent used compute_correlation and reported r = 0.6313.
#    It marked the result significant but didn't explain that the
#    relationship was positive or that the p-value rounded to 0.0.
# 2. The chart surprised me: the agent saved it even though it kept
#    making formatting errors when trying to report that it was done.
#    Opening the PNG let me verify what actually happened.
# 3. I'd add a tool that returns a country's happiness scores by year.
#    It could answer questions like "How did Mexico's score change
#    over time?" without making the agent reread the raw CSVs.