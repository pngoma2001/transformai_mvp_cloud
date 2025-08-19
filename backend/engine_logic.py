import pandas as pd
from engine import analyze


def run_analysis(df: pd.DataFrame):
    """Wrapper around engine.analyze for backend use."""
    return analyze(df)
