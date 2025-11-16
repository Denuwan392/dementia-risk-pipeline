# utils.py
import pandas as pd

def engineer_features(df):
    """Apply the same engineering as in the notebook."""
    df = df.copy()
    # AGE_GROUP
    df['AGE_GROUP'] = pd.cut(
        df['AGE'],
        bins=[0, 64, 74, 84, 130],
        labels=['65-', '65-74', '75-84', '85+'],
        right=False
    )
    # EDUC_GROUP
    df['EDUC_GROUP'] = pd.cut(
        df['EDUC'],
        bins=[0, 8, 12, 16, 100],
        labels=['<HS', 'HS', 'Some College', 'College+'],
        right=False
    )
    return df