import pandas as pd
from typing import Optional

def class_imbalance_handle(
    real_df: pd.DataFrame,
    fake_df: pd.DataFrame,
    k: int,
    seed: Optional[int] = None) -> pd.DataFrame:

    seed = 2025 if seed is None else seed
      
    n_real = len(real_df)
    start_idx = k * n_real
    end_idx = start_idx + n_real

    # Handle wrap-around if the slice exceeds the number of fake samples
    if end_idx > len(fake_df):
        remaining = fake_df[start_idx:]
        n_remaining = len(remaining)

        # Sample additional fake data from the start to match the required slice
        extra = fake_df[:start_idx].sample(n=n_real - n_remaining, replace=False, random_state=seed)
        fake_slice = pd.concat([remaining, extra], axis=0).reset_index(drop=True)

    else:
        fake_slice = fake_df[start_idx:end_idx].reset_index(drop=True)

    # Combine real and fake samples, then shuffle
    train_df = pd.concat([real_df, fake_slice], ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return train_df