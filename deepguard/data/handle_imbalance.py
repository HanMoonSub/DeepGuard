import pandas as pd
from typing import Optional

def class_imbalance_handle(
    real_df: pd.DataFrame,
    fake_df: pd.DataFrame,
    k: int,
    seed: int = 2025):
    """
    Constructs a balanced training dataset by combining real data with a specific slice of fake data.

    This function implements a sliding-window approach to select synthetic (fake) samples 
    to match the count of real samples. It ensures a 1:1 class distribution for 
    iterative training or cross-validation scenarios.

    Args:
        real_df (pd.DataFrame): The original/minority class dataframe.
        fake_df (pd.DataFrame): The synthetic/majority class dataframe to sample from.
        k (int): The fold index used to determine the starting position of the fake data slice.
        seed (int, optional): Random seed for reproducibility during sampling and shuffling. Defaults to 2025.

    """

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