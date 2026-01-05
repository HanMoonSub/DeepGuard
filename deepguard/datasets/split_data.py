import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(
    meta_df: pd.DataFrame,
    label_key: str,
    origin_vid: str, 
    seed: int = 2025,
    debug: bool = False,
    test_size: float = 0.2,
    debug_ratio: float = 0.1,
    balance_val: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    # Split real and fake(0: REAL, 1: FAKE)
    
    assert meta_df[label_key].dtype in [int, float], "Label must be 0 or 1, not string"
    
    df_real = meta_df[meta_df[label_key] == 0].reset_index(drop=True)
    df_fake = meta_df[meta_df[label_key] == 1].reset_index(drop=True)

    # Split real Images
    real_train, real_val = train_test_split(df_real, test_size=test_size, random_state=seed, shuffle=True)

    # Match fake images to corresponding real train/val
    fake_train = df_fake[df_fake[origin_vid].isin(real_train[origin_vid].unique())].reset_index(drop=True)
    fake_val = df_fake[df_fake[origin_vid].isin(real_val[origin_vid].unique())].reset_index(drop=True)

    # Sample fake_val to match real_val length
    if balance_val:
        fake_val = fake_val.sample(len(real_val), random_state=seed)

    # Concatenate real + fake
    train_df = pd.concat([real_train, fake_train], axis=0)
    valid_df = pd.concat([real_val, fake_val], axis=0)

    # Debug mode: downsample dataset
    if debug:
        n_train = max(1, int(len(train_df) * debug_ratio))
        n_valid = max(1, int(len(valid_df) * debug_ratio))
        train_df = train_df.sample(n=n_train, random_state=seed).reset_index(drop=True)
        valid_df = valid_df.sample(n=n_valid, random_state=seed).reset_index(drop=True)


    # Shuffle final datasets
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    valid_df = valid_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return train_df, valid_df