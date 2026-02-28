import pandas as pd
from sklearn.model_selection import train_test_split, GroupKFold

def split_data(
                meta_df: pd.DataFrame,
                seed: int = 2025,
                label_col: str = 'label',
                ori_vid: str = 'ori_vid', 
                dataset: str = 'kodf',
                group_col: str = "pid",
                test_size: float = 0.2,
                balance_val: bool = True,
                debug: bool = False,
                debug_ratio: float = 0.1,
) :
    """Splits the dataset into training and validation sets based on original video IDs.

    This function ensures that all fake videos derived from the same original video 
    stay within the same split (Train or Val) to prevent data leakage. It first 
    splits the real videos and then maps the corresponding fake videos.

    Args:
        meta_df: The complete metadata DataFrame containing labels and video IDs.
        seed: Random seed for reproducibility. Defaults to 2025.
        label_col: Name of the column containing labels (0 for REAL, 1 for FAKE).
        ori_vid: Name of the column identifying the original video source.
        test_size: Proportion of the dataset to include in the validation split.
        balance_val: If True, samples the fake validation set to match the 
            length of the real validation set.
        debug: If True, downsamples the final datasets for quick testing.
        debug_ratio: The ratio of data to keep when debug mode is enabled.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing (train_df, valid_df).
    """
    
    # Split real and fake(0: REAL, 1: FAKE)
    
    assert meta_df[label_col].dtype in [int, float], "You need to label encoding {'FAKE': 1, 'REAL': 0}"
    
    df_real = meta_df[meta_df[label_col] == 0].reset_index(drop=True)
    df_fake = meta_df[meta_df[label_col] == 1].reset_index(drop=True)

    # Split real Images
    if dataset.lower() != "kodf":
        real_train, real_val = train_test_split(df_real, test_size=test_size, random_state=seed, shuffle=True)
    else: 
        df_real_shuffled = df_real.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        n_splits = int(1 / test_size)
        gkf = GroupKFold(n_splits=n_splits)
        
        splits = list(gkf.split(df_real_shuffled, groups=df_real_shuffled[group_col]))
        train_idx, val_idx = splits[0]
        
        real_train = df_real_shuffled.iloc[train_idx]
        real_val = df_real_shuffled.iloc[val_idx]
    
    # Match fake images to corresponding real train/val
    fake_train = df_fake[df_fake[ori_vid].isin(real_train[ori_vid].unique())].reset_index(drop=True)
    fake_val = df_fake[df_fake[ori_vid].isin(real_val[ori_vid].unique())].reset_index(drop=True)

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
