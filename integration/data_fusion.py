import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def normalise_and_OHE_data(ablation_data, continuous_vars, categorical_vars):
    # Normalise
    scaler = StandardScaler()
    ablation_data[continuous_vars] = scaler.fit_transform(ablation_data[continuous_vars])

    encoder = OneHotEncoder(drop='first', sparse_output=False)  # 'drop=first' removes redundancy for binary variables

    # One-Hot-Encoding
    encoded_categorical = encoder.fit_transform(ablation_data[categorical_vars])

    # Create new column names for one-hot-encoded vars
    encoded_columns = encoder.get_feature_names_out(categorical_vars)

    # To DataFrame
    df_encoded_categorical = pd.DataFrame(encoded_categorical, columns=encoded_columns)

    # Remove original columns and add new ones
    ablation_data.drop(columns=categorical_vars, inplace=True)
    result = pd.concat([ablation_data, df_encoded_categorical], axis=1)

    return result

def combine_data(ecg_df=None, egm_df=None, patient_df=None, ablation_df=None, axis=0):
    dfs = [df for df in [ecg_df, egm_df, patient_df, ablation_df] if df is not None]
    
    if not dfs:
        return pd.DataFrame()
    
    combined_df = pd.concat(dfs, axis=axis, ignore_index=(axis==0))
    
    return combined_df