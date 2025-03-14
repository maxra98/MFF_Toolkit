import pandas as pd
import numpy as np
import random

def generate_synthetic_data(df, num_patients=50, random_state=None):

    # Set random number generator to fixed seed
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    synthetic_data = []
    
    for i in range(num_patients):
        new_row = df.copy()
        
        new_row["PatientId"] = i
        
        # AblationCount vary slightly (±2 ablations, min. 1)
        new_row["AblationCount"] = max(1, new_row["AblationCount"].values[0] + np.random.randint(-2, 3))

        # Set gender randomly (0 or 1)
        new_row["Gender"] = np.random.choice([0, 1])

        # RSPV, LIPV, LSPV, RIPV vary with slight noise
        for col in ["RSPV", "LIPV", "LSPV", "RIPV"]:
            new_row[col] = max(0, new_row[col].values[0] + np.random.randint(-1, 2))  # Min 0, max +1

        new_row["Diagnosis"] = np.random.choice(df["Diagnosis"].unique())

        # Modify numerical data with normally distributed noise
        for col in ["NadirTemperature_mean", "NadirTemperature_median", "NadirTemperature_var", 
                    "AblationTime_mean", "AblationTime_median", "AblationTime_var", "t_end_mean", "t_end_median", "t_end_var"]:
            mean = df[col].values[0]
            std_dev = 0.05 * abs(mean)  # 5% des Mittelwertes als Streuung
            new_row[col] = mean + np.random.normal(0, std_dev)

        synthetic_data.append(new_row)

    # Finale DataFrame
    synthetic_df = pd.concat(synthetic_data, axis=0)
    synthetic_df.reset_index(drop=True, inplace=True)
    
    return synthetic_df

def generate_synthetic_ecg_from_ptb_xl(df_ecgs, num_ecgs=1):
    
    if num_ecgs <= len(df_ecgs):
        df_random_ecgs = df_ecgs.sample(n=num_ecgs, random_state=42)
        return df_random_ecgs
    else:
        print(f"Number of ECGs in passed DataFrame (length: {len(df_ecgs)}) too small for desired number (num: {num_ecgs})!")

def generate_synthetic_egm(df, total_time=10, sampling_rate=1000, no_effect_ratio=0.5, min_effect=0.1, max_effect=0.9):
    
    signals = []

    # Time vector from 0 to total_time
    t = np.linspace(0, total_time, int(total_time * sampling_rate))
    
    frequency = 1.0

    for idx, row in df.iterrows():
        ablation_time = row['t_end_mean']
        
        # Define randomly which group
        if np.random.rand() < no_effect_ratio:
            # Group A: No effect
            fade = np.ones_like(t)
        else:
            # Group B: Apply damping factor after t_end
            damping_factor = np.random.uniform(min_effect, max_effect)
            fade = np.ones_like(t)
            fade[t >= ablation_time] = damping_factor
        
        # Generate synthetic egm signals as sinus waves
        ekg_signal = np.sin(2 * np.pi * frequency * t) * fade
        
        signals.append((t, ekg_signal))
    
    return signals

def create_labels(egm_data, df, s=0.8, delta=0.05, one_hot_encode=False):

    labels = []

    for i, (t, egm_signal) in enumerate(egm_data):
        t_end = df.iloc[i]["t_end_mean"]
        
        # EKG before and after t_end
        pre_ablation = egm_signal[t < t_end]
        post_ablation = egm_signal[t >= t_end]
        
        # Calculate mean amplitude values
        A_pre = np.mean(np.abs(pre_ablation))
        A_post = np.mean(np.abs(post_ablation))
        
        # Classification
        if s * A_pre - A_post > delta:
            labels.append(1)  # Ablation successful
        else:
            labels.append(0)  # Ablation duration was too short

    if one_hot_encode:
        ohe_labels = [[0,0] for i in list(labels)]
        for index, e in enumerate(labels):
            if e == 0:
                ohe_labels[index][0] = 1
            elif e == 1:
                ohe_labels[index][1] = 1

        labels = np.array(ohe_labels)

    return np.array(labels)