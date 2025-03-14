import numpy as np
import os
import glob
import pandas as pd
from scipy.signal import savgol_filter

def determine_ablation_duration(time, temp, window=11, polyorder=2, delta=0.5):
    
    # Check if the temp data is long enough for the Savitzky-Golay filter
    if len(temp) < window:
        print(f"Input data length ({len(temp)}) is less than the required window size ({window}). Skipping processing.")
        return None, None, None, None, None
    
    # Smooth the signal
    smooth_temp = savgol_filter(temp, window, polyorder)
    
    # Determine the baseline temp (median of the first 10 seconds)
    baseline = np.median(smooth_temp[time < 10])
    
    # Find the start of the ablation: first time point where temp falls below (baseline - delta)
    start_indices = np.where(smooth_temp < (baseline - delta))[0]
    if len(start_indices) == 0:
        raise ValueError("No start point found – check the value of 'delta' or the quality of the data.")
    start_idx = start_indices[0]
    
    # Find the nadir temp (lowest point) starting from the ablation start
    rel_min_idx = np.argmin(smooth_temp[start_idx:])
    min_idx = start_idx + rel_min_idx
    nadir_temp = temp[min_idx]

    # If nadir temp is higher then 0°C -> Measuring error
    if temp[min_idx] > 0:
        print(f"Nadir temperature of input data ({temp[min_idx]}) is too high. Skipping processing.")
        return None, None, None, None, None

    # Calculate ablation duration
    t_start = time[start_idx]
    t_end = time[min_idx]
    duration = t_end - t_start
    
    return duration, nadir_temp, t_start, t_end, smooth_temp

def extract_values(df):

    x_values_list = []
    y_values_list = []
    
    # Iterate over each row in the DataFrame
    for i in range(len(df)):
        # Get the raw string values from the respective columns
        x_raw = df.iloc[i]['ns1:XValues']
        y_raw = df.iloc[i]['ns1:YValues']

        # Split by comma and filter out empty strings (which can occur due to trailing commas)
        x_values_str = [val.strip() for val in x_raw.split(',') if val.strip() != '']
        y_values_str = [val.strip() for val in y_raw.split(',') if val.strip() != '']
        
        # Convert the list of strings to a numpy array of floats
        try:
            x_array = np.array(x_values_str, dtype=float)
            y_array = np.array(y_values_str, dtype=float)
        except ValueError as e:
            raise ValueError(f"Error converting values to float on row {i}: {e}")

        x_values_list.append(x_array)
        y_values_list.append(y_array)
    
    return x_values_list, y_values_list

def load_ablation_data(folder_path, window, polyorder, delta):

    temp_data = []
    pressure_data = []
    flow_data = []

    for file_path in glob.glob(os.path.join(folder_path, '*.xlsx')):
        print(f"Processing file: {file_path}")
        try:
            data = pd.read_excel(file_path, dtype=str)

            temp = []
            pressure = []
            flow = []

            if ((len(data) % 3) == 0):
                for i in range(0, len(data), 3):
                    temp.append(data.iloc[i])
                    pressure.append(data.iloc[i+1])
                    flow.append(data.iloc[i+2])
                
                ### Get the ablation duration and nadir for temperature data
                x_values, y_values = extract_values(pd.DataFrame(temp))

                duration_array = []
                nadir_temp_array = []
                t_end_array = []

                for i in range(0, len(temp)):
                    duration, nadir_temp, _, t_end, _ = determine_ablation_duration(x_values[i], y_values[i], window=window, polyorder=polyorder, delta=delta)
                    duration_array.append(duration)
                    nadir_temp_array.append(nadir_temp)
                    t_end_array.append(t_end)

                temp_df = pd.concat([pd.DataFrame(temp).reset_index(drop=False), pd.DataFrame(duration_array, columns=['AblationTime']), pd.DataFrame(nadir_temp_array, columns=['NadirTempCalc']), 
                                     pd.DataFrame(t_end_array, columns=['t_end'])], axis=1)
                temp_data.append(temp_df.dropna(subset=['AblationTime']).reset_index(drop=False))

                ### Get pressure informations (To do)
                
                pressure_data.append(pd.DataFrame(pressure))

                ### Get flow informations (To do)
                
                flow_data.append(pd.DataFrame(flow))
            else:
                print("Unable to split data into temperature, pressure and cooling capacity values. Returning full data instead!")
                return data, _, _
        
        except ValueError as e:
            raise ValueError(f"Unable to process files!: {e}")
    
    return temp_data, pressure_data, flow_data

def summarise_ablation_data(temp_data):
    summary_list = []
    numeric_columns = ["NadirTemperature", "AblationTime", "t_end"]

    for df in temp_data:
        summary = {} 
        
        if "PatientId" in df.columns:
            summary['PatientId'] = df['PatientId'].iloc[0]
        else:
            print(f"Failed to create column 'PatientId'!")

        summary['AblationCount'] = len(df)
        
        if "Gender" in df.columns:
            if df['Gender'].iloc[0] == "Male":
                summary['Gender'] = 0
            elif df['Gender'].iloc[0] == "Female":
                summary['Gender'] = 1
            else:
                print("Gender could not be determined. Instead, the gender was set to 'male'!")
                summary['Gender'] = 0
        else:
            print(f"Failed to create column 'Gender'!")

        if "Location" in df.columns:
            type_counts = df['Location'].value_counts().to_dict() 
            summary.update(type_counts)
        else:
            print(f"Failed to create column 'Location'!")

        if "ns1:Diagnosis" in df.columns:
            # AF diagnosis mapping
            if df["ns1:Diagnosis"].iloc[0] == "ParoxysmalAF":
                summary['Diagnosis'] = 0
            # Needs to be added later ...

        
        # Numerical columns (calculate mean or median)
        for col in numeric_columns:
            if col in df.columns:
                try:
                    # Convert to float
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    summary[f"{col}_mean"] = df[col].mean()
                    summary[f"{col}_median"] = df[col].median()
                    summary[f"{col}_var"] = df[col].var() 
                except Exception as e:
                    print(f"Error when converting the column {col}: {e}")

        summary_list.append(summary)

    # Replace NaN with 0
    summary_df = pd.DataFrame(summary_list).fillna(0)
    
    return summary_df