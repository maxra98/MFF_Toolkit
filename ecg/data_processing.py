import pandas as pd
import numpy as np
import wfdb
import ast
from tqdm import tqdm

def load_raw_ecg_data(df, sampling_rate, path):
    data = []
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        i=0
        for f in tqdm(df.filename_hr):
            data.append(wfdb.rdsamp(path+f))
            i+=1

    data = np.array([signal for signal, meta in data])
    return data

def load_and_process_ptb_xl_data(ptb_xl_path):
    Y = pd.read_csv(ptb_xl_path + 'ptbxl_database.csv', index_col='ecg_id')

    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(ptb_xl_path+'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))
    
    # Apply diagnostic superclass
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

    # Sort the list of diagnostic superclasses alphabetically (or you can customize sorting)
    Y['diagnostic_superclass'] = Y['diagnostic_superclass'].apply(lambda x: sorted(x) if x is not None else None)

    # Reduce to the first class if not empty, otherwise set to None
    Y['diagnostic_superclass'] = Y['diagnostic_superclass'].apply(lambda x: [x[0]] if x is not None and len(x) > 0 else None)

    # Remove entries where diagnostic_superclass is None
    Y = Y.dropna(subset=['diagnostic_superclass'])

    # Remove entries that have more than one label value
    condition = lambda x: len(x) != 1
    filtered_entries = Y[Y['diagnostic_superclass'].apply(condition)]

    Y.drop(filtered_entries.index, inplace=True)

    return Y

def get_NORM_AF_data(data):

    # Filter out ECGs with AFIB as scp_code
    Y_with_AFIB = data[data['scp_codes'].apply(lambda x: 'AFIB' in x)]

    # Filter all AF data points in which 'diagnostic_superclass' contains the value NORM
    norm_af_reports = Y_with_AFIB[Y_with_AFIB['diagnostic_superclass'].apply(lambda x: x == ['NORM'])]

    # List of words that are allowed
    af_words = ['atrial fibrillation', 'fÖrmaksflimmer', 'vorhofflimmern']

    # Filter the DataFrame to output the data points that do not contain any of the allowed words in `af_words`
    filtered_reports = Y_with_AFIB[Y_with_AFIB['report'].apply(lambda report: all(word not in report.lower() for word in af_words))]

    # List of words that are allowed
    af_words = ['atrial fibrillation', 'fÖrmaksflimmer', 'vorhofflimmern']

    # Filter the DataFrame to output the data points that do not contain any of the allowed words in `af_words`
    filtered_reports = Y_with_AFIB[Y_with_AFIB['report'].apply(lambda report: all(word not in report.lower() for word in af_words))]

    # Drop the filtered data points
    Y_AFIB_filtered = Y_with_AFIB.drop(filtered_reports.index)

    # Get NORM data
    # NORM data
    Y_norm = data[data['diagnostic_superclass'].apply(lambda x: x == ['NORM'])]
    Y_norm_filtered = Y_norm.drop(norm_af_reports.index)

    # Add 'Label' column for binary classification in NORM = 0 and AF = 1
    Y_norm_filtered['Label'] = 0
    Y_AFIB_filtered['Label'] = 1

    return Y_norm_filtered, Y_AFIB_filtered

def oversample_data(X_train, y_train):
    unique, counts = np.unique(y_train, return_counts=True)
    class_distribution = dict(zip(unique, counts))

    minority_class = min(class_distribution, key=class_distribution.get)
    majority_count = max(class_distribution.values())

    # Extract indices of the minority class
    minority_indices = np.where(y_train == minority_class)[0]

    # Multiplying the minority class
    oversampled_indices = np.tile(minority_indices, (majority_count // len(minority_indices) + 1))[:majority_count]
    oversampled_minority_X = X_train[oversampled_indices]
    oversampled_minority_y = y_train[oversampled_indices]

    # Create new data set
    X_train_balanced = np.concatenate([X_train[y_train != minority_class], oversampled_minority_X])
    y_train_balanced = np.concatenate([y_train[y_train != minority_class], oversampled_minority_y])

    # Mix data
    indices = np.random.permutation(len(X_train_balanced))
    X_train_balanced = X_train_balanced[indices]
    y_train_balanced = y_train_balanced[indices]

    return X_train_balanced, y_train_balanced






