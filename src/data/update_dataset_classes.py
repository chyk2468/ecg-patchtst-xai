import pandas as pd
import ast
import os

# Paths
base_path = 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/'
scp_path = os.path.join(base_path, 'scp_statements.csv')
db_path = os.path.join(base_path, 'ptbxl_database.csv')

# 1. Update scp_statements.csv
scp_df = pd.read_csv(scp_path, index_col=0)

def determine_superclass(row):
    if pd.notnull(row['diagnostic_class']) and row['diagnostic_class'] != '':
        return row['diagnostic_class']
    elif row['rhythm'] == 1.0:
        return 'RHYTHM'
    elif row['form'] == 1.0:
        return 'FORM'
    return None

scp_df['superclass'] = scp_df.apply(determine_superclass, axis=1)
scp_df.to_csv(scp_path)
print(f"Updated {scp_path} with 'superclass' column.")

# 2. Update ptbxl_database.csv
db_df = pd.read_csv(db_path, index_col='ecg_id')

# Create a mapping dictionary for easy lookup
mapping = scp_df['superclass'].to_dict()

def aggregate_7_superclass(scp_codes_str):
    scp_codes = ast.literal_eval(scp_codes_str)
    superclasses = []
    for code in scp_codes.keys():
        if code in mapping:
            sclass = mapping[code]
            if sclass:
                superclasses.append(sclass)
    return list(set(superclasses))

db_df['superclass'] = db_df['scp_codes'].apply(aggregate_7_superclass)
db_df.to_csv(db_path)
print(f"Updated {db_path} with 'superclass' column.")
