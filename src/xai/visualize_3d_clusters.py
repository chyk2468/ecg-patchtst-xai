import pandas as pd
import numpy as np
import ast
from sklearn.decomposition import PCA
import plotly.express as px
import os

# Paths
base_path = 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/'
db_path = os.path.join(base_path, 'ptbxl_database.csv')
scp_path = os.path.join(base_path, 'scp_statements.csv')

print("Loading data...")
df = pd.read_csv(db_path, index_col='ecg_id')
df.scp_codes = df.scp_codes.apply(ast.literal_eval)
df.superclass = df.superclass.apply(ast.literal_eval)

scp_info = pd.read_csv(scp_path, index_col=0)
all_scp_codes = scp_info.index.tolist()

print("Creating feature matrix from SCP likelihoods...")
# Create a feature matrix where each column is an SCP code
features = np.zeros((len(df), len(all_scp_codes)))
for i, (idx, row) in enumerate(df.iterrows()):
    for code, likelihood in row['scp_codes'].items():
        if code in all_scp_codes:
            features[i, all_scp_codes.index(code)] = likelihood

print("Performing PCA to 3D...")
pca = PCA(n_components=3)
components = pca.fit_transform(features)

# Determine dominant superclass for coloring
# Priority: MI > HYP > CD > STTC > FORM > RHYTHM > NORM
priority = ['MI', 'HYP', 'CD', 'STTC', 'FORM', 'RHYTHM', 'NORM']

def get_dominant_class(classes):
    for p in priority:
        if p in classes:
            return p
    return 'UNKNOWN'

df['dominant_superclass'] = df['superclass'].apply(get_dominant_class)

# Prepare dataframe for plotting
plot_df = pd.DataFrame(data=components, columns=['PC1', 'PC2', 'PC3'])
plot_df['Class'] = df['dominant_superclass'].values
plot_df['Patient_ID'] = df['patient_id'].values
plot_df['Age'] = df['age'].values

print("Generating 3D interactive plot...")
fig = px.scatter_3d(
    plot_df, x='PC1', y='PC2', z='PC3',
    color='Class',
    title='3D Superclass Clustering (PCA of SCP Likelihoods)',
    labels={'PC1': 'Dim 1', 'PC2': 'Dim 2', 'PC3': 'Dim 3'},
    opacity=0.7,
    hover_data=['Patient_ID', 'Age']
)

fig.update_traces(marker=dict(size=3))
output_file = 'superclass_3d_clustering.html'
fig.write_html(output_file)
print(f"Success! Visualization saved to {output_file}")
