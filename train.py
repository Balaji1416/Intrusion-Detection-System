from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import pandas as pd
import numpy as np

def read_sample(filename, nrows=100000):
    return pd.read_csv(filename, nrows=nrows)

df1 = read_sample("intrusion_data.csv")
df2 = read_sample("intrusion_data2.csv")
df3 = read_sample("intrusion_data3.csv")

data = pd.concat([df1, df2, df3], ignore_index=True)




# Define important features
selected_features = [
    'Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
    'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min',
    'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min',
    'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max'
]

# Select features and target
X = data[selected_features]
y = data['Label']

# Convert to numeric and handle missing/infinite values
X = X.apply(pd.to_numeric, errors='coerce')
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# 🚀 SAMPLE DATA (for fast training): Use a small subset
X_sample, _, y_sample, _ = train_test_split(X, y_encoded, train_size=0.1, stratify=y_encoded, random_state=42)

# 🚀 LIGHTWEIGHT RANDOM FOREST MODEL
fast_model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42, n_jobs=-1)
fast_model.fit(X_sample, y_sample)

# Save model
pickle.dump(fast_model, open("intrusion_model.pkl", "wb"))

print("✅ Super fast model trained and saved!")
