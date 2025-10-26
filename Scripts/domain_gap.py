import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the databases
with open('/home/cs21d002_eashaan/PhD/Objective1/data/processed/embedding_dbs/spring-projects_spring-data-mongodb_blob_embeddings32.pkl', 'rb') as f:
    source_blob_db = pickle.load(f)

with open('/home/cs21d002_eashaan/PhD/Objective1/data/processed/embedding_dbs/spring-projects_spring-data-jpa_blob_embeddings32.pkl', 'rb') as f:
    target_blob_db = pickle.load(f)

# Get all embeddings as NumPy arrays
source_embeddings = np.array(list(source_blob_db.values()))
target_embeddings = np.array(list(target_blob_db.values()))

# Create labels: 0 for source, 1 for target
source_labels = np.zeros(source_embeddings.shape[0])
target_labels = np.ones(target_embeddings.shape[0])

# Combine all embeddings and labels
all_embeddings = np.concatenate([source_embeddings, target_embeddings])
all_labels = np.concatenate([source_labels, target_labels])

# Run t-SNE (it's slow, so you might sample it)
print("Running t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
tsne_results = tsne.fit_transform(all_embeddings)

# Plot the results
plt.figure(figsize=(10, 7))
# Plot source (blue)
plt.scatter(tsne_results[all_labels == 0, 0], tsne_results[all_labels == 0, 1], c='blue', label='Source (Anvil)', alpha=0.5)
# Plot target (red)
plt.scatter(tsne_results[all_labels == 1, 0], tsne_results[all_labels == 1, 1], c='red', label='Target (KotlinPoet)', alpha=0.5)
plt.legend()
plt.title('t-SNE Visualization of Project Semantic Spaces')
plt.savefig('spring_mongodb_jpa_domain_gap_visualization.png')
print("Saved plot to domain_gap_visualization.png")

print("Training Domain Classifier...")
# Use the same data: all_embeddings (X) and all_labels (y)
X_train, X_test, y_train, y_test = train_test_split(all_embeddings, all_labels, test_size=0.2, random_state=42)

# Train a simple model
classifier = LogisticRegression(random_state=42)
classifier.fit(X_train, y_train)

# Get accuracy
predictions = classifier.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Domain Classifier Accuracy: {accuracy * 100:.2f}%")