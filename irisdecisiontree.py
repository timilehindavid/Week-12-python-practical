import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix

# ---------- 1. LOAD THE DATA ----------
script_folder = os.path.dirname(os.path.abspath(__file__))   # folder this script lives in
csv_path = os.path.join(script_folder, "Iris.csv")           # look for Iris.csv in that same folder
df = pd.read_csv(csv_path).drop(columns="Id")                # "Id" is just a row number, so drop it

features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
X = df[features]       # the measurements the model learns from
y = df["Species"]      # the answer the model tries to predict

# ---------- 2. SPLIT INTO TRAIN AND TEST ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # keep 20% of the data hidden for testing
    random_state=42,     # makes the split the same every time you run it
    stratify=y           # keep the same mix of species in both parts
)
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# ---------- 3. TRAIN THE DECISION TREE ----------
model = DecisionTreeClassifier(random_state=42)   # create the model
model.fit(X_train, y_train)                       # learn from the training data

# ---------- 4. PREDICT ON THE TEST DATA ----------
y_pred = model.predict(X_test)                    # predict species for rows the model has never seen

# ---------- 5. EVALUATE ----------
accuracy = accuracy_score(y_test, y_pred)                        # share of predictions that were right
precision = precision_score(y_test, y_pred, average="macro")     # when it predicts a species, how often is it right?
recall = recall_score(y_test, y_pred, average="macro")           # of all real members of a species, how many did it find?

print(f"\nAccuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}  (macro average across the 3 species)")
print(f"Recall:    {recall:.3f}  (macro average across the 3 species)")

print("\nPer-species report:\n", classification_report(y_test, y_pred))
print("Confusion matrix (rows = actual, columns = predicted):\n", confusion_matrix(y_test, y_pred))

# ---------- 6. VISUALISE THE TREE ----------
plt.figure(figsize=(14, 8))
plot_tree(model, feature_names=features, class_names=model.classes_, filled=True)
plt.title("Decision tree for the Iris dataset")
plt.show()
