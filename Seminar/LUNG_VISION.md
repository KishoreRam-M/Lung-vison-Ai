## **1. Setup: Installing Libraries**

```python
# !pip install torchxrayvision torch torchvision pillow scikit-learn matplotlib gradio tqdm
```

**What it does:**

* Installs the Python libraries needed to run the project:

  * `torch` & `torchvision`: For creating and training neural networks.
  * `torchxrayvision`: Special tools for chest X-ray image models.
  * `pillow`: Handling images.
  * `scikit-learn`: Calculating accuracy, splitting datasets.
  * `matplotlib`: Plotting graphs.
  * `gradio`: Create a web interface to test your model easily.
  * `tqdm`: Shows progress bars while the model trains/tests.

---

## **2. Imports**

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import torchxrayvision as xrv
import gradio as gr
from tqdm import tqdm
```

**Explanation:**

* These lines make the libraries available in your Python code so you can use them.
* For example:

  * `torch` = the brain of ML/DL.
  * `PIL.Image` = lets you open, resize, and manipulate images.
  * `matplotlib.pyplot` = lets you draw graphs to see how well your model is learning.

---

## **3. Device Setup**

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
```

**What it does:**

* Checks if your computer has a GPU (`cuda`) for faster training. If not, it uses the CPU.
* Training on GPU is much faster for images.

---

## **4. Dataset Class**

```python
class LungDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_to_idx = {cls: i for i, cls in enumerate(sorted(set(labels)))}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.class_to_idx[self.labels[idx]]
        img = Image.open(img_path).convert("L")  # Grayscale
        if self.transform:
            img = self.transform(img)
        return img, label
```

**Explanation (step by step):**

* **Purpose:** Tells the model how to read images and labels from your folder.
* `__init__`:

  * `image_paths` = locations of all X-ray images.
  * `labels` = what disease each image has (`Covid`, `Normal`, `Pneumonia`).
  * `transform` = preprocessing steps like resizing, rotating, or flipping.
* `__len__`: returns total number of images.
* `__getitem__`:

  * Reads an image from disk.
  * Converts it to grayscale (`L`).
  * Applies transformations.
  * Returns the image and its label as numbers.

---

## **5. Prepare Dataset**

```python
DATA_DIR = r"C:\Users\KRM\PycharmProjects\Lung-Vison\data\Covid19-dataset\train"
classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
num_classes = len(classes)
print("Classes:", classes)
```

* Reads your folder to get the **classes** (`Covid`, `Normal`, `Viral Pneumonia`) and counts them.

**Gather image paths and labels:**

```python
image_paths, labels = [], []
for cls in classes:
    cls_path = os.path.join(DATA_DIR, cls)
    imgs = [os.path.join(cls_path, f) for f in os.listdir(cls_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_paths.extend(imgs)
    labels.extend([cls] * len(imgs))
```

* Loops through each class folder.
* Collects all image file paths.
* Creates a label for each image.

---

## **6. Split Dataset**

```python
train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
    image_paths, labels, test_size=0.3, stratify=labels, random_state=42
)
val_imgs, test_imgs, val_labels, test_labels = train_test_split(
    temp_imgs, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)
```

**Explanation:**

* Splits data into **train (70%)**, **validation (15%)**, **test (15%)**.
* `stratify=labels` ensures each set has a balanced number of each class.

---

## **7. Image Transformations**

```python
IMG_SIZE = 224
train_transform = transforms.Compose([...])
valid_transform = transforms.Compose([...])
```

**What it does:**

* Resizes images to `224x224` pixels.
* For training:

  * Random flips, rotations → make model robust.
* For validation/testing:

  * Only resize and normalize.
* `Normalize` = scales pixel values so model learns better.

---

## **8. Create Datasets and DataLoaders**

```python
train_ds = LungDataset(train_imgs, train_labels, transform=train_transform)
val_ds = LungDataset(val_imgs, val_labels, transform=valid_transform)
test_ds = LungDataset(test_imgs, test_labels, transform=valid_transform)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
```

* `DataLoader` = feeds data to the model in **batches** instead of all at once.
* `shuffle=True` = randomly shuffles images during training to prevent bias.

---

## **9. Load Pretrained Model**

```python
model = xrv.models.DenseNet(weights="densenet121-res224-all")
model.apply_sigmoid = False
model.op_threshs = None
in_features = model.classifier.in_features
model.classifier = nn.Linear(in_features, num_classes)
model = model.to(device)
```

* Loads a **DenseNet** model trained on chest X-rays.
* Replaces the last layer to match the number of your classes (`Covid`, `Normal`, `Pneumonia`).

---

## **10. Loss Function & Optimizer**

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
```

* `criterion` = tells the model **how wrong it is**.
* `optimizer` = updates the model to make it better.

---

## **11. Training Loop**

```python
for epoch in range(num_epochs):
    model.train()
    ...
    model.eval()
    ...
    print(...)
```

**Step by step:**

1. **Training phase:**

   * Model looks at each batch.
   * Predicts labels.
   * Calculates error (`loss`).
   * Adjusts itself to reduce error.
2. **Validation phase:**

   * Checks model performance on unseen images.
   * Calculates accuracy and loss.
3. **Tracking:**

   * Saves losses and accuracies for plotting later.

---

## **12. Plot Training Graphs**

```python
plt.figure(figsize=(12, 4))
...
plt.show()
```

* Shows how model **improves over time**:

  * Loss decreases = model learns.
  * Accuracy increases = model predicts better.

---

## **13. Save Model**

```python
torch.save(model.state_dict(), "lung_vision_model.pth")
```

* Saves trained model so you don’t need to train it again.

---

## **14. Testing & Metrics**

```python
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
```

* Evaluates the model on **test images** it has never seen.
* Metrics:

  * Accuracy = % correct predictions.
  * Precision, Recall, F1 = more detailed performance metrics.

---

## **15. Prediction & Visualization Functions**

```python
def plot_prediction(...): ...
def predict_image_gradio(...): ...
```

* `plot_prediction` = shows the X-ray and predicted probabilities.
* `predict_image_gradio` = prepares a new X-ray image for prediction and returns probabilities for each class.

---

## **16. Gradio Web Interface**

```python
interface = gr.Interface(...)
if __name__ == "__main__":
    interface.launch(share=True, inbrowser=True)
```

* Makes it **super easy** to test the model:

  * Upload an X-ray image.
  * Model predicts whether it’s `Covid`, `Pneumonia`, or `Normal`.
* `share=True` → you can share your app link with anyone.

---

## **🔹 Workflow Summary (Step by Step)**

1. **Collect Data:** Chest X-ray images sorted by disease.
2. **Preprocess Data:** Resize, normalize, augment images for training.
3. **Split Data:** Train, validation, test sets.
4. **Build Dataset:** Tell the model how to read images and labels.
5. **Load Model:** Use a pretrained DenseNet and adapt it to your classes.
6. **Train Model:** Model learns patterns from X-ray images.
7. **Validate Model:** Check performance on unseen data.
8. **Plot Training Graphs:** See how the model improves.
9. **Test Model:** Measure accuracy, precision, recall, F1-score.
10. **Save Model:** So you can reuse it.
11. **Predict New Images:** Use Gradio interface for easy predictions.
