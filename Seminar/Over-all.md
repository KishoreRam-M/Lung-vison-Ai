# 🖥️ AI-Powered Lung Disease Diagnostic System

## Tech Stack (Detailed Explanation)

### 1. Frontend (User Interface)
**React**  
- Provides an **intuitive, responsive web app** for doctors and healthcare workers.  
- Doctors can upload **X-rays, CT scans, and patient records**.  
- Allows viewing of **AI-generated diagnostic reports** along with **Grad-CAM heatmaps** for image explainability.  
- Ensures seamless interaction and **real-time feedback** from the AI system.

---

### 2. Backend (APIs & Integration Layer)
**Spring Boot (Java)**  
- Handles **secure authentication, request management, and business logic**.  
- Acts as the **primary API gateway**, connecting the frontend, AI models, and database.  
- Provides endpoints for **user management, patient data, and report retrieval**.  

**FastAPI (Python)**  
- Exposes **machine learning (ML) and deep learning (DL) inference endpoints**.  
- Optimized for **fast AI model serving** and integration with **PyTorch/CNNs and Scikit-learn pipelines**.  
- Handles **real-time predictions** for uploaded medical images and patient records.

---

### 3. Database
**PostgreSQL**  
- Stores **structured patient data**: demographics, symptoms, lab results, and medical history.  
- Stores **AI outputs**: diagnostic predictions, probability scores, Grad-CAM heatmaps metadata.  
- Enables **auditing, retraining, and continuous learning** by maintaining historical logs.  

---

### 4. Machine Learning (Structured Data Analysis)
**Scikit-learn**  
- Implements classical ML models: **Logistic Regression, Random Forest, SVM**.  
- Analyzes **patient reports, symptoms, lab tests, and vital statistics**.  
- Produces **interpretable probabilistic predictions**, complementing deep learning image analysis.  
- Useful for **early risk assessment** and multi-modal diagnosis.

---

### 5. Deep Learning (Image Analysis)
**PyTorch** (with **torchxrayvision, ResNet, VGG, custom CNNs**)  
- Specializes in **medical image analysis** for X-rays and CT scans.  
- **Pretrained CNN models** (ResNet, VGG) fine-tuned on lung disease datasets for high accuracy.  
- **Custom CNNs** optimized for local population data (e.g., Indian patient datasets).  
- **Grad-CAM explainability** shows **where AI focuses in the image**, building trust with radiologists.  

---

## 🔄 Workflow (Step-by-Step Detailed Version)

### 1. Data Collection & Preprocessing
- **Structured Data**: Patient records, symptoms, lab results, vitals.  
- **Unstructured Data**: Chest X-rays and CT scans.  
- **Preprocessing Tasks**:  
  - Structured: normalization, handling missing values, feature encoding.  
  - Unstructured: image resizing, contrast enhancement, noise removal.  
- **Output**: Clean, standardized dataset ready for ML and DL pipelines.  

---

### 2. ML Pipeline (Structured Data → Predictive Models)
- **Input**: Patient history, symptoms, lab tests.  
- **Models**: Logistic Regression, Random Forest, SVM (via Scikit-learn).  
- **Output**: Probabilistic predictions for diseases (e.g., TB, Pneumonia).  
- **Advantages**: Interpretable, fast, reliable; complements image-based DL predictions.

---

### 3. DL Pipeline (Medical Images → CNN Architectures)
- **Input**: Chest X-rays, CT scans.  
- **Models**: Pretrained CNNs (ResNet, VGG, DenseNet) + fine-tuned models using **torchxrayvision**.  
- **Output**: Image-based classification scores and **Grad-CAM heatmaps** for visual explainability.  
- **Advantages**: State-of-the-art accuracy in detecting lung diseases from images.

---

### 4. Fusion & Real-Time Diagnosis
- **Fusion Layer**: Combines predictions from ML (structured data) and DL (images).  
- **Weighted Decision Mechanism**: Balances clinical data and image-based findings.  
- **Final Diagnosis Report** includes:  
  - Disease probability scores.  
  - Grad-CAM heatmaps as supporting evidence.  
  - Cross-validation with patient record data.  

---

### 5. Continuous Learning & Database Storage
- **Storage**: All raw data + results stored in PostgreSQL for **auditing and retraining**.  
- **Model Updates**: New data used to **retrain and improve models over time**.  
- **Federated Learning Support**: Allows deployment across multiple hospitals while maintaining **data privacy**.  

---

## ⚡ Example Flow (Storyline)
1. Doctor uploads **patient data + X-ray** through the React app.  
2. **Spring Boot** routes the request to the **FastAPI model server**.  
3. **ML models** (Scikit-learn) analyze patient reports and lab results.  
4. **DL models** (PyTorch CNNs) analyze the X-ray/CT scans.  
5. **Fusion Layer** merges the outputs and generates the **final diagnostic report** with heatmaps.  
6. **PostgreSQL** stores results and raw data for future analysis, retraining, and continuous learning.
