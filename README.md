# Disease Prediction from Gene Expression (RNA-seq)

This project predicts cancer types from bulk RNA-seq gene expression data, inspired by the [BulkRNABert study].  
Unlike the full multi-omics approach, this version focuses only on **RNA-seq gene expression** for a smaller set of cohorts, making the project more lightweight and accessible.

---

## 📌 Project Goals
- Preprocess RNA-seq gene expression data from [TCGA](https://portal.gdc.cancer.gov/).  
- Build machine learning and deep learning models to classify cancer types.  
- Compare baseline models with simple neural networks.  
- Provide reproducible code and results for educational use.

---

## 📊 Dataset
- **Source:** The Cancer Genome Atlas (TCGA).  
- **Cohorts used:**  
  - BRCA (Breast Invasive Carcinoma)  
  - BLCA (Bladder Carcinoma)  
  - GBMLGG (Glioblastoma + Lower Grade Glioma)  
  - LUAD (Lung Adenocarcinoma)  
  - UCEC (Uterine Corpus Endometrial Carcinoma)  
- **Features:** ~19,000 gene expression values (TPM, log-normalized).  
- **Labels:** Cancer type.  
- **Format:** CSV with rows = samples, columns = genes, plus a `label` column.

👉 If you want to try a binary task, combine **TCGA cancer samples** vs **GTEx normal tissues**.

---

## ⚙️ Project Structure
```
├── data/
│   └── processed/         # Preprocessed CSV matrices (samples x genes)
├── notebooks/             # Jupyter notebooks for analysis & experiments
├── src/
│   ├── preprocess.py      # Scripts to preprocess raw TCGA data
│   ├── models.py          # ML & DL models for classification
│   └── train.py           # Training & evaluation script
├── .gitignore
├── LICENSE.md
├── README.md
```

---

## 🧪 Methods
Models implemented:
1. **Random Forest**  
2. **Support Vector Machine (SVM)**  
3. **Feed-forward Neural Network (MLP)**  
4. *(Optional)* Autoencoder for feature reduction + classifier

Evaluation:
- Weighted F1, Macro F1, Accuracy, ROC-AUC
- 80/20 stratified split or 5-fold cross-validation

---

## 🚀 Usage
1. Clone the repo:
   ```bash
   git clone https://github.com/ahmedunais/disease-prediction-rnaseq.git
   cd disease-prediction-rnaseq
   ```
2. Preprocess data:
   ```bash
   python src/preprocess.py
   ```
3. Train models:
   ```bash
   python src/train.py --model mlp
   python src/train.py --model svm
   ```

---

## 📈 Results
- **Baseline models** (SVM, RF) provide strong performance on selected cohorts.  
- **MLP** improves classification by capturing nonlinear relationships.  
- Future work: Incorporate GTEx normals, try transformer embeddings, expand to pan-cancer classification.


