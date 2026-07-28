# 🍅 SariWise Tomato Ripeness Classifier 

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge\&logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine_Learning-orange?style=for-the-badge\&logo=scikitlearn)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green?style=for-the-badge\&logo=opencv)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge\&logo=jupyter)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

### Machine Learning-Based Tomato Ripeness Classification using Support Vector Machines (SVM)

A computer vision and machine learning project that classifies tomato ripeness levels from images using handcrafted image features and a Support Vector Machine (SVM) model.

🔗 **Live Demo:** https://bit.ly/4k51jvd

</div>

---

## 📖 Overview

TomatoRipeness is a machine learning project developed to automate the classification of tomato ripeness through image processing techniques.

The system extracts visual features from tomato images and uses a Support Vector Machine (SVM) classifier to determine the ripeness category, enabling consistent and objective quality assessment.

This project demonstrates the application of:

* Computer Vision
* Feature Engineering
* Machine Learning Classification
* Agricultural Technology

---

## 🎯 Objectives

* Classify tomatoes based on ripeness levels
* Explore image preprocessing techniques
* Extract meaningful visual features from images
* Train and evaluate an SVM classification model
* Demonstrate the use of machine learning in agriculture

---

## 🏗️ Project Structure

```text
TomatoRipeness/
│
├── dataset/                # Tomato image dataset
├── notebooks/              # Jupyter notebooks
├── models/                 # Trained ML models
├── outputs/                # Generated results and visualizations
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/aro-wen/TomatoRipeness.git
cd TomatoRipeness
```

---

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

or

```bash
jupyter lab
```

Open the notebook and run the cells sequentially to train, test, and evaluate the classifier.

---

## ⚙️ Prerequisites

Before running the project, ensure you have:

* Python 3.10+
* Git
* Visual Studio Code (Recommended)
* Jupyter Notebook or JupyterLab
* Internet connection for package installation

---

## 🛠️ Recommended VS Code Extensions

Install the following extensions:

| Extension | Publisher |
| --------- | --------- |
| Python    | Microsoft |
| Jupyter   | Microsoft |
| Pylance   | Microsoft |

Access Extensions Marketplace:

```text
Ctrl + Shift + X
```

---

## 🤖 Machine Learning Pipeline

### Data Preparation

* Image collection and labeling
* Dataset organization
* Data preprocessing

### Feature Extraction

* Color-based features
* Texture analysis
* Image normalization

### Model Training

* Support Vector Machine (SVM)
* Hyperparameter tuning
* Cross-validation

### Evaluation

* Accuracy measurement
* Confusion matrix analysis
* Performance comparison

---

## 📦 Main Libraries Used

```text
numpy
opencv-python
matplotlib
pandas
scikit-learn
jupyter
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 📊 Results

The classifier predicts the ripeness category of tomato images based on extracted visual characteristics.

Example classes:

* 🟢 Unripe
* 🟡 Partially Ripe
* 🔴 Ripe

Add your model performance metrics here:

```text
Accuracy: 98.76%
Precision: 98.7%
Recall: 98.4%
```


## 🧹 .gitignore

Recommended entries:

```gitignore
venv/
__pycache__/
.ipynb_checkpoints/
*.pyc
.DS_Store
```

---

## 🔬 Future Improvements

* Expand dataset size
* Experiment with additional feature extraction methods
* Compare SVM with Random Forest and XGBoost
* Implement deep learning models such as CNNs
* Deploy as a web or mobile application

---

## 👩‍💻 Author

**Leila Arowen A. Dumindin**

* 🎓 BS Computer Engineering, Pamantasan ng Lungsod ng Maynila
* 🌟 DOST-SEI Merit Scholar
* 🤖 Machine Learning and Computer Vision Enthusiast

GitHub: https://github.com/aro-wen

---

<div align="center">

### ⭐ If you found this project useful, consider giving it a star!

Built with Python, OpenCV, Scikit-Learn, and a passion for AI-driven agriculture.

</div>
