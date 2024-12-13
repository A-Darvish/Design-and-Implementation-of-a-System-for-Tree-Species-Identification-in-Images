# **Tree Species Classification Using Deep Learning and Machine Learning**

An intelligent system for automated tree species classification using leaf images. This project leverages deep learning (ResNet34) for feature extraction and ensemble methods (StackingClassifier, RandomForest, and ...) for classification.

---

## **Table of Contents**
1. [Introduction](#introduction)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Methodology](#methodology)
5. [Results](#results)
6. [Usage](#usage)
7. [System Architecture](#system-architecture)
8. [Technologies Used](#technologies-used)
9. [Contributors](#contributors)

---

## **Introduction**
The goal of this project is to simplify and automate the classification of tree species based on leaf images. This system can aid:
- **Ecologists** in biodiversity monitoring.
- **Farmers** in tree species identification for better care.
- **Tourists** in learning about native trees.

---

## **Features**
- **Feature Extraction:** Pretrained ResNet34 used to extract 512 features from images.
- **Ensemble Classification:** Combines SVM and Random Forest using a stacking approach.
- **Custom Dataset:** Includes manually collected and preprocessed images of 28 tree species.
- **User Interface:** Simple web interface for users to upload leaf images and get predictions.

---

## **Dataset**
- **Sources:** 
  - Manually collected leaf images.
  - Crawled from the web using a custom scraper.
  - Frame extractions from leaf videos.
- **Data Stats:** ~32,000 images representing over 30 tree species.
- **Preprocessing:** Images normalized using mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.

---

## **Methodology**
### **Steps:**
1. **Data Collection and Preprocessing:**
   - Crawling images from the web.
   - Manual video frame extraction.
   - Dataset splitting into training and testing sets.
2. **Feature Extraction:**
   - Using ResNet34 pretrained on ImageNet.
3. **Classification:**
   - Classifier models: SVM, Random Forest, and Naive Bayes.
   - Stacking method for the final classification.
4. **Evaluation:**
   - Metrics used: Accuracy, Precision, Recall, F1-Score.

---

## **Results**
| **Model**          | **Accuracy** | **Precision** | **Recall** | **F1-Score** |
|---------------------|--------------|---------------|------------|--------------|
| SVM                | 93.41%       | 92.5%         | 91.8%      | 92.1%        |
| Random Forest       | 83.61%       | 84.3%         | 83.6%      | 83.9%        |
| StackingClassifier  | **94.98%**   | **93.5%**     | **92.9%**  | **93.2%**    |

Confusion matrices and detailed metrics are available in the repository.

---

## **Usage**
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/tree-species-classification.git
   cd tree-species-classification
   ```

2. **Run the web interface:**
   ```bash
   python app.py
   ```

3. **Access the interface:**
   Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

4. **Upload a leaf image** and get predictions.

---

## **System Architecture**
```
Data Collection -> Preprocessing -> Feature Extraction -> Classification -> Web Interface
```
### Key Modules:
- **Feature Extraction:** ResNet34.
- **Classification Models:** SVM, Random Forest, StackingClassifier.
- **Web Framework:** Flask.
- **UI:** HTML + CSS for a simple user interface.

---

## **Technologies Used**
- **Python Libraries:** PyTorch, scikit-learn, Flask, NumPy, Pandas, Matplotlib.
- **Deep Learning Model:** ResNet34 pretrained on ImageNet.
- **Web Framework:** Flask.

---

## **Contributors**
- **Arvand Darvish**: Developed the project as part of a BSc thesis.
- **Supervisor:** Dr. Ahmad Nickabadi.

---

## **License**
This project is licensed under the MIT License. See `LICENSE` for more details.
