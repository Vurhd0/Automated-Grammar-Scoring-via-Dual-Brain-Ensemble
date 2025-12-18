# Dual-Brain Ensemble for Automated Grammar Scoring

## Executive Summary
This repository contains a multi-modal AI pipeline designed to evaluate language proficiency from audio recordings. The system fuses Acoustic Physics (delivery style) with a Dual-Brain Embedding System (content analysis). By combining these approaches, the model effectively captures fluency, confidence, and deep semantic meaning. The final score is generated through a weighted ensemble of XGBoost and Ridge Regression, ensuring robust performance even with limited training data.

---

## Methodology and Architecture

### 1. Feature Engineering Strategy
The pipeline extracts a dense 40-dimensional feature vector for each audio file using a three-stage process:

#### A. Acoustic Physics (Librosa)
Fundamental signal metrics are extracted to evaluate the quality of delivery:
* **Silence Ratio:** Quantifies fluency and hesitation.
* **Pitch Stability (f0):** Serves as a proxy for speaker confidence and intonation.
* **Speaking Rate:** Measures words per minute (WPM).

#### B. Linguistic Analysis (Whisper + T5)
* **Transcription:** Audio is transcribed using the OpenAI Whisper (Base) model.
* **Grammar Distance:** A T5 Grammar Correction model generates a corrected version of the transcript. The Levenshtein distance between the original speech and the correction provides a quantitative Error Density score.
* **Rule-Based Checks:** LanguageTool is used to scan for specific rule violations.

#### C. The Dual-Brain Embedding System
To capture semantic meaning without overfitting, two distinct Transformer models are employed:
1.  **MiniLM-L6 (Brain 1):** Focuses on efficiency, capturing broad keywords and surface-level context.
2.  **MPNet-Base (Brain 2):** Focuses on precision, capturing subtle semantic nuances that smaller models might miss.

---

### 2. Architecture: Compression and Ensemble
Raw embeddings generate over 1,100 dimensions, which can lead to overfitting on small datasets. This solution addresses that issue through the following steps:

1.  **Compression (PCA):** Principal Component Analysis is applied to compress each language model output down to 16 high-variance components. This retains the core linguistic signal while filtering out noise.
    * *Total Features: 8 Scalars + 16 (MiniLM) + 16 (MPNet) = 40 Features.*

2.  **Hybrid Ensemble:**
    * **XGBoost (60% Weight):** Identifies non-linear patterns and complex interactions between acoustics and grammar.
    * **Ridge Regression (40% Weight):** Provides linear stability to prevent the ensemble from overfitting to outliers.

---

## Setup and Dependencies

### Prerequisites
The project requires Python 3.8+ and a Java 17 environment to support the grammar checking backend.

### Installation
Run the following commands to configure the environment:

```bash
# 1. Install System Dependencies (Java for LanguageTool)
apt-get update -qq
apt-get install -y openjdk-17-jdk-headless -qq
export JAVA_HOME="/usr/lib/jvm/java-17-openjdk-amd64"

# 2. Install Python Libraries
pip install -q openai-whisper language-tool-python textstat librosa xgboost sentence-transformers transformers torch Levenshtein scikit-learn
