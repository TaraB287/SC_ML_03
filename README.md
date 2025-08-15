# Dog vs. Cat Image Classification using HOG and SVM

This project demonstrates a classic computer vision task: classifying images of dogs and cats. Instead of using deep learning, this implementation uses a traditional machine learning approach, combining the Histogram of Oriented Gradients (HOG) feature descriptor with a linear Support Vector Machine (SVM) classifier.

This project is an excellent introduction to image classification pipelines, feature extraction, and model training using scikit-learn.

## Table of Contents
* [Project Overview](#-project-overview)
* [How It Works](#-how-it-works)
* [Project Structure](#-project-structure)
* [Getting Started](#-getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation & Setup](#installation--setup)
* [Usage](#-usage)
* [Model Evaluation](#-model-evaluation)
* [Submitting to Kaggle](#-submitting-to-kaggle)
* [Dependencies](#-dependencies)

## 📝 Project Overview

The goal of this project is to build a binary classifier that can distinguish between images of dogs and cats. The project uses the "Dogs vs. Cats" dataset from a Kaggle competition. The core of the project involves:

1.  **Data Preparation**: Downloading and unzipping the image dataset from Kaggle.
2.  **Feature Extraction**: Converting images into a numerical format that a machine learning model can understand. This is done by extracting Histogram of Oriented Gradients (HOG) features.
3.  **Model Training**: Training a Linear Support Vector Machine (SVM) on the extracted HOG features to learn the difference between dog and cat images.
4.  **Evaluation**: Assessing the model's performance on a held-out test set using metrics like accuracy.
5.  **Prediction**: Generating predictions on the official test dataset and formatting them for a Kaggle submission.

## How It Works

The classification pipeline follows these key steps:

1.  **Image Preprocessing**: Each image is read, converted to grayscale, and resized to a consistent dimension (64x128 pixels). This standardization is crucial for consistent feature extraction.

2.  **HOG Feature Extraction**: The HOG descriptor is applied to each grayscale image. HOG captures the shape and texture of objects by calculating the distribution of intensity gradients or edge directions. This results in a feature vector for each image.

3.  **SVM Classification**: A `StandardScaler` is used to normalize the feature vectors. Then, a `LinearSVC` (Support Vector Classifier) is trained on these scaled features. The SVM finds an optimal hyperplane that best separates the feature vectors of the two classes (dogs and cats).

4.  **Prediction**: For a new image, the same preprocessing and HOG extraction steps are applied. The trained SVM then predicts whether the resulting feature vector corresponds to a dog or a cat.

## Project Structure

After running the initial setup cells in the notebook, the directory will be structured as follows:

```
/content/
|-- dogs-vs-cats.zip
|-- sampleSubmission.csv
|-- test1.zip
|-- train.zip
|-- test1/
|   |-- 1.jpg
|   |-- 2.jpg
|   |-- ...
|-- train/
|   |-- cat.0.jpg
|   |-- dog.0.jpg
|   |-- ...
|-- submission.csv
|-- kaggle.json
```


## Getting Started

To get this project up and running, follow the steps below. This project is designed to be run in a Google Colab environment.

### Prerequisites

* A Kaggle account is required to download the dataset. You will need to generate an API token (`kaggle.json`).

### Installation & Setup

1.  **Clone the Repository**:
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Upload to Google Colab**: Open [Google Colab](https://colab.research.google.com/) and upload the `Dogs_vs_Cats.ipynb` notebook.

3.  **Get Kaggle API Key**:
    * Go to your Kaggle account page.
    * In the "API" section, click on "Create New API Token". This will download `kaggle.json`.

4.  **Run the Notebook**: Execute the cells in the notebook sequentially.
    * The first cell will prompt you to upload your `kaggle.json` file.
    * The notebook will then automatically download and unzip the dataset.

## Usage

Simply run all the cells in the `Dogs_vs_Cats.ipynb` notebook from top to bottom. The notebook is divided into logical sections:

1.  **Setup and Data Download**: Configures the Kaggle API and downloads the dataset.
2.  **Feature Extraction**: Processes a subset of the training images to create HOG features and corresponding labels.
3.  **Model Training**: Splits the data and trains the SVM model using a `Pipeline`.
4.  **Model Evaluation**: Evaluates the trained model on the validation set and prints a classification report.
5.  **Kaggle Submission**: Generates predictions on the official test set and saves the results to `submission.csv`.

## Model Evaluation

The model's performance is evaluated on a validation set (20% of the processed data). The final accuracy achieved is approximately **60.75%**.

The classification report provides a more detailed breakdown:

| Class | Precision | Recall | F1-Score | Support |
| :---: | :-------: | :----: | :------: | :-----: |
|  Cat  |   0.61    |  0.57  |   0.59   |   200   |
|  Dog  |   0.60    |  0.64  |   0.62   |   200   |

While not state-of-the-art, this result is a good baseline for a traditional machine learning approach on a complex image dataset.

## Submitting to Kaggle

After running the entire notebook, a `submission.csv` file will be generated. You can submit this file to the [Dogs vs. Cats Kaggle competition](https://www.kaggle.com/c/dogs-vs-cats) to see your score on the official leaderboard.

The final output file has the following format:

| id    | label |
| :---- | :---- |
| 1     | 1     |
| 2     | 1     |
| 3     | 0     |
| ...   | ...   |

Where `label` `1` corresponds to a 'dog' and `0` corresponds to a 'cat'.

## Dependencies

The project relies on the following Python libraries:

* `numpy`
* `pandas`
* `opencv-python`
* `scikit-image`
* `scikit-learn`
* `tqdm`
* `kaggle`

All dependencies are installed and managed within the Google Colab environment.

## Developed by

**Tara Siddappa Busaraddi**

* **GitHub**: [TaraB287](https://github.com/TaraB287)
* **LinkedIn**: [tarabusaraddi](https://www.linkedin.com/in/tarabusaraddi)
