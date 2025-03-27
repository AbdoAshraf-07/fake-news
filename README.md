
Fake News Detection Using Machine Learning
Overview
This project aims to detect fake news articles using Machine Learning (ML) algorithms. With the rapid spread of misinformation on the internet, detecting fake news has become a critical task. This project uses natural language processing (NLP) and machine learning models to classify news articles as real or fake.

Project Structure
The project is structured into several key parts:

Data Preprocessing: Cleaning and preparing the dataset by removing noise, normalizing the text, and tokenizing the data.

Feature Extraction: Using TF-IDF (Term Frequency-Inverse Document Frequency) to extract important features from the text.

Model Training: Training the machine learning models, specifically Random Forest and XGBoost.

Model Evaluation: Evaluating the models using metrics such as accuracy, confusion matrix, and classification report.

Technologies Used
Python for programming

Scikit-learn for machine learning tasks (feature extraction and model training)

XGBoost for gradient boosting classifier

NLTK (Natural Language Toolkit) for text preprocessing

Pandas and NumPy for data manipulation

Matplotlib and Seaborn for visualization

Getting Started
Prerequisites
To run this project locally, you need to install the following libraries:

bash
Copy
Edit
pip install pandas scikit-learn xgboost nltk matplotlib seaborn
Dataset
The dataset used in this project is the Fake and Real News Dataset available from Kaggle. You can download it using the following link:

Fake and Real News Dataset - Kaggle

Project Workflow
Preprocessing the Data:

The data is cleaned by removing special characters, converting text to lowercase, and removing stop words.

Lemmatization is applied to reduce words to their root form.

Extracting Features:

TF-IDF is used to transform the textual data into a numerical format suitable for machine learning models.

Training the Models:

Random Forest Classifier and XGBoost models are trained on the extracted features.

Evaluating the Models:

The models are evaluated using accuracy scores, confusion matrix, and classification reports.

Notebooks
This project is divided into 4 Jupyter Notebooks:

data_preprocessing.ipynb: For cleaning and preparing the data.

feature_extraction.ipynb: For extracting features from the cleaned text.

train_model.ipynb: For training the machine learning models.

evaluate_model.ipynb: For evaluating the performance of the trained models.

How to Use
Clone the repository or download the project files.

Run each Jupyter Notebook in sequence:

First, run data_preprocessing.ipynb to clean the data.

Then, run feature_extraction.ipynb to extract features.

Follow it with train_model.ipynb to train the models.

Finally, run evaluate_model.ipynb to evaluate the models' performance.

Results
The models are evaluated on a test set, and their accuracy is calculated. The XGBoost model performs well, with a high classification accuracy in detecting fake and real news.

Contributing
Feel free to fork the repository, submit issues, and make pull requests. Contributions are welcome!

License
This project is licensed under the MIT License - see the LICENSE file for details.

