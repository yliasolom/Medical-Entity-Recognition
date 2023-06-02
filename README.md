# **Medical Entity Recognition with Pretrained Transformers**

# Project Description:
In this notebook I aim to explore the capabilities of pretrained transformer models, with a particular focus on BERT (Bidirectional Encoder Representations from Transformers), for the task of identifying medical entities within textual data.

Medical entity recognition is a critical component of various healthcare applications, including clinical decision support systems, electronic health record analysis, and biomedical research.

---------------------------------------------
## Data
The [NCBI Disease](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/)  corpus is a valuable dataset consisting of 793 PubMed abstracts that have been annotated with 6,892 disease mentions. It serves as a valuable resource for researchers and developers working on tasks related to disease recognition and information extraction from biomedical literature.

The NCBI Disease corpus offers opportunities for training and evaluating models in the field of natural language processing and machine learning. It enables tasks such as disease recognition, named entity recognition, entity linking, and information extraction from scientific articles.

By leveraging the NCBI Disease corpus, researchers and developers can advance the state-of-the-art in biomedical text mining, contribute to the development of clinical decision support systems, and facilitate the discovery of novel insights and knowledge in the field of medicine.
(https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/annotationprocess.png)


## The NCBI disease corpus is fully annotated at the mention and concept level to serve as a research resource for the biomedical natural language processing community.


### Corpus Characteristics

793 PubMed abstracts
6,892 disease mentions
790 unique disease concepts
Medical Subject Headings (MeSH®)
Online Mendelian Inheritance in Man (OMIM®)
91% of the mentions map to a single disease concept
divided into training, developing and testing sets.

### Corpus Annotation
Fourteen annotators
Two-annotators per document (randomly paired)
Three annotation phases
Checked for corpus-wide consistency of annotations

The abstracts are split into sentences, which already have been tokenized for us. There are 5433 sentences in the training data, 924 in the validation data and another 941 in the test data.

-----------------------------

I choose one of the available PubMedBERTs — BERT models that have been pretrained on abstracts (and in this case, also full texts) from PubMed. I start by getting the tokenizer that was used for pretraining this model, because texts need to be tokenized in exactly the same manner!

Then I used this tokenizer to tokenize the texts (every sentence in our corpus is a list of words, so I need to tell the tokenizer the text has already been split into words). In addition, I'll  pad and truncate the texts. Sentences that are longer than 256 tokens will be truncated, and all sentences will be padded to the length of the (resulting) longest one.

### Evaluating the results

| Metric               | Value                 |
|----------------------|-----------------------|
| eval_loss            | 0.04833297058939934   |
| eval_accuracy        | 0.9847328244274809    |
| eval_f1              | 0.8363954505686789    |
| eval_precision       | 0.8363954505686789    |
| eval_recall          | 0.8363954505686789    |
| eval_runtime         | 6.0943                |
| eval_samples_per_second | 154.406              |
| eval_steps_per_second | 9.681                 |
| epoch                | 4.0                   |

 
 ### Based on the evaluation metrics, the BERT pretrained model performs well on the NCBI dataset corpus. Here are the key observations:

- Evaluation Loss: The evaluation loss of 0.048 indicates the average loss of the model's predictions on the evaluation dataset. A lower evaluation loss suggests that the model is making more accurate predictions.

- Evaluation Accuracy: The evaluation accuracy of 0.985 indicates the proportion of correctly predicted tokens in the evaluation dataset. A higher accuracy indicates that the model is performing well in correctly identifying and classifying tokens.

- Evaluation F1 Score: The evaluation F1 score of 0.836 measures the model's balance between precision and recall. It considers both false positives and false negatives, and a higher F1 score indicates a better overall performance.

- Evaluation Precision: The evaluation precision of 0.836 represents the proportion of correctly predicted positive (named entity) tokens out of all predicted positive tokens. It measures the model's ability to avoid false positives.

- Evaluation Recall: The evaluation recall of 0.836 represents the proportion of correctly predicted positive (named entity) tokens out of all true positive tokens. It measures the model's ability to capture all positive tokens.

- Evaluation Samples per Second: The evaluation samples per second value of 154.406 represents the number of samples processed by the model per second during evaluation. 


### It can be concluded that the BERT pretrained model demonstrates good performance on the NCBI dataset corpus, achieving high accuracy, precision, recall, and F1 score. The model appears to have learned the patterns in the data and is capable of identifying named entities effectively. 

# Conclusion:
The model identified several entities related to a genetic disorder called ataxia-telangiectasia. It recognized terms such as "ataxia," "telangiectasia," "recessive," "multi-system disorder," and "mutations in the ATM gene at 11q22." These findings suggest that the model is able to accurately identify and label relevant medical entities associated with ataxia-telangiectasia.
























----------------------------------------
-----------------------------------
### **Fake News Detection**

# This project aims to develop models that can distinguish between real and fake news articles based on their headlines. The project includes the implementation of two different approaches: Tf-Idf with Logistic Regression and LightGBM models as well as DistillBert model.

### Dataset:
The dataset used for training and evaluation consists of the following files:

- train.csv: This file contains the labeled training data. Each row contains a news headline and its corresponding label (0 for real news, 1 for fake news).
- test.csv: This file is provided for demonstration purposes and contains news headlines. The labels in this file are set to 0.

### Here're some of the project's best features:

*   Approach 1:
**Tf-Idf with Logistic Regression and LightGBM**

- The Tf-Idf-based models utilize the following preprocessing steps:

1.Lowercasing the text data.
2.Removing punctuation and extra spaces.
3.Removing stop words.
4.Lemmatizing the text using the WordNetLemmatizer from the NLTK library.
5.Vectorizing the corpus using the TfidfVectorizer.

- Two models are trained using this approach:

**Logistic Regression:** This  baseline model achieves the highest accuracy on the test set (83.98%) with relatively low training time (0.40 seconds) and prediction time (0.10 seconds).

**LightGBM with Unigram:** This model achieves an accuracy of 70.96% on the test set but has longer training time (2.18 seconds) and prediction time (0.05 seconds) compared to Logistic Regression.

**LightGBM with Bigram:** This model achieves the lowest accuracy on the test set (60.89%) with intermediate training time (0.99 seconds) and prediction time (0.05 seconds).

The models are evaluated using the F1 score metric, and the results are as follows:

Model               | F1 Score
--------------------|---------
Logistic Regression | 0.840
LightGBM Unigram    | 0.710
LightGBM Bigram     | 0.609


**The Logistic Regression model achieves the highest F1 score of *0.840***


*   Approach 2: DistillBert

In this approach, the DistillBert architecture was used for fake news detection. The following observations were made during the training process:

1.The model's performance consistently improved as the training progressed.
2.The F1 score and AUC-ROC metrics increased, indicating better classification performance and discrimination ability between real and fake news.
3.The training loss decreased over epochs, suggesting that the model learned from the training data.
4.The validation loss initially decreased and then fluctuated but remained relatively low, indicating good generalization to unseen data.
5.After 30 epochs, the model achieved an **F1 score 0.8087** and an **AUC-ROC score 0.8098**.

- These results suggest that the DistillBert model effectively learns the underlying patterns in the data and makes accurate predictions. 

---------------------------------------------------

### Instructions

1. Training and Prediction with DistillBert

- Train the model:

`python train_model_distillbert.py --train-file <path-to-train-file>`

This code is designed to train a sequence classification model using the DistilBERT architecture. This code trains a DistilBERT-based sequence classification model. Then code tokenizes the text data, creates PyTorch datasets, and performs training and validation. The model's performance is evaluated using F1 score and AUC-ROC. The loss curves are plotted and saved, and the trained model is saved as well.
To get predictions (being in a scr folder), use your tab-delimited file and see results in the report folder. Make sure you have a trained DistilBERT model available in the "models/distillbert_weights" directory before running this script.

- Get prediction:
`python predict_model_distillbert.py --test-file <your_test_file.csv>`

1. Training and Prediction with LinearRegression and LightGBM:

- Train the model:

`python train_model_reg.py --train-file <path-to-train-file>`

This code trains and saves two models, Logistic Regression and LightGBM, for fake news detection using a given training dataset. It performs the following steps:
Performs text preprocessing, including lowercasing, removal of stopwords, lemmatization, and stemming.
Uses grid search to find the best hyperparameters for both models.
Trains the models on the training data.
Evaluates the models using F1 score and plots the ROC curves.
Saves the trained models as pickle files in the "weights/basic" directory

- Get prediction:

`python predict_model_reg.py t<est_file> <lr_model_file> <lgbm_model_file> <lr_predictions_file> <lgbm_predictions_file>`

Replace the placeholders test_file, lr_model_file, lgbm_model_file, lr_predictions_file, and lgbm_predictions_file with the actual file paths you want to use.


`test_file:` Path to the test file containing the data on which you want to make predictions.
`lr_model_file:` Path to the logistic regression model file (.joblib) that you want to use for prediction.
`lgbm_model_file:` Path to the LightGBM model file (.joblib) that you want to use for prediction.
`lr_predictions_file:` Path to save the logistic regression predictions as a CSV file.
`lgbm_predictions_file:` Path to save the LightGBM predictions as a CSV file.


For example:
`python predict_model_reg.py data/test_data.csv models/lr_model.joblib models/lgbm_model.joblib predictions/lr_predictions.csv predictions/lgbm_predictions.csv`



### *Conclusion*
In this project, I developed models to classify news headlines as real or fake. The logistic regression model demonstrated the highest accuracy on the test dataset, while the LightGBM models achieved slightly lower performance, although it is not tends to be overfitted. 

Based on the F1 scores, the Logistic Regression model achieved the highest performance among the Tf-Idf-based models.

In the second approach, the DistillBert architecture was used for fake news detection. During the training process, the model's performance consistently improved as the training progressed. The F1 score and AUC-ROC metrics increased, indicating better classification performance and discrimination ability between real and fake news. The training loss decreased over epochs, suggesting that the model learned from the training data. The validation loss initially decreased and then fluctuated but remained relatively low, indicating good generalization to unseen data. After 30 epochs, the DistillBert model achieved an F1 score of 0.8087 and an AUC-ROC score of 0.8098.

These results indicate that the DistillBert model effectively learns the underlying patterns in the data and makes accurate predictions.

In conclusion, this project successfully developed models for classifying news headlines as real or fake. The logistic regression model demonstrated the highest accuracy on the test dataset, while the LightGBM models achieved slightly lower performance. The choice of model may depend on specific requirements, such as speed or interpretability. The DistillBert model also showed promising results, indicating its effectiveness in learning patterns and making accurate predictions.

The choice of model may depend on the specific requirements, such as speed or interpretability.

