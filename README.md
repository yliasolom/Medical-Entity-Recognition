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
