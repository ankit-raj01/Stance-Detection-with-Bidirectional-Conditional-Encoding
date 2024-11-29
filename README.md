# Stance Detection with Bidirectional Conditional Encoding

In this project, we have a dataset of approximately 5000 tweets. Corresponding to each tweet, we have a target, i.e., a person or subject to which the tweet is targeting, and a stance as ‘against’, ‘favor’, or ‘none’. Our goal is to predict the stance towards a previously unseen target that isn't present in the training data. In our dataset, the training data comprises tweets targeting one of the five targets, i.e., ‘Climate Change is a Real Concern’, ‘Atheism’, ‘Feminist Movement’, ‘Hillary Clinton’, and ‘Legalization of Abortion’. However, we have to predict the stance towards the target ‘Donald Trump’, which isn't present in the training data. We have used the training data with the ‘Hillary Clinton’ target as validation set because, like ‘Donald Trump’, ‘Hillary Clinton’ is also a politician, and the model is expected to perform similarly on tweets targeting politicians.

We have converted our output label from categorical data to numerical. Then, we have removed extra whitespaces, numbers, punctuations, URLs, and HTML data and converted everything to lowercase. Further, we have performed lemmatization, and converted each word of our tweet and target to a 100-dimensional embedding using word2vec. Then, we have prepared our dataset using ‘TensorDataset’ and passed it to the ‘DataLoader’.

We have created a ‘BiLSTM’ class with two LSTMs, one for the target and another one for the tweet. We have passed input targets to the LSTM for the target with initial hidden and cell states initialized to zero, and after all the targets have been passed, we have obtained the final hidden and cell states as ‘ht’ and ‘ct’ respectively. We have passed this obtained cell state ‘ct’ as the initial cell state to the LSTM for the tweet. This is called Conditional Encoding. After all the tweets have been passed, we have obtained the final hidden and cell states as ‘hT’ and ‘cT’ respectively. We have passed this obtained hidden state through a linear layer and then through a Softmax, as we have to perform multiclass classification. We have used the Cross Entropy loss and the Adam optimizer. We have trained our model with the batch size set to 22, the learning rate set to 1e-5, and the number of epochs set to 11. Finally, on testing, we have achieved an Accuracy and F1-score of around 35.23%. The low score has been primarily due to using a limited subset of the original dataset used in reference paper, as the complete dataset has not been publicly accessible.

## LSTM
https://github.com/user-attachments/assets/7a28030b-dc93-49bd-80cf-66a9f8fda519

## One-directional LSTMs
https://github.com/user-attachments/assets/a72e4d2e-1f7a-4926-8fff-6282d0baa2ce

## Bidirectional Conditional Encoding of LSTMs
![Screenshot (4036)](https://github.com/user-attachments/assets/2daceede-d535-4664-863b-72fd697e3979)
