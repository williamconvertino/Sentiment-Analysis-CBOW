# Sentiment Analysis with CBOW-Style Neural Networks

This project aims to train a neural network for sentiment analysis, predicting whether a given body of text expresses positive or negative sentiment.

## Model Architecture
The model consists of a word embedding layer followed by a dense layer with a sigmoid activation function. We will need to train the word embedding vectors as well as the weights and bias of our dense layer.

### Feature Representation
For each document, the feature vector is computed as the average of word embeddings for each word in the document.

## Loss Function
Binary cross-entropy loss function is used to evaluate model performance. Adam is used to improve the efficiency of our training.

## Training, Testing, and Validation Sets
Training and testing sets are provided, and 20% of the training data is reserved for a validation set. Early stopping is implemented using TensorFlow's callback mechanism.

## Evaluation
The model achieves high accuracy and low loss on both training and validation sets. Evaluation on the test set demonstrates strong performance, indicating success in the sentiment analysis task. On the testing set, it acheived a total loss of 0.2388 and an accuracy of 0.9360.

## Conclusion
Despite the model's simplicity and lack of consideration for word order, it performs well in practical applications of sentiment analysis.
