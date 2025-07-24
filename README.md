# k-Nearest Neighbors (k-NN)

This project is an implementation of k-NN without the use of advanced plug and play libraries. This code was made to analyse and manually go through the logic of k-NN algorithm for learning.

## What is k-NN?
- Supervised machine learning algorithm 
- Used for Classification and Regression
- Lazy learner
    - No training phase
    - Stores the data and does all computation at prediction time.

## Working Principle: 
- For a given input, k-NN finds the k closest training examples (neighbors) and makes a prediction based on them.

"Closeness": Measured using distance metrics like Euclidean, Manhattan, etc.

### Tunning Parameters
- k : (number of neighbors).

### Output Type:

- Classification: Take the majority label among the k nearest neighbors.

- Regression: Take the average value of the k nearest neighbors.




