# Naïve Bayes classifier algorithm

Dataset : Yelp dataset (kaggle : https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset) 

## **Naive Bayes Model for Yelp Star Rating Prediction**
### Overview
The Naive Bayes model developed in this project predicts star ratings for Yelp reviews based on their content. Leveraging the Naive Bayes algorithm, a popular choice for text classification tasks, the model analyzes review text and assigns a star rating ranging from 1 to 5.

### Dataset
The model is trained and evaluated on the Yelp dataset, which consists of a large collection of user-generated reviews across diverse businesses and locations. Each review in the dataset is associated with a star rating, serving as the target variable for prediction.

### Methodology
* Preprocessing: The review text undergoes preprocessing steps such as tokenization, removal of stop words, and stemming or lemmatization to extract meaningful features.
* Model Training: The Naive Bayes classifier is trained on the preprocessed dataset, learning the probabilistic relationship between features (words) and star ratings. The model calculates the likelihood of each star rating given the observed features.
* Prediction: Given a new review input, the trained model predicts the star rating by evaluating the likelihood of each rating based on the review text. The rating with the highest probability is assigned to the input review.
### Results
The Naive Bayes model demonstrates promising performance in predicting star ratings for Yelp reviews. By analyzing the textual content of reviews, the model accurately assigns star ratings, providing valuable insights for businesses and consumers to evaluate the quality and sentiment of reviews.

### Future Improvements
To further enhance the model's predictive accuracy and robustness, potential avenues for improvement include:

* Experimenting with different feature representations, such as word embeddings or contextualized embeddings (e.g., BERT).
* Incorporating domain-specific features or metadata from Yelp reviews, such as review length, user information, or business categories.
* Fine-tuning model hyperparameters or exploring ensemble methods to optimize performance on diverse review datasets.
