Breast Cancer Classification
This code performs classification of breast cancer using various machine learning algorithms. It includes the following steps:

Data Loading: The code imports the Pandas library and uses it to load the dataset from a CSV file.

Data Preprocessing: The dataset is preprocessed to prepare it for training the classifiers. The 'diagnosis' column is mapped to numerical values (0 for 'B' and 1 for 'M') 

to represent benign and malignant diagnoses, respectively. The dataset is split into input features (X) and target variable (Y).

Train-Test Split: The dataset is split into training and testing sets using the train_test_split function from Scikit-learn. The testing set size is set to 25% of the total data.

Feature Scaling: The input features are standardized using the StandardScaler from Scikit-learn to ensure that all features have a similar scale.

Model Selection: Several classification algorithms are selected for comparison. The chosen models are Logistic Regression, Linear Discriminant Analysis, K-Nearest Neighbors, Decision
Tree Classifier, Gaussian Naive Bayes, and Support Vector Machine. These models are initialized with their respective default hyperparameters.

Cross-Validation: The models are evaluated using k-fold cross-validation with 10 folds. The StratifiedKFold class from Scikit-learn is used to ensure that the class distribution is preserved
during the splitting process. The accuracy scores of each model are calculated using the cross_val_score function.

Algorithm Comparison: The average accuracy scores of the models are stored in the res list. The names of the models are stored in the names list. The comparison is visualized using a bar chart 
using the Matplotlib library.

Model Training and Prediction: A Logistic Regression model is selected, initialized, and trained using the preprocessed training data. The trained model is then used to predict the diagnosis of 
a new sample defined in the value variable. The predicted diagnosis is printed to the console.
