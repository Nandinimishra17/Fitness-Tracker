import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix

#AIM-> Create Python scripts to process, visualize, and model accelerometer and gyroscope data 
# to create a machine learning model that can classify barbell exercises and count repetitions.

#Experiment with feature selection, model selection and hyperparameter tuning with grid search 
# to find the combination that results in the highest classification accuracy.

# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

df = pd.read_pickle("../../data/interim/03_data_features.pkl")


# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------
df_train = df.drop(["participant", "category", "set"], axis = 1) #for time being we dont want these columns

X = df_train.drop("label", axis = 1) #just a way of selection of X and y variables for splitting the train and testing dataset
y = df_train["label"] #take up exactly the col we dropped for X

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 42, stratify=y) #test_size = 0.25 meaning 25% of the data is left for testing and the rest 75% data for training. 
#The stratify parameter-> Since we're using a labeled data set we want to make sure that our train and test split up in such a way that they both contain enough labels of all the instances that we can pick from.
#so we dont want that all of our training set contains only/majority bench press and squat data and our test set contains all the rowing data. Thus equal distribution of all labels. 

#Now visualising on this distribution via a distrribution plot->
len(X_train)
len(X_test)
len(y_train)
len(y_test)
fig, ax = plt.subplots(figsize=(10, 5))
df_train["label"].value_counts().plot(
    kind="bar", ax=ax, color="lightblue", label="Total"
)
y_train.value_counts().plot(kind="bar", ax=ax, color="dodgerblue", label="Train") #since y was assigned the label col
y_test.value_counts().plot(kind="bar", ax=ax, color="royalblue", label="Test")
plt.legend()
plt.show()
# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------
#Splitting up the different features into subsets to later chack whether the additional features that we've added using in the future engineering phase are actually beneficial to the predictive performance of the models
#so we split up starting with the basic features that we originally had in our dataset
basic_features = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
square_features = ["acc_r", "gyr_r"]
pca_features = ["pca_1", "pca_2", "pca_3"]
time_features = [ f for f in df_train.columns if "_temp_" in f ] #in feature engineering we labelles all the features with regards to time- having notation temp, so we do a list comprehension (which helps in their extraction ia a niche lil way)saying for all f in training set cols, if _temp_ is there, add it to time_features list
frequency_features = [f for f in df_train.columns if ("_freq" in f) or ("_pse" in f)]
cluster_features = ["cluster"]

print("Basic Features: ", len(basic_features))
print("Square Features: ", len(square_features))
print("PCA Features: ", len(pca_features))
print("Time Features: ", len(time_features))
print("Frequency Features: ", len(frequency_features))
print("Cluster features: ", len(cluster_features))

#creating 4 diff feature sets, set will be used since sets remove duplicacy
feature_set_1 = list(set(basic_features))
feature_set_2 = list(set(basic_features + square_features + pca_features))
feature_set_3 = list(set( feature_set_2 + time_features))
feature_set_4 = list(set(feature_set_3 + frequency_features + cluster_features))


# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------
#for this we will be going to loop over all the individual features and start small(meaning a forward selection), try to see one individual feature using simple decision tree and see what accuracy is on scoring our labels, and then once we have the feature with highest accuracy- we will start this process all over again but by adding all new features to this best performing feature
#and as u add more features the accuracy will start to increase, bacause we give the model more information meaning that the model can learn from more data and thus becomes better at PREDICTING THE LABEL
#But the accuracy curve comes to a point after increasing after which it starts decreasing and the results start to deminish. So we shall not even provide too much info- making it a complex model

learner = ClassificationAlgorithms()

max_features = 10
selected_features, ordered_features, ordered_scores = learner.forward_selection(
    max_features, X_train, y_train
)#this is going to loop over all the individual cols in dataframe, and train a decision tree (no. of cols = 117 times)- gets the best parameter in its first iteration. Then in the next (2nd iteration), it's going to loop over 116 rest cols besides the best performing one and then its going to do all the training again. And similary for the next further iterations it does the same for the no. of max_features = 10 given. 
#accuracy predicted on trainig data- based on data it has already seen
#Feature selection - which of the features contribute most to the accuracy

selected_features = [
'acc_x_freq_0.0_Hz_ws_14',
 'duration',
 'acc_y_freq_0.0_Hz_ws_14',
 'acc_z_temp_mean_ws_5',
 'acc_z_freq_2.5_Hz_ws_14',
 'gyr_z_freq_weighted',
 'acc_z_freq_weighted',
 'gyr_y_freq_2.143_Hz_ws_14',
 'acc_r_freq_0.357_Hz_ws_14',
 'acc_y_temp_mean_ws_5'
]
ordered_scores= [0.9209527096996893,
 0.9924059371763894,
 0.9993096306523991,
 0.9996548153261995,
 1.0,
 1.0,
 1.0,
 1.0,
 1.0,
 1.0]

plt.figure(figsize=(10,5))
plt.plot(np.arange(1, max_features + 1, 1), ordered_scores)
plt.xlabel("Number of features")
plt.ylabel("Accuracy")
plt.xticks(np.arange(1, max_features + 1, 1))
plt.show()

# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# --------------------------------------------------------------
# Grid Search is a way in order to come up with the set of hyper parameters for your models so for each of the models- in this case scikit-learn models that we will be using can have a different parameters and you can set different values for them.
# And in order to find the optimal combinations, we define a grid search over all of the diff combinations that we wanna test for (multiplicative) 
#And then we validate our results using the k-fold cross validation and in this case we will be using 5- fold cross validation which also a way of splitting the dataset into train and test set without touching the original test set defined previously.

possible_feature_sets = [
    feature_set_1,
    feature_set_2,
    feature_set_3,
    feature_set_4,
    selected_features,
]

feature_names = [
    "Feature Set 1", 
    "Feature Set 2", 
    "Feature Set 3", 
    "Feature Set 4",
    "Selected Features",
]

iterations = 1
score_df = pd.DataFrame()
#given beneath block of code appears to be part of a grid search process for 
# evaluating the performance of various machine learning classifiers using different
# feature sets.

for i, f in zip(range(len(possible_feature_sets)), feature_names):
    print("Feature set:", i)
    selected_train_X = X_train[possible_feature_sets[i]]
    selected_test_X = X_test[possible_feature_sets[i]] #Extracts the selected features for both training and testing datasets.
    
    #Initializes variables to store the performance metrics for each classifier.
    # First run non deterministic classifiers  (Neural Network and Random Forest) to average their score. 
    performance_test_nn = 0
    performance_test_rf = 0

    for it in range(0, iterations):
        print("\tTraining neural network,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.feedforward_neural_network(
            selected_train_X,
            y_train,
            selected_test_X,
            gridsearch=False,
        )
        performance_test_nn += accuracy_score(y_test, class_test_y)

        print("\tTraining random forest,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.random_forest(
            selected_train_X, y_train, selected_test_X, gridsearch=True
        )
        performance_test_rf += accuracy_score(y_test, class_test_y)
        #For a specified number of iterations, it trains a feedforward neural network and a random forest classifier on the selected feature set.
        #It accumulates the performance scores (accuracy) obtained from each iteration.
        #After all iterations, the average performance scores for the neural network and random forest classifiers are calculated.
    
    performance_test_nn = performance_test_nn / iterations
    performance_test_rf = performance_test_rf / iterations

    # And we run our deterministic classifiers (K-Nearest Neighbors, Decision Tree, Naive Bayes), each of these classifiers on the selected feature set.
    #Calculates the performance score (accuracy) for each classifier. :
    print("\tTraining KNN")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.k_nearest_neighbor(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_knn = accuracy_score(y_test, class_test_y)

    print("\tTraining decision tree")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.decision_tree(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_dt = accuracy_score(y_test, class_test_y)

    print("\tTraining naive bayes")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.naive_bayes(selected_train_X, y_train, selected_test_X)

    performance_test_nb = accuracy_score(y_test, class_test_y)

    # Save results to dataframe
    models = ["NN", "RF", "KNN", "DT", "NB"]
    new_scores = pd.DataFrame(
        {
            "model": models,
            "feature_set": f,
            "accuracy": [
                performance_test_nn,
                performance_test_rf,
                performance_test_knn,
                performance_test_dt,
                performance_test_nb,
            ],
        }
    )
    score_df = pd.concat([score_df, new_scores])

#Overview Summary:

#The code performs a grid search over possible feature sets, evaluating the performance of various classifiers.
#For each feature set, it trains non-deterministic classifiers (Neural Network and Random Forest) multiple times and averages their performance.
#It then evaluates deterministic classifiers (K-Nearest Neighbors, Decision Tree, Naive Bayes) on the same feature set.
#Performance scores for all classifiers are stored in a DataFrame, which can be used for further analysis and comparison.
    
#Suppose you have a dataset with various features like age, income, and education level, 
# #and you want to predict whether a person will buy a product or not. Each feature set represents a combination of these features. 
# #The code evaluates how well different classifiers perform in predicting the buying behavior based on different combinations of features. 
# For example, one feature set might include only age and income, while another might include all three features.
# The code runs various classifiers (Neural Network, Random Forest, KNN, Decision Tree, Naive Bayes) on each feature set and stores the accuracy of each classifier in a DataFrame for analysis.
#Basically- kaunsa feature set (combination of various features/cols from our dataset) kaunse model pe best perfromance(most right prediction)/accuracy deta hai.
#By evaluating the performance of these classifiers on the dataset, we can select the model that offers the best balance of prediction accuracy, interpretability, and generalization ability for our specific task of detecting fraudulent credit card transactions.

# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------
#Previously we - on featured sets we were checking accuracy on training data, here using several diff classifiers, we trained our model on TEST DATA- these are results originated from unseen data
score_df.sort_values(by="accuracy", ascending=False)

plt.figure(figsize=(10,10))
sns.barplot(x = "model", y="accuracy", hue="feature_set", data=score_df)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0.7, 1)
plt.legend(loc="lower right")
plt.show()

# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------
#For us the best model is Decision tree

(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.decision_tree(
    X_train[selected_features], y_train, X_test[selected_features], gridsearch=True
)#we're going to validate on X_test
#test_y contains the true labels of the test set.
#class_test_y contains the predicted labels generated by the classifier for the test set. In machine learning, after training a classifier, it's important to evaluate its performance on unseen data, which is typically referred to as the test set. class_test_y stores the predicted labels generated by the classifier on this test set.
accuracy = accuracy_score(y_test, class_test_y)

classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels= classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show() 
#But now we'll look at the confusion matrix in order to see what are those few cases where it predicted the wrong labels and what's it doing right
#Now if we see the whole data set and pick a random record from the test set, there's a very high chance that somewhere in the training set there is a record that is almost identical.
#Thus you can see how it is easy for the model to determine that something was a bench press if it has been trained on data from the same set from that participant
#So the next and the final test to validate our approach is to split the train and the test split based on the participants
# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------
#So now we are going to subtract A from the participant data meaning that we will train on all but participant A, thus in that way we will provide the trained models with the data from a participant that was performing the exercises ata different time and is a completely different person.
#So the participant's way of doing a deadlift or anything else is slightly different from what the model has seen and what the model has been able to generalize to- which is the ultimate test.

participant_df = df.drop(["set", "category"], axis=1)

X_train = participant_df[participant_df["participant"] != "A"].drop("label", axis=1)
y_train = participant_df[participant_df["participant"] != "A"]["label"]

X_test = participant_df[participant_df["participant"] == "A"].drop("label", axis=1)
y_test = participant_df[participant_df["participant"] == "A"]["label"]

X_train = X_train.drop(["participant"], axis=1)
X_test = X_test.drop(["participant"], axis=1)

fig, ax = plt.subplots(figsize=(10, 5))
df_train["label"].value_counts().plot(
    kind="bar", ax=ax, color="lightblue", label="Total"
)
y_train.value_counts().plot(kind="bar", ax=ax, color="dodgerblue", label="Train") #since y was assigned the label col
y_test.value_counts().plot(kind="bar", ax=ax, color="royalblue", label="Test")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------
#a fitness tracker that can generalize to new participants.
(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.decision_tree(
    X_train[selected_features], y_train, X_test[selected_features], gridsearch=True
)#we're going to validate on X_test
#test_y contains the true labels of the test set.
#class_test_y contains the predicted labels generated by the classifier for the test set. In machine learning, after training a classifier, it's important to evaluate its performance on unseen data, which is typically referred to as the test set. class_test_y stores the predicted labels generated by the classifier on this test set.
accuracy = accuracy_score(y_test, class_test_y)

classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels= classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show() 
#this time the accuracy came as 97%

# --------------------------------------------------------------
# Try a simpler model with the selected features
# --------------------------------------------------------------
#for us the number 2 best model that came in was random forest
selected_features= [
 'acc_z_freq_0.0_Hz_ws_14',
 'acc_x_freq_0.0_Hz_ws_14',
 'gyr_r_freq_0.0_Hz_ws_14',
 "acc_z",
 'pca_1',
 'acc_r_temp_std_ws_5',
 "gyr_y_temp_std_ws_5",
 'acc_z_freq_0.357_Hz_ws_14',
 'gyr_z_freq_2.143_Hz_ws_14',
 'gyr_x_temp_std_ws_5'
    
]
selected_features =[
'acc_x_freq_0.0_Hz_ws_14',
 'duration',
 'acc_y_freq_0.0_Hz_ws_14',
 'acc_z_temp_mean_ws_5',
 'acc_z_freq_2.5_Hz_ws_14',
 'gyr_z_freq_weighted',
 'acc_z_freq_weighted',
 'gyr_y_freq_2.143_Hz_ws_14',
 'acc_r_freq_0.357_Hz_ws_14',
 'acc_y_temp_mean_ws_5'
]

(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.random_forest(
    X_train[feature_set_4], y_train, X_test[feature_set_4], gridsearch=True
)#we're going to validate on X_test
#test_y contains the true labels of the test set.
#class_test_y contains the predicted labels generated by the classifier for the test set. In machine learning, after training a classifier, it's important to evaluate its performance on unseen data, which is typically referred to as the test set. class_test_y stores the predicted labels generated by the classifier on this test set.
accuracy = accuracy_score(y_test, class_test_y)

classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels= classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show() 
print("For Random Forest")

#accuracy 93%, no. of wrong predictions abt ohp being a bench = 59
#with second collection of selected features, accuracy = close to 98%
#checking for neural network

(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.feedforward_neural_network(
    X_train[selected_features], y_train, X_test[selected_features], gridsearch=False
)#we're going to validate on X_test
#test_y contains the true labels of the test set.
#class_test_y contains the predicted labels generated by the classifier for the test set. In machine learning, after training a classifier, it's important to evaluate its performance on unseen data, which is typically referred to as the test set. class_test_y stores the predicted labels generated by the classifier on this test set.
accuracy = accuracy_score(y_test, class_test_y)

classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels= classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show() 

#accuracy 96%, no. of wrong predictions abt ohp being a bench = 48

#now with the new list of selected features-For random forest- accuracy = 97%, listed abv- no. of wrong predictions abt ohp being a bench = 26

(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.k_nearest_neighbor(
    X_train[selected_features], y_train, X_test[selected_features]
)#we're going to validate on X_test
#test_y contains the true labels of the test set.
#class_test_y contains the predicted labels generated by the classifier for the test set. In machine learning, after training a classifier, it's important to evaluate its performance on unseen data, which is typically referred to as the test set. class_test_y stores the predicted labels generated by the classifier on this test set.
accuracy = accuracy_score(y_test, class_test_y)

classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels= classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show() 