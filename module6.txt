from sklearn.naive_bayes import GaussianNB

# Train Naive Bayes Classifier
nb = GaussianNB()
nb.fit(X_train, y_train)
predictR = nb.predict(X_test)

# Evaluate the model
print('Classification report of Naive Bayes Results:')
print(classification_report(y_test, predictR))

# Confusion matrix
cm = confusion_matrix(y_test, predictR)
print('Confusion Matrix result of Naive Bayes is:\n', cm)

# Sensitivity and Specificity
sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity:', sensitivity)
specificity = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity:', specificity)

# Cross validation accuracy
accuracy = cross_val_score(nb, X, y, scoring='accuracy')
print('Cross validation test results of accuracy:')
print(accuracy)
print("Accuracy result of Naive Bayes is:", accuracy.mean() * 100)

# Plot confusion matrix
plot_confusion_matrix(cm, title='Confusion matrix-NaiveBayesClassifier')
sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%')
