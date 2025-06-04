from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

# Train Logistic Regression model
logR = LogisticRegression()
logR.fit(X_train, y_train)
predictR = logR.predict(X_test)

# Evaluate the model
print('Classification report of Logistic Regression Results:')
print(classification_report(y_test, predictR))

# Confusion matrix
cm = confusion_matrix(y_test, predictR)
print('Confusion Matrix result of Logistic Regression is:\n', cm)

# Sensitivity and Specificity
sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity:', sensitivity)
specificity = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity:', specificity)

# Cross validation accuracy
accuracy = cross_val_score(logR, X, y, scoring='accuracy')
print('Cross validation test results of accuracy:')
print(accuracy)
print("Accuracy result of Logistic Regression is:", accuracy.mean() * 100)

# Plot confusion matrix
def plot_confusion_matrix(cm, title='Confusion matrix-LogisticRegression', cmap=plt.cm.Blues):
    target_names = ['Predict', 'Actual']
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

plot_confusion_matrix(cm)
sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%')
