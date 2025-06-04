# Calculating the accuracies for each model
logR = LogisticRegression()
logR.fit(X_train, y_train)
logR_accuracy = cross_val_score(logR, X, y, scoring='accuracy').mean() * 100

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_accuracy = cross_val_score(rfc, X, y, scoring='accuracy').mean() * 100

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc_accuracy = cross_val_score(dtc, X, y, scoring='accuracy').mean() * 100

nb = GaussianNB()
nb.fit(X_train, y_train)
nb_accuracy = cross_val_score(nb, X, y, scoring='accuracy').mean() * 100

# Creating a DataFrame to display the accuracies
accuracy_data = {
    'Algorithm': ['Logistic Regression', 'Random Forest Classifier', 'Decision Tree Classifier', 'Naive Bayes'],
    'Accuracy (%)': [logR_accuracy, rfc_accuracy, dtc_accuracy, nb_accuracy]
}

accuracy_df = pd.DataFrame(accuracy_data)

# Displaying the DataFrame
print(accuracy_df)

# Plotting the accuracies for comparison
plt.figure(figsize=(10, 6))
sns.barplot(x='Algorithm', y='Accuracy (%)', data=accuracy_df)
plt.title('Accuracy Comparison of Different Algorithms')
plt.ylabel('Accuracy (%)')
plt.xlabel('Algorithm')
plt.xticks(rotation=45)
plt.show()
