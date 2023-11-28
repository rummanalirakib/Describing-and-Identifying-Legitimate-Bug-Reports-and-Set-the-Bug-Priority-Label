import csv
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def dateDifference(date1, date2):
    # Input date strings
    date_str1 = date1
    date_str2 = date2
    
    # Parse the date strings into date objects
    date1 = datetime.strptime(date_str1, "%Y-%m-%d").date()
    date2 = datetime.strptime(date_str2, "%Y-%m-%d").date()
    
    # Calculate the date difference
    date_difference = date1 - date2
    return abs(date_difference.days)


def LableMapping(resolution):
    if resolution == 'FIXED' or resolution=='WONTFIX':
        return "Valid Bug"
    else:
        return "Invalid Bug"


bugNum = {}
recentBugNum = {}
dict = {}
validBug = {}
validRate = {}
ReporterExperience = []
finalLabels = []
# Open the CSV file for reading
with open('New DataSet.csv', 'r', encoding='ISO-8859-1') as csv_file:
    # Create a CSV reader
    csv_reader = csv.reader(csv_file)
    csv_data = list(csv_reader)
    for row in reversed(csv_data):
        key = row[6]
        resolution = row[4]
        
        if key in dict:
            dict[key] += 1
        else:
            dict[key] = 1
        
        bugNum[row[0]] = dict[key]
        
        if resolution == 'FIXED' or resolution=='WONTFIX':
            if key in validBug:
                validBug[key] += 1
            else:
                validBug[key] = 1
        else:
            if key not in validBug:
                validBug[key] = 0
                
        if dict[key] != 0:
            validRate[row[0]] = validBug[key] / dict[key]
        else:
            validRate[row[0]] = 0
        
    
    # Iterate through each row in the CSV file
    print(len(csv_data))
    flag=0
    for i in range(0, len(csv_data)):
        finalLabels.insert(i, LableMapping(csv_data[i][4]))
        if flag==0:
            flag=1
            continue

        count = 0
        for j in range(i+1, len(csv_data)):
            temp = dateDifference(csv_data[i][5].split()[0], csv_data[j][5].split()[0])
            if temp > 90 :
                break
            
            if csv_data[i][6] == csv_data[j][6]:
                count = count +1
        
        recentBugNum[csv_data[i][0]] = count
        
    
df_validRate = pd.DataFrame(validRate.values(), columns=['ValidRate'])
df_bugNum = pd.DataFrame(bugNum.values(), columns=['BugNum'])
df_recentBugNum = pd.DataFrame(recentBugNum.values(), columns=['RecentBugNum'])
    
# Concatenate the DataFrames horizontally
merged_df = pd.concat([df_validRate, df_bugNum, df_recentBugNum], axis=1)
print(merged_df)

# Iterate through the DataFrame
for i in range(0, len(merged_df)):
    for j in range(0, 3):  # Assuming you want to check the first three columns
        if pd.isna(merged_df.iloc[i, j]):
            merged_df.iat[i, j] = 0

    
        

print(len(merged_df))
print(len(finalLabels))

# Step 1: Initialize the StandardScaler
scaler = StandardScaler()

# Step 2: Fit and transform the features using the scaler
scaled_features = scaler.fit_transform(merged_df)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaled_features, finalLabels, test_size=0.3, random_state=42, stratify=None)
print(len(merged_df))
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
svc_model = SVC(kernel='linear')
rf_classifier.fit(X_train, y_train)



print(len(merged_df))
# Make predictions using SVC model
y_pred = rf_classifier.predict(X_test)

# Calculate classification metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
f1 = f1_score(y_test, y_pred, average='micro')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print("Classification Report:\n", report)
   
    
            
