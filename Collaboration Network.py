from sklearn.ensemble import RandomForestClassifier
import csv
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import requests
import networkx as nx

labelMapping = {}
nodes = 0

def dateDifference(date1, date2):
    # Check if either date string is empty
    if not date1 or not date2:
        return None  # or handle this case according to your requirements

    # Parse the date strings into date objects
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

def emailMapping(email):
    global nodes
    if email not in labelMapping:
        nodes += 1
        labelMapping[email] = nodes
        
    return labelMapping[email]

def largestConnectedComponen(G, reporterKey):
    # Find connected components
    connected_components = list(nx.connected_components(G))

    # Find the largest connected component
    largest_connected_component = max(connected_components, key=len)
    if reporterKey in largest_connected_component:
        return True
    else:
        return False
    

connectedEdge = [[] for _ in range(500005)]
InDegree = {}
OutDegree = {}
         
finalLabels = []
dict = {}
count1=0;
NetWorkInDegree = {}
NetWorkOutDegree = {}
NetworkTotalDegree = {}
ClosenessCentrality = {}
LongestConnectedComponent = {}
BetweenNessCentrality = {}
EigenVectorCentrality = {}
ClusteringCoefficient = {}
KCoreNess = {}
prevDate = ""
tempCommentInfo = ""
G = nx.Graph()
Grph = nx.Graph()
# Open the CSV file for reading
with open('Another DataSet.csv', 'r', encoding='utf-8') as csv_file:
    # Create a CSV reader
    csv_reader = csv.reader(csv_file)
    csv_data = list(csv_reader)
    for row in reversed(csv_data):
        reporterKey = emailMapping(row[6])
        resolution = row[4]
        bug_id = row[0]  # Replace with the desired bug ID
        commentUser = row[8]
        separetedCommentUser = commentUser.split(',')
        count1=count1+1
        finalLabels.insert(count1, LableMapping(row[4]))
        
        for comment in separetedCommentUser:
            commentUserkey = emailMapping(comment)
            
            connectedEdge[commentUserkey].append(reporterKey)
            G.add_edge(commentUserkey, reporterKey)
            Grph.add_edge(commentUserkey, reporterKey)
            if reporterKey == commentUserkey:
                if reporterKey not in OutDegree:
                    OutDegree[reporterKey] = 1
                else:
                    OutDegree[reporterKey] += 1 
            else:
                if reporterKey not in InDegree:
                    InDegree[reporterKey] = 1
                else:
                    InDegree[reporterKey] += 1
                    
        if reporterKey not in InDegree:
            InDegree[reporterKey] = 0
            
        NetWorkInDegree[bug_id] = InDegree.get(reporterKey, 0)
        NetWorkOutDegree[bug_id] = OutDegree.get(reporterKey, 0)
        NetworkTotalDegree[bug_id] = InDegree.get(reporterKey, 0) + OutDegree.get(reporterKey, 0)
        LongestConnectedComponent[bug_id] = largestConnectedComponen(G, reporterKey)
            
        if InDegree[reporterKey] > 0:
           ClosenessCentrality[bug_id] = 1/InDegree[reporterKey]
        else:
           ClosenessCentrality[bug_id] = 0
    
        if count1==0:
            prevDate=row[5]
            tempCommentInfo = separetedCommentUser
        else:
            date_diff = dateDifference(prevDate, row[5])
            if date_diff is not None and date_diff >= 30:
                for comment in tempCommentInfo:
                    commentUserkey = emailMapping(comment)
                    
                    if reporterKey in connectedEdge[commentUserkey]:
                        # Remove the connection
                        G.remove_edge((commentUserkey, reporterKey))
                        

                    if reporterKey == commentUserkey:
                        OutDegree[reporterKey]-=1

                    else:
                        InDegree[reporterKey]-=1

    betweenness_centrality = nx.betweenness_centrality(Grph)
    eigenvector_centrality = nx.eigenvector_centrality(Grph)
    clustering_coefficient = nx.clustering(Grph)
    Grph.remove_edges_from(nx.selfloop_edges(Grph))

    k_value = 2
    k_core_subgraph = nx.k_core(Grph, k=k_value)    
    k_coreness = nx.core_number(Grph)
    for row in reversed(csv_data):
        reporterKey = emailMapping(row[6])
        bug_id = row[0]
        if reporterKey in betweenness_centrality:
            BetweenNessCentrality[bug_id] = betweenness_centrality[reporterKey]
        else:
            BetweenNessCentrality[bug_id] = 0
            
        if reporterKey in EigenVectorCentrality:
            EigenVectorCentrality[bug_id] = eigenvector_centrality[reporterKey]
        else:
            EigenVectorCentrality[bug_id] = 0
            
        if reporterKey in clustering_coefficient:
            ClusteringCoefficient[bug_id] = clustering_coefficient[reporterKey]
        else:
            ClusteringCoefficient[bug_id] = 0
            
        if reporterKey in k_coreness:
            KCoreNess[bug_id] = k_coreness[reporterKey]
        else:
            KCoreNess[bug_id] = k_coreness[reporterKey]


df_NetWorkInDegree =  pd.DataFrame(NetWorkInDegree.values(), columns=['In Degree'])
df_NetWorkOutDegree = pd.DataFrame(NetWorkOutDegree.values(), columns=['Out Degree'])
df_NetworkTotalDegree = pd.DataFrame(NetworkTotalDegree.values(), columns=['Total Degree'])
df_ClosenessCentrality = pd.DataFrame(ClosenessCentrality.values(), columns=['Closeness Centrality'])
df_LongestConnectedComponent = pd.DataFrame(LongestConnectedComponent.values(), columns=['Longest Connected Component'])
df_BetweenNessCentrality = pd.DataFrame(BetweenNessCentrality.values(), columns=['Between ness Centrality'])
df_EigenVectorCentrality = pd.DataFrame(EigenVectorCentrality.values(), columns=['Eigen Vector Centrality'])
df_ClusteringCoefficient = pd.DataFrame(ClusteringCoefficient.values(), columns=['Clustering CoEfficient'])
df_KCoreNess = pd.DataFrame(KCoreNess.values(), columns=['K Coreness'])
merged_df = pd.concat([df_NetWorkInDegree, df_NetWorkOutDegree, df_NetworkTotalDegree, df_ClosenessCentrality, df_LongestConnectedComponent, df_BetweenNessCentrality, df_EigenVectorCentrality, df_ClusteringCoefficient, df_KCoreNess], axis=1)
print(merged_df)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(merged_df, finalLabels, test_size=0.3, random_state=42, stratify=None)

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
   
    