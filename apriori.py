# Apriori

# Importing the libraries
import numpy as np
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None, na_values = True)
dataset.fillna(0, inplace = True)
###

transactions = []
for i in range(0,len(dataset)):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20) if str(dataset.values[i,j])!='0'])
##for i in range(0, 7501):
  ##  transactions.append([str(dataset.values[i,j]) for j in range(0, 20)if str(dataset.values[i,j])! = 0])
    
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules  
one_hot_encoding =TransactionEncoder()
## transform the data into one_hot_encoding format
one_hot_txns = one_hot_encoding.fit(transactions).transform(transactions, sparse=False)
one_hot_txns.astype("int")

## convert the matrix into dataframe
one_hot_txns_df=pd.DataFrame(one_hot_txns, columns= one_hot_encoding.columns_)
##one_hot_txns_df.drop(columns= 'nan', axis = 1)
len(one_hot_txns_df.columns)

## checking the dataset
one_hot_txns_df.iloc[5:40, 2:5]
##
frequent_itemset =apriori(one_hot_txns_df, min_support= 0.003, use_colnames=True)
###
frequent_itemset.sample(10, random_state= 90)
one_hot_txns_df.columns
len(one_hot_txns_df.columns)
##
rules = association_rules(frequent_itemset, metric="lift", min_threshold=1)
rules.sample(5)
###
b=rules.sort_values('lift', ascending = False)[0:10]
dataset.columns


# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, max_length = 3)

# Visualising the results
results = list(rules)
##
def inspect(results):
    lhs =[tuple(result[2][0][0])[0]for result in results]
    rhs =[tuple(result[2][0][1])[0]for result in results]
    support =[result[1] for result in results]
    confidence =[result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(lhs,rhs, support, confidence, lifts))
resultsindataframe =pd.DataFrame(inspect(results), columns =['Left Hand Side', 'Right Hand Side', 'Support',
                                 'Confidence', 'Lifts'])
