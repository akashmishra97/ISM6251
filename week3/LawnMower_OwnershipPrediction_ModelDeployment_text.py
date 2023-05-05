import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pickle

# Uncomment the following snippet of code to debug problems with finding the .pkl file path
# This snippet of code will exit the program and print the current working directory.
# import os
# exit(os.getcwd())

LawnMower_Model = pickle.load(open('./data/svm_lin_model.pkl', "rb"))

print("\n*****************************************************")
print("* The USF Super Simple Model to predict Ownership of Lawn Mowers *")
print("*****************************************************\n")
Income = float(input("Enter the Household Income "))
Lot_Size = float(input("Enter the Property lot size "))
df = pd.DataFrame({'Income': [Income], 'Lot_Size': [Lot_Size]})
result = LawnMower_Model.predict(df)
Ownership = ('Potential lawn mower Owner', 'Not a Potential lawn mower Owner')
print(f"\nThe USF Simple Model to predict Ownership of Lawn Mowers indicates that the property's owner is {Ownership[result[0]]}.\n")
