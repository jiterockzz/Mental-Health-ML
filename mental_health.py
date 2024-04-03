import warnings
import pickle
warnings.filterwarnings("ignore")
#importing 
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv("survey.csv")
male_str = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man","msle", "mail", "malr","cis man", "Cis Male", "cis male"]
trans_str = ["trans-female", "something kinda male?", "queer/she/they", "non-binary","nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]
female_str = ["cis female", "f", "female", "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail"]

for(row,col) in data.iterrows():
    if str.lower(col.Gender) in male_str:
        data['Gender'].replace(to_replace=col.Gender, value='male', inplace=True)
    
    if str.lower(col.Gender) in female_str:
        data["Gender"].replace(to_replace=col.Gender, value='female', inplace=True)

    if str.lower(col.Gender) in trans_str:
        data["Gender"].replace(to_replace=col.Gender, value='trans', inplace=True)
        
#removing unnecessary points
stk_list = ['A little about you', 'p']
data = data[data['Gender'].isin(stk_list)]
data['Gender']=data['Gender'].map({'male':0,'female':1, 'trans':2})
data['family_history']=data['family_history'].map({'No':0,'Yes':1})
data['treatment']=data['treatment'].map({'No':0,'Yes':1})

data = np.array(data)

X = data[1:,1:-1]
y = data[1:, -1]
y = y.astype('int')
X = X.astype('int')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# with open('model.pkl', 'wb')as f:
#     pickle.dump(classifier,f)
pickle.dump(classifier,open('stacking_model.pkl','wb'))
model =pickle.load(open('stacking_model.pkl','rb'))