import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle as pickle

def create_model(data):
    
    X = data.drop(columns=['diagnosis'], axis=1)
    y = data['diagnosis']
    
    # scale data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # split data
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=66)
    
    # train model
    model =LogisticRegression()
    model.fit(X_train, y_train)
    
    # test model
    y_pred = model.predict(X_test)
    print(f"Accuracy: ", accuracy_score(y_test, y_pred))
    print(f"Classification Report: ", classification_report(y_test, y_pred))    
    return model, scaler, 

    
def get_clean_data():
    data = pd.read_csv('data/data.csv')
    
    data = data.drop(columns=['Unnamed: 32', 'id'])
    
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data
    
    
def main():
    data = get_clean_data()
    
    print(data.info())
    
    model, scaler = create_model(data)
    
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    

    
if __name__ == "__main__":
    main()