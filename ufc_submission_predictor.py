import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
df_fights = pd.read_csv('/kaggle/input/datasets/rajeevw/ufcdata/raw_total_fight_data.csv', sep=';')
df_fighters = pd.read_csv('/kaggle/input/datasets/rajeevw/ufcdata/raw_fighter_details.csv')
df_fighters.isnull().sum()
df_fighters = df_fighters.drop(columns=['Reach', 'Stance', 'DOB'])
df_fighters['Weight'] = df_fighters['Weight'].str.replace('lbs.' , '').astype(float)
df_fighters['Weight'] = df_fighters['Weight'] * 0.453592
def convert_height(h):
    if pd.isna(h):
        return None
    parts = h.split("'")
    feet = int(parts[0])
    inches = int(parts[1].replace('"', '').strip() or 0)
    return feet * 30.48 + inches * 2.54
df_fighters['Height'] = df_fighters['Height'].apply(convert_height)
df_fighters['Height'] = df_fighters['Height'].fillna(df_fighters['Height'].mean())
df_fighters['Weight'] = df_fighters['Weight'].fillna(df_fighters['Weight'].mean())
df = df_fights[['R_fighter', 'B_fighter', 'win_by', 'Fight_type']].copy()
df = df.merge(df_fighters , left_on='R_fighter' , right_on='fighter_name')
df = df.merge( df_fighters , suffixes = ('_R' ,'_B'), left_on='B_fighter' , right_on='fighter_name')
df['win_by'].value_counts()
df['target'] = np.where(df['win_by'] == 'Submission' ,  1 , 0  )
X = df.drop(columns=['R_fighter', 'B_fighter', 'win_by', 'Fight_type', 'fighter_name_R' , 'fighter_name_B' , 'target'])
y = df['target']
cols = ['Str_Acc_R', 'Str_Def_R' , 'TD_Acc_R' , 'TD_Def_R' , 'Str_Acc_B' , 'Str_Def_B' , 'TD_Acc_B' , 'TD_Def_B']
for col in cols:
    X[col] = X[col].str.replace('%', '').astype(float)
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)
scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)
model = RandomForestClassifier()
model.fit(X_train_scaler , y_train)
y_pred = model.predict(X_test_scaler)
print(classification_report(y_test, y_pred))
