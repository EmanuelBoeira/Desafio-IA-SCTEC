import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

df = pd.read_csv('./hotel_bookings.csv')

#data cleaning
#company has most null values, so can be dropped
df.drop(columns=['company'], inplace=True)

#keep these columns probably caused data-leakage
df.drop(columns=['reservation_status_date'], inplace=True)
df.drop(columns=['reservation_status'], inplace=True)

#children and country has low null values, so they has been replaced by the most common value
df['children'].fillna(df['children'].mode()[0], inplace=True)
df['country'].fillna(df['country'].mode()[0], inplace=True)

#the null values in agent has been replaced by 1 that was chosen as a value that represent the no_agent
df['agent'].fillna(1, inplace=True)

#------------------------------------------
x = df.drop(columns=['is_canceled'])
y = df['is_canceled']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

numeric_features = ['lead_time', 'arrival_date_year', 'arrival_date_week_number', 'arrival_date_day_of_month', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies', 'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled', 'booking_changes', 'agent', 'days_in_waiting_list', 'adr', 'required_car_parking_spaces', 'total_of_special_requests']
categorical_features = ['arrival_date_month', 'meal', 'country', 'market_segment', 'distribution_channel', 'reserved_room_type', 'assigned_room_type', 'deposit_type', 'customer_type']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='drop'
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

model_lr = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
model_lr.fit(X_train_processed, y_train)

y_pred_lr = model_lr.predict(X_test_processed)
y_proba_lr = model_lr.predict_proba(X_test_processed)[:,1]

print("\nAccuracy:", accuracy_score(y_test, y_pred_lr))
print("\nAUC:", roc_auc_score(y_test, y_proba_lr))

#conclusão: após a predição do modelo é possível ver que as métricas Accuracy = 0.8196 e AUC-ROC = 0.8962, 
#o que indica uma precição com boa chance de acerto (82%) e predição com poucos chutes aleatórios.
