import sys
import pandas as pd
import xgboost as xgb

from src.data import load_data_train_raw, load_data_test_raw, save_submission, pre_process_data

args = sys.argv

if len(args) != 3:
    print('-------------')
    print('Usage:')
    print('run.py data_directory_path prediction_file_to_save_path')
    exit(1)

data_directory_path = args[1]
prediction_file_to_save_path = args[2]

# Load train data
print("\nReading training data")
df_train_x, df_train_y, df_train_y_unique = load_data_train_raw(data_directory_path)

print("Extracting training features")
df_train_x_new = pre_process_data(df_train_x)
cols_to_use = df_train_x_new.columns

# Recalculate the TrackNumbers
df_train_y_unique_new = df_train_x_new.reset_index()[['TrackNumber_NEW']]
df_train_y_unique_new['TrackNumber'] = df_train_y_unique_new['TrackNumber_NEW'].apply(lambda x: x % 1000000000)
df_train_y_unique_new_good = df_train_y_unique_new.join(df_train_y_unique.set_index('TrackNumber'), on='TrackNumber').drop('TrackNumber', axis=1)

# Train & save the model
print("Training the model")
model = xgb.XGBClassifier(max_depth=4, colsample_bytree=.05, subsample=.9, n_estimators=500, learning_rate=.075)
model.fit(df_train_x_new.fillna(0)[cols_to_use], df_train_y_unique_new_good['Type'])

# Load test data
print("Reading testing data")
df_test_x, df_test_y, df_test_y_unique = load_data_test_raw(data_directory_path)

# Calculate test features
df_test_x_new = pre_process_data(df_test_x)

df_test_y_unique_new = df_test_x_new.reset_index()[['TrackNumber_NEW']]
df_test_y_unique_new['TrackNumber'] = df_test_y_unique_new['TrackNumber_NEW'].apply(lambda x: x % 1000000000)
df_test_y_unique_new_full = df_test_y_unique_new.join(df_test_y_unique.set_index('TrackNumber'), on='TrackNumber')

# Predict
print("Calculating predictions")
df_test_preds = pd.DataFrame(model.predict_proba(df_test_x_new.fillna(0)[cols_to_use]), index=df_test_x_new.index, columns=model.classes_)
df_test_preds['TrackNumber'] = df_test_y_unique_new_full.set_index('TrackNumber_NEW')['TrackNumber']

# Aggregate back by TrackNumber
df_test_preds_agg = df_test_preds.groupby('TrackNumber').median()

# Save predictions
save_submission(df_test_preds_agg, df_test_y_unique, prediction_file_to_save_path)
