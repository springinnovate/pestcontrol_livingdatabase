import numpy
from sklearn.svm import SVR
from sklearn.linear_model import RidgeCV
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Load the CSV data into a pandas DataFrame
df = pd.read_csv('base_data/MiridCACottonNLCDallLim.csv')

model1 = [
    'nlcd.natural.pixel.prop',
    'lat',
    'long',
]

model2 = [
    'nlcd.natural.pixel.prop',
    'lat',
    'long',
    'Dormancy_1.nlcd.natural',
    'EVI_Area_1.nlcd.natural',
    'Senescence_1.nlcd.natural',
]

remote_response_variables = [
    'Dormancy_1.nlcd.natural',
    'EVI_Amplitude_1.nlcd.natural',
    'EVI_Area_1.nlcd.natural',
    'Greenup_1.nlcd.natural',
    'Maturity_1.nlcd.natural',
    'MidGreendown_1.nlcd.natural',
    'MidGreenup_1.nlcd.natural',
    'nlcd.natural.pixel.prop',
    'Peak_1.nlcd.natural',
    'Senescence_1.nlcd.natural',
]

lat_lng_response_variables = [
    'lat',
    'long',
 ]

full_response_variables = remote_response_variables + lat_lng_response_variables

# Create a figure and axes for subplots
fig, axs = plt.subplots(3, figsize=(14//3, 14))

for fig_index, (field_names, experiment_id) in enumerate([
        (remote_response_variables, 'remote sensed only'),
        (model1, 'natural only w/ lat/lng'),
        (model2, '+dorm evi')
        #(lat_lng_response_variables, 'lat/long only'),
        #(full_response_variables, 'remote and lat/lng')
        ]):
    target_field = 'may_june_total_insects'
    X = df[field_names]

    y = df[target_field]

    # # Split the dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RidgeCV(alphas=[0.1], cv=10)

    # Define parameter grid
    #svr_parameters = {'kernel':('linear', 'poly', 'sigmoid'), 'C':[1, 10], 'epsilon':[0.1,0.2,0.5,0.3]}
    # Define regressor
    # Define grid search
    #model = GridSearchCV(SVR(), svr_parameters, verbose=3, n_jobs=-1)
    #model = SVR(kernel=)
    pipeline = Pipeline([
        #('scaler', StandardScaler(with_mean=False)),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print("R^2 score: ", r2)

    # For a linear SVM, the weights of the features can be accessed with model.coef_
    print(
        "Weights (Coefficients): ",
        '\n'.join(
            str(v) for v in sorted(
                zip(field_names, pipeline.named_steps['model'].coef_),
                key=lambda q: -abs(q[1]))))
    #print(pipeline.named_steps['model'].best_params_)

    # Plot the test data
    axs[fig_index].scatter(
        y_test, y_pred, s=1, color='blue', label='Predictions')
    # Plot a perfect prediction line
    axs[fig_index].plot(
        [min(y_test), max(y_test)], [min(y_test), max(y_test)],
        linewidth=1, color='red', label='Perfect fit')

    axs[fig_index].set_title(f'{experiment_id} with R^2={r2:.3f}')
    axs[fig_index].set_xlabel('True values')
    axs[fig_index].set_ylabel('Predicted values')
    axs[fig_index].legend()

    # # Plot the original data
    # plt.scatter(y_test, y_pred, color='blue', label='Test vs pred')
    # plt.show()
    # Plot the model's predictions
    #plt.plot(X, y_pred, color='red', label='Fit line')

plt.tight_layout()
plt.show()

# Now, you can use X_train, X_test, y_train, y_test in your scikit-learn model.
