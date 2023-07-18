from sklearn.linear_model import LassoLarsCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import TweedieRegressor
from sklearn import linear_model
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

# 0.318
# 0.314
('EVI_Amplitude_1.nlcd.natural Greenup_1.nlcd.natural', 2.2416843178477945)
('EVI_Amplitude_1.nlcd.natural^2', 1.5777542531080966)
('nlcd.natural.pixel.prop^2', 1.1284810757697668)
('EVI_Area_1.nlcd.natural nlcd.natural.pixel.prop', 1.0197997390828808)
('MidGreendown_1.nlcd.natural^2', 0.953876564710748)
('MidGreendown_1.nlcd.natural', 0.8774614196213635)
('Greenup_1.nlcd.natural^2', 0.65246086968104)
('Dormancy_1.nlcd.natural Greenup_1.nlcd.natural', 0.5064009412323296)
remote_response_variables = [
    'Dormancy_1.nlcd.natural',
    'EVI_Amplitude_1.nlcd.natural',
    'EVI_Area_1.nlcd.natural',
    'Greenup_1.nlcd.natural',
    'MidGreendown_1.nlcd.natural',
    'nlcd.natural.pixel.prop',
    #'Maturity_1.nlcd.natural',
    #'MidGreenup_1.nlcd.natural',
    #'Peak_1.nlcd.natural',
    #'Senescence_1.nlcd.natural',
]

lat_lng_response_variables = [
    #'lat',
    'long',
    #'Dormancy_1.nlcd.natural',
 ]

with_dormancy = remote_response_variables+[
    'Dormancy_1.nlcd.natural',
]

sin_transform_variables = [
    'Dormancy_1.nlcd.natural',
    'Greenup_1.nlcd.natural',
    'Maturity_1.nlcd.natural',
    'MidGreendown_1.nlcd.natural',
    'MidGreenup_1.nlcd.natural',
]

with_long = with_dormancy + lat_lng_response_variables

target_field = 'may_june_total_insects'
df = df[df[target_field] != 0]

# Create a figure and axes for subplots
fig, axs = plt.subplots(2, 3, figsize=(14, 14//3*2))

for fig_index, (field_names, experiment_id) in enumerate([
        (remote_response_variables, 'remote sensed only'),
        (with_dormancy, 'with dormancy'),
        (with_long, 'all+long'),
        #(lat_lng_response_variables, 'long only'),
        ]):
    X = df[field_names].copy()

    for sin_field in sin_transform_variables:
        if sin_field in X:
            print(f'transform {sin_field}')
            X[sin_field] = (numpy.sin(2*numpy.pi*(X[sin_field]+365*.21)/365))**1

    y = df[target_field]

    # # Split the dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    #model = RidgeCV(alphas=[0.1, 1, 10], cv=10)
    #model = LassoLarsCV(positive=True, max_iter=10000, n_jobs=-1)
    #model = linear_model.LinearRegression()
    #model = TweedieRegressor(power=1, alpha=0.1, link='auto')
    model = SVR(kernel='poly', C=1, epsilon=0.5)
    #svr_parameters = {'kernel':('linear', 'poly'), 'C':[1, 10], 'epsilon':[0.1,0.2,0.5,0.3]}
    #model = GridSearchCV(SVR(), svr_parameters, verbose=3, n_jobs=-1)

    # Define parameter grid
    # Define regressor
    # Define grid search
    #model = SVR(kernel=)
    pipeline = Pipeline([
        ('polynomialfeatures', PolynomialFeatures(2)),
        ('scaler', StandardScaler(with_mean=True)),
        ('scaler2', MinMaxScaler()),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print("R^2 score: ", r2)

    # For a linear SVM, the weights of the features can be accessed with model.coef_
    if hasattr(pipeline.named_steps['model'], 'coef_'):
        poly_features = pipeline.named_steps['polynomialfeatures']
        feature_names = poly_features.get_feature_names(input_features=X.columns)

        print(
            "Weights (Coefficients):\n",
            '\n'.join(
                str((v[0].replace(' ', '*'), v[1])) for v in sorted(
                    zip(feature_names, pipeline.named_steps['model'].coef_),
                    key=lambda q: -abs(q[1])) if v[1] != 0))
    #print(pipeline.named_steps['model'].best_params_)

    # Plot the test data
    axs[0, fig_index].scatter(
        y_test, y_pred, s=1, color='blue', label='Predictions')
    # Plot a perfect prediction line
    axs[0, fig_index].plot(
        [min(y_test), max(y_test)], [min(y_test), max(y_test)],
        linewidth=1, color='red', label='Perfect fit')

    axs[0, fig_index].set_title(f'{experiment_id} with R^2={r2:.3f}')
    axs[0, fig_index].set_xlabel('True values')
    axs[0, fig_index].set_ylabel('Predicted values')
    axs[0, fig_index].legend()

    # axs[1, fig_index].scatter(
    #     y, X[field_names[0]], s=1, color='black', label='Comparison')
    # axs[1, fig_index].set_xlabel(field_names[0])


    # # Plot the original data
    # plt.scatter(y_test, y_pred, color='blue', label='Test vs pred')
    # plt.show()
    # Plot the model's predictions
    #plt.plot(X, y_pred, color='red', label='Fit line')

plt.tight_layout()
plt.show()

# Now, you can use X_train, X_test, y_train, y_test in your scikit-learn model.
