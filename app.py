# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 10:34:10 2023
@author: pratiksanghvi
"""
import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
from collections.abc import Mapping, MutableMapping
import bisect

app = Flask(__name__)

#usedcar = pd.read_csv("D:\\Learn\\CarDekho_SPPrediction\\used_car_price_cleaned.csv")



global_make_data_html = ['Jeep', 'Renault', 'Toyota', 'Honda', 'Volkswagen', 'Maruti',
       'Mahindra', 'Hyundai', 'Nissan', 'Kia', 'MG', 'Tata', 'BMW',
       'Mercedes-Benz', 'Datsun', 'Volvo', 'Audi', 'Porsche', 'Ford',
       'Chevrolet', 'Skoda', 'Lexus', 'Land', 'Mini', 'Jaguar',
       'Mitsubishi', 'Force', 'Premier', 'Fiat', 'Maserati', 'Bentley',
       'Isuzu']

global_seats_data_html = ['5 Seats', '6 Seats', '7 Seats', '4 Seats', '8 Seats', '2 Seats']
global_year_data_html = [2017, 2021, 2016, 2018, 2015, 2014, 2020, 2019, 2012, 2008, 2013,
                        2022, 2010, 2005, 2009, 2006, 2011, 2007, 2002, 2004, 1998, 2003,
                        1995, 2000, 2001, 1999]
min_global_enginecc_number_html = 0
max_global_enginecc_number_html = 9999
global_ownership_data_html = ['1st Owner', '2nd Owner', '3rd Owner', '4th Owner', '5th Owner','0th Owner']
global_transmission_data_html = ['Manual', 'Automatic']
global_fuelType_data_html = ['Diesel', 'Petrol', 'Cng', 'Electric', 'Lpg']
min_global_kmdriven_number_html = 0
max_global_kmdriven_number_html = 99000000


# -----------------------------------------------------------------------------------

@app.route('/', methods=["GET", "POST"])
def usedCarPredictionRenderTemplate():
    if request.method == "POST":
        user_data_transform_for_prediction()

    return render_template("car_index.html", car_make_data=global_make_data_html,
                           seats_data=global_seats_data_html,
                           year_data=global_year_data_html,
                           ownership_data=global_ownership_data_html,
                           transmission_data=global_transmission_data_html,
                           fuel_type_data=global_fuelType_data_html)


# Loading Pickle files-----------------------------------------------------
pickle_in = open("D:\\Learn\\CarDekho_SPPrediction\\rf_regressor.pkl", "rb")
rf_regressor = pickle.load(pickle_in)

pickle_in_knn = open("D:\\Learn\\CarDekho_SPPrediction\\knn_regressor.pkl", "rb")
knn_regressor = pickle.load(pickle_in_knn)

pickle_in_lasso = open("D:\\Learn\\CarDekho_SPPrediction\\lasso_regressor.pkl", "rb")
lasso_regressor = pickle.load(pickle_in_lasso)

pickle_in_linear = open("D:\\Learn\\CarDekho_SPPrediction\\lr_regressor.pkl", "rb")
linear_regressor = pickle.load(pickle_in_linear)


# -----------------------------------------------------------------------------

@app.route('/predict', methods=["GET", "POST"])
def user_data_transform_for_prediction():
    car_make = str(request.form.get('car_make'))
    seats = str(request.form.get('seats'))
    year = str(request.form.get('year'))
    ownership = str(request.form.get('ownership'))
    transmission = str(request.form.get('transmission'))
    # user_fuel_type = str(request.form.get('fuel_type_data'))
    features = []
    for x in request.form.values():
        features.append(x)
    engine = int(features[6])
    kms_driven = int(features[7])
    user_fuel_type = str(features[5])

    # --- Transform the incoming data-----------------------------------
    dummy_car_make_user = return_index_from_list(global_make_data_html, car_make.lower())
    # -------------------------------------------------------------------
    temp_seats = seats.replace(" Seats", "")
    dummy_seats = int(temp_seats)
    # -----------------------------------------------------------------------
    dummy_year = int(year)
    # --------------------------------------------------------------------------
    dummy_ownership = return_index_from_list(global_ownership_data_html, ownership.lower())
    # ---------------------------------------------------------------------------
    dummy_transmission = return_index_from_list(global_transmission_data_html, transmission.lower())
    # ---------------------------------------------------------------------------
    dummy_fuel_type = return_index_from_list(global_fuelType_data_html, user_fuel_type.lower())
    # ---------------------------------------------------------------------------
    dummy_engine = int(engine)
    # ---------------------------------------------------------------------------
    dummy_kms_driven = int(kms_driven)
    # --------------------------------------------------------------------------

    # Prepare the dataset for the model to predict
    df = pd.DataFrame({"Make": dummy_car_make_user, "Seats": dummy_seats, "manufacture": dummy_year,
                       "ownership": dummy_ownership, "transmission": dummy_transmission,
                       "fuel_type": dummy_fuel_type, "engine_in_cc": dummy_engine,
                       "kms_driven": dummy_kms_driven}, index=[0])

    if request.form['action'] == 'Predict_RF':
        prediction = rf_regressor.predict(df)
        print(prediction)
        # x = np.ascontiguousarray(df_usedcar, dtype=int)
        # prediction = rf_regressor.predict(x)

    if request.form['action'] == 'Predict_KNN':
        prediction = knn_regressor.predict(df)
        print(prediction)
        # x = np.ascontiguousarray(df_usedcar, dtype=int)
        # prediction = knn_regressor.predict(x)

    if request.form['action'] == 'Predict_LR':
        prediction = linear_regressor.predict(df)
        print(prediction)
        # x = np.ascontiguousarray(df_usedcar, dtype=int)
        # prediction = linear_regressor.predict(x)

    if request.form['action'] == 'Predict_Lasso':
        prediction = lasso_regressor.predict(df)
        print(prediction)
        # x = np.ascontiguousarray(df_usedcar, dtype=int)
        # prediction = lasso_regressor.predict(x)

    return render_template('car_index.html', car_make_data=global_make_data_html,
                           seats_data=global_seats_data_html,
                           year_data=global_year_data_html,
                           ownership_data=global_ownership_data_html,
                           transmission_data=global_transmission_data_html,
                           fuel_type_data=global_fuelType_data_html,
                           prediction_text='Used {} with {} , Manufacture year {} and with {}\'s ownership'
                                           '{} transmission and {} engine having engine in cc - {}  has the Predicted Price with accuracy ~30% '
                                           'to be  Rs {:,.2f}'.format(car_make, seats, year, ownership, transmission,
                                                                      user_fuel_type,
                                                                      engine, round(prediction[0], 2)))


def return_index_from_list(list_name, value_to_get_index):
    temp_list = [x.lower() for x in list_name]
    myDict = dict((e, i) for i, e in enumerate(temp_list))
    return myDict[value_to_get_index]


if __name__ == '__main__':
    # app.run(host='192.168.1.100', port=8000, debug=True)# - run on local machine
    app.run(debug=True, use_reloader=False)