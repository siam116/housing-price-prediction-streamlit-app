import streamlit as st
import pandas as pd

from joblib import load
from utils import CombinesAttrAdder, DataFrameSelector

st.title('California Housing Price Prediction')

st.sidebar.markdown("## Predict Housing Price")
st.sidebar.markdown('## User Input')


class HousingPricePredictor:
    def __init__(self, imported_model, imported_pipeline):
        self.model = imported_model
        self.pipeline = imported_pipeline

    def predict(self, X):
        X_prepared = self.pipeline.transform(X)
        prediction = self.model.predict(X_prepared)
        return prediction[0]


def user_input_features():
    longitude = st.sidebar.text_input("Longitude", "-118.39", key='b1').strip()
    latitude = st.sidebar.text_input("Latitude", "33.94", key='v2').strip()
    housing_median_age = st.sidebar.text_input("Housing Median Age", "40", key='n3').strip()
    median_income = st.sidebar.text_input("Median Income", "5.056", key='b4').strip()
    total_rooms = st.sidebar.text_input("Total Rooms", "1789", key='n5').strip()
    total_bedrooms = st.sidebar.text_input("Total Bedrooms", "700", key='b6').strip()
    population = st.sidebar.text_input("Population", "2234", key='n7').strip()
    households = st.sidebar.text_input("Households", "899", key='b8').strip()

    data = {'longitude': [float(longitude)],
            'latitude': [float(latitude)],
            'housingMedianAge': [float(housing_median_age)],
            'totalRooms': [float(total_rooms)],
            'totalBedrooms': [float(total_bedrooms)],
            'population': [float(population)],
            'households': [float(households)],
            'medianIncome': [float(median_income)]
            }

    features = pd.DataFrame(data)
    return features


input_df = user_input_features()

imported_model = load('rf_final_model.joblib')
imported_pipeline = load('pipeline.joblib')


def handle_click():
    st.session_state.Longitude = False
    st.session_state.Latitude = False
    st.session_state.HousingMedianAge = False
    st.session_state.MedianIncome = False
    st.session_state.TotalRooms = False
    st.session_state.TotalBedrooms = False
    st.session_state.Population = False
    st.session_state.Households = False

    housing_price_predictor = HousingPricePredictor(imported_model, imported_pipeline)
    test_pred = housing_price_predictor.predict(input_df)
    st.header("Result")
    st.write('Predicted Housing Price: ${:,.2f}'.format(test_pred))
    st.write("User Input Features")
    st.dataframe(input_df)


st.sidebar.button('Predict', on_click=handle_click)