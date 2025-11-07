import pandas as pd
import numpy as np
import pickle
import streamlit as st

#Import models
'''
with open('../models/standard_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('../models/grid_boost.pkl', 'rb') as file:
    grid = pickle.load(file)

columns = ['customer_type', 'fabricant', 'follow_up_needed', 'make_and_model', 'mileage', 'parts_used_normalized', 'parts_used', 'repair_date', 'service_description', 'service_status', 'service_type', 'urgency_level', 'vehicle_type']

encoders = {}

for column in columns:
    with open(f'../models/label_encoder_{column}.pkl', 'rb') as file:
        encoders[f'{column}'] = pickle.load(file)
'''
#Streamlit

st.title('Predictor precio taller')
st.text('Este modelo predice precios de reparacion de vehiculos motorizados a partir de ciertos parametros.')

#Entries

col1, col2, col3, col4 = st.columns(4)

cat = ['vehicle_type', 'make_and_model', 'service_type', 'service_description', 'repair_date', 'parts_used',
       'service_status', 'follow_up_needed', 'urgency_level', 'customer_type', 'service_duration_hours',
       'mileage_at_service', 'tow_distance_miles', 'fabricant', 'parts_used_normalized', 'mileage_bin']

options = [(x, x + 1) for x in range(16)]

with col1: 
    st.selectbox(cat[0], ['Van', 'Car', 'Bus', 'Motorcycle', 'SUV', 'Truck'])
    st.selectbox(cat[2], options[2])
    st.selectbox(cat[5], options[5])
    st.selectbox(cat[7], options[7])

with col2:
    st.selectbox(cat[1], ['Harley-Davidson Odyssey',
 'Ford Accord',
 'Honda Coach',
 'Harley-Davidson Road King',
 'Harley-Davidson Civic',
 'Honda Camry',
 'Honda Pilot',
 'Yamaha Transit',
 'Harley-Davidson School Bus',
 'Dodge Silverado',
 'Honda Transit',
 'Chevrolet School Bus',
 'Harley-Davidson RAV4',
 'Chevrolet Transit',
 'BMW Road King',
 'Harley-Davidson Tacoma',
 'Chevrolet Silverado',
 'Toyota Explorer',
 'Ford Sportster',
 'Ford RAV4',
 'Yamaha Civic',
 'Ford Tacoma',
 'BMW Ram 1500',
 'Chevrolet CR-V',
 'Yamaha Sportster',
 'Yamaha Sienna',
 'Toyota Transit',
 'BMW CR-V',
 'Honda Accord',
 'Yamaha Pilot',
 'Honda Silverado',
 'Yamaha Accord',
 'Yamaha School Bus',
 'Yamaha CR-V',
 'Honda YZF-R6',
 'BMW Mustang',
 'Yamaha Coach',
 'Harley-Davidson Mustang',
 'Dodge RAV4',
 'Ford Silverado',
 'Yamaha Odyssey',
 'Chevrolet Sienna',
 'Yamaha Silverado',
 'Dodge Transit',
 'BMW School Bus',
 'BMW Camry',
 'Honda Sportster',
 'Chevrolet RAV4',
 'Chevrolet Tacoma',
 'Ford Coach',
 'Yamaha Tacoma',
 'Toyota Corolla',
 'Toyota Accord',
 'Honda Sienna',
 'Dodge Road King',
 'Yamaha RAV4',
 'Chevrolet Coach',
 'Yamaha Road King',
 'Yamaha F-150',
 'Ford Explorer',
 'Dodge Pilot',
 'BMW Pilot',
 'Ford CR-V',
 'Chevrolet F-150',
 'BMW Explorer',
 'BMW Odyssey',
 'Toyota F-150',
 'Dodge Ram 1500',
 'Dodge Sportster',
 'Dodge Odyssey',
 'Dodge School Bus',
 'Harley-Davidson Coach',
 'Chevrolet YZF-R6',
 'Toyota Sportster',
 'Dodge YZF-R6',
 'Harley-Davidson Sportster',
 'Ford Road King',
 'Ford Transit',
 'Toyota Ninja ZX-14R',
 'BMW Accord',
 'Ford F-150',
 'Dodge Coach',
 'BMW Ninja ZX-14R',
 'Harley-Davidson Ram 1500',
 'Ford Sienna',
 'Dodge Ninja ZX-14R',
 'Toyota Ram 1500',
 'Ford Odyssey',
 'Chevrolet Ninja ZX-14R',
 'BMW F-150',
 'Toyota Road King',
 'Dodge Corolla',
 'Yamaha Ninja ZX-14R',
 'Harley-Davidson Transit',
 'Dodge Tacoma',
 'Toyota Sienna',
 'Dodge Sienna',
 'Chevrolet Odyssey',
 'Honda Corolla',
 'Harley-Davidson F-150',
 'Harley-Davidson Pilot',
 'Yamaha Camry',
 'Ford Civic',
 'Harley-Davidson Explorer',
 'Toyota Odyssey',
 'Harley-Davidson Silverado',
 'Honda Ram 1500',
 'Chevrolet Explorer',
 'Chevrolet Road King',
 'Chevrolet Sportster',
 'Dodge CR-V',
 'Honda Mustang',
 'Dodge Explorer',
 'Honda Civic',
 'Ford Ninja ZX-14R',
 'Dodge Civic',
 'Ford Corolla',
 'Chevrolet Pilot',
 'Honda School Bus',
 'Toyota RAV4',
 'Toyota School Bus',
 'Harley-Davidson YZF-R6',
 'Dodge Camry',
 'Ford Camry',
 'Chevrolet Ram 1500',
 'BMW Coach',
 'Toyota Tacoma',
 'BMW Sienna',
 'Ford School Bus',
 'Toyota Pilot',
 'Harley-Davidson CR-V',
 'Toyota Silverado',
 'Toyota Coach',
 'Toyota Mustang',
 'Harley-Davidson Accord',
 'Harley-Davidson Sienna',
 'Honda Tacoma',
 'Yamaha YZF-R6',
 'Ford Pilot',
 'BMW Civic',
 'Yamaha Explorer',
 'Honda F-150',
 'Honda RAV4',
 'Honda CR-V',
 'BMW RAV4',
 'BMW Transit',
 'BMW Silverado',
 'Toyota Civic',
 'Chevrolet Civic',
 'Toyota CR-V',
 'Ford Mustang',
 'Toyota YZF-R6',
 'Dodge F-150',
 'Harley-Davidson Corolla',
 'Dodge Mustang',
 'BMW Tacoma',
 'Yamaha Ram 1500',
 'Toyota Camry',
 'Honda Odyssey',
 'Honda Explorer',
 'Chevrolet Mustang',
 'Yamaha Corolla',
 'Ford Ram 1500',
 'Honda Road King',
 'BMW YZF-R6',
 'Harley-Davidson Ninja ZX-14R',
 'Harley-Davidson Camry',
 'Chevrolet Camry',
 'BMW Corolla',
 'Chevrolet Accord',
 'Honda Ninja ZX-14R',
 'BMW Sportster',
 'Chevrolet Corolla',
 'Ford YZF-R6',
 'Yamaha Mustang',
 'Dodge Accord'])
    st.selectbox(cat[3], options[3])
    st.selectbox(cat[14], options[14])
    st.number_input(cat[11], options[11][0], options[11][1])

with col3:
    st.selectbox(cat[9], options[9])
    st.selectbox(cat[6], options[6])
    st.selectbox(cat[8], options[8])
    st.number_input(cat[12], options[12][0], options[12][1])

with col4: 
    st.selectbox(cat[13], options[13])
    st.number_input(cat[10], options[10][0], options[10][1])
    st.date_input(cat[4])
    st.selectbox(cat[15], options[15])

st.button('Predice el precio con los parametros seleccionados')


#Data processing (encoder & scaling)



#Predict

