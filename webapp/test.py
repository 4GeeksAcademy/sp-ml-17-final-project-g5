import pandas as pd
import numpy as np
import pickle
import streamlit as st
import datetime
from pathlib import Path

# base del proyecto (dos niveles arriba de este archivo)
BASE = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE / "models"

def load_pickle(filename):
    path = MODELS_DIR / filename
    if not path.exists():
        st.error(f"Fichero no encontrado: {path}")
        st.stop()
    with open(path, "rb") as f:
        return pickle.load(f)

# Import models (usar rutas relativas con pathlib)
scaler = load_pickle("standard_scaler.pkl")
grid = load_pickle("grid_boost.pkl")

columns = ['customer_type', 'fabricant', 'follow_up_needed', 'make_and_model', 'mileage_bin', 'parts_used_normalized', 'parts_used', 'repair_date', 'service_description', 'service_status', 'service_type', 'urgency_level', 'vehicle_type']

encoders = {}
for column in columns:
    encoders[column] = load_pickle(f"label_encoder_{column}.pkl")

# Streamlit
st.title('Predictor precio taller')
st.text('Este modelo predice precios de reparacion de vehiculos motorizados a partir de ciertos parametros.')

# Entries
entrada = {}  # usar dict y luego convertir a DataFrame

col1, col2, col3, col4 = st.columns(4)

cat = ['vehicle_type', 'make_and_model', 'service_type', 'service_description', 'repair_date', 'parts_used',
       'service_status', 'follow_up_needed', 'urgency_level', 'customer_type', 'service_duration_hours',
       'mileage_at_service', 'tow_distance_miles', 'fabricant', 'parts_used_normalized', 'mileage_bin']

options = [['Van', 'Car', 'Bus', 'Motorcycle', 'SUV', 'Truck'], ['Harley-Davidson Odyssey',
 'Ford Accord', 'Honda Coach', 'Harley-Davidson Road King', 'Harley-Davidson Civic',
 'Honda Camry', 'Honda Pilot', 'Yamaha Transit', 'Harley-Davidson School Bus',
 'Dodge Silverado', 'Honda Transit', 'Chevrolet School Bus', 'Harley-Davidson RAV4',
 'Chevrolet Transit', 'BMW Road King', 'Harley-Davidson Tacoma', 'Chevrolet Silverado',
 'Toyota Explorer', 'Ford Sportster', 'Ford RAV4', 'Yamaha Civic', 'Ford Tacoma', 'BMW Ram 1500',
 'Chevrolet CR-V', 'Yamaha Sportster', 'Yamaha Sienna', 'Toyota Transit', 'BMW CR-V',
 'Honda Accord', 'Yamaha Pilot', 'Honda Silverado', 'Yamaha Accord', 'Yamaha School Bus',
 'Yamaha CR-V', 'Honda YZF-R6', 'BMW Mustang', 'Yamaha Coach', 'Harley-Davidson Mustang',
 'Dodge RAV4', 'Ford Silverado', 'Yamaha Odyssey', 'Chevrolet Sienna', 'Yamaha Silverado',
 'Dodge Transit', 'BMW School Bus', 'BMW Camry', 'Honda Sportster', 'Chevrolet RAV4',
 'Chevrolet Tacoma', 'Ford Coach', 'Yamaha Tacoma', 'Toyota Corolla', 'Toyota Accord',
 'Honda Sienna', 'Dodge Road King', 'Yamaha RAV4', 'Chevrolet Coach', 'Yamaha Road King',
 'Yamaha F-150', 'Ford Explorer', 'Dodge Pilot', 'BMW Pilot', 'Ford CR-V', 'Chevrolet F-150',
 'BMW Explorer', 'BMW Odyssey', 'Toyota F-150', 'Dodge Ram 1500', 'Dodge Sportster', 'Dodge Odyssey',
 'Dodge School Bus', 'Harley-Davidson Coach', 'Chevrolet YZF-R6', 'Toyota Sportster', 'Dodge YZF-R6',
 'Harley-Davidson Sportster', 'Ford Road King', 'Ford Transit', 'Toyota Ninja ZX-14R',
 'BMW Accord', 'Ford F-150', 'Dodge Coach', 'BMW Ninja ZX-14R', 'Harley-Davidson Ram 1500',
 'Ford Sienna', 'Dodge Ninja ZX-14R', 'Toyota Ram 1500', 'Ford Odyssey', 'Chevrolet Ninja ZX-14R',
 'BMW F-150', 'Toyota Road King', 'Dodge Corolla', 'Yamaha Ninja ZX-14R', 'Harley-Davidson Transit',
 'Dodge Tacoma', 'Toyota Sienna', 'Dodge Sienna', 'Chevrolet Odyssey', 'Honda Corolla',
 'Harley-Davidson F-150', 'Harley-Davidson Pilot', 'Yamaha Camry', 'Ford Civic', 'Harley-Davidson Explorer',
 'Toyota Odyssey', 'Harley-Davidson Silverado', 'Honda Ram 1500', 'Chevrolet Explorer', 'Chevrolet Road King',
 'Chevrolet Sportster', 'Dodge CR-V', 'Honda Mustang', 'Dodge Explorer', 'Honda Civic', 'Ford Ninja ZX-14R',
 'Dodge Civic', 'Ford Corolla', 'Chevrolet Pilot', 'Honda School Bus', 'Toyota RAV4', 'Toyota School Bus',
 'Harley-Davidson YZF-R6', 'Dodge Camry', 'Ford Camry', 'Chevrolet Ram 1500', 'BMW Coach',
 'Toyota Tacoma', 'BMW Sienna', 'Ford School Bus', 'Toyota Pilot', 'Harley-Davidson CR-V',
 'Toyota Silverado', 'Toyota Coach', 'Toyota Mustang', 'Harley-Davidson Accord', 'Harley-Davidson Sienna',
 'Honda Tacoma', 'Yamaha YZF-R6', 'Ford Pilot', 'BMW Civic', 'Yamaha Explorer', 'Honda F-150',
 'Honda RAV4', 'Honda CR-V', 'BMW RAV4', 'BMW Transit', 'BMW Silverado', 'Toyota Civic',
 'Chevrolet Civic', 'Toyota CR-V', 'Ford Mustang', 'Toyota YZF-R6', 'Dodge F-150', 'Harley-Davidson Corolla',
 'Dodge Mustang', 'BMW Tacoma', 'Yamaha Ram 1500', 'Toyota Camry', 'Honda Odyssey', 'Honda Explorer',
 'Chevrolet Mustang', 'Yamaha Corolla', 'Ford Ram 1500', 'Honda Road King', 'BMW YZF-R6', 'Harley-Davidson Ninja ZX-14R',
 'Harley-Davidson Camry', 'Chevrolet Camry', 'BMW Corolla', 'Chevrolet Accord', 'Honda Ninja ZX-14R',
 'BMW Sportster', 'Chevrolet Corolla', 'Ford YZF-R6', 'Yamaha Mustang', 'Dodge Accord'], ['Towing',
 'Engine Repair', 'Tire Replacement', 'Oil Change', 'Brake Service', 'Transmission Repair', 'Battery Replacement'],
['no parts used.', 'Engine Mount.', 'Tire.', 'TPMS Sensor.', 'Oil Drain Plug.', 'Rotors.', 'Timing Belt.',
 'Brake Fluid.', 'Clutch Kit.', 'Gearbox Seal.', 'Battery Terminal.', 'Calipers.', 'Synthetic Oil.',
 'Battery Holder.', 'Oil Pump.', 'Brake Pads.', 'Valve Stem.', 'Battery.', 'Transmission Fluid.',
 'Oil Filter.', 'Spark Plugs.', 'Rim.'], 0, ['No parts used', 'Engine Mount, Oil Pump, Timing Belt', 'Valve Stem, TPMS Sensor, Tire',
 'Valve Stem, Tire, TPMS Sensor', 'Oil Filter, Synthetic Oil, Oil Drain Plug', 'Brake Fluid, Rotors, Calipers',
 'Rotors, Brake Pads, Brake Fluid', 'Oil Pump, Timing Belt, Spark Plugs', 'Oil Drain Plug, Oil Filter, Synthetic Oil',
 'Calipers, Brake Fluid, Rotors', 'Rotors, Brake Fluid, Calipers', 'Clutch Kit, Gearbox Seal, Transmission Fluid',
 'Gearbox Seal, Transmission Fluid, Clutch Kit', 'Transmission Fluid, Clutch Kit, Gearbox Seal', 'Calipers, Rotors, Brake Pads',
 'Spark Plugs, Oil Pump, Timing Belt', 'Battery, Battery Holder, Battery Terminal', 'Rim, Tire, Valve Stem',
 'Valve Stem, Rim, Tire', 'Brake Pads, Calipers, Rotors', 'Synthetic Oil, Oil Filter, Oil Drain Plug',
 'Battery Holder, Battery Terminal, Battery', 'Rotors, Brake Pads, Calipers', 'Rim, TPMS Sensor, Valve Stem',
 'Battery Terminal, Battery Holder, Battery', 'Rim, TPMS Sensor, Tire', 'Synthetic Oil, Oil Drain Plug, Oil Filter',
 'Oil Drain Plug, Synthetic Oil, Oil Filter', 'Rim, Tire, TPMS Sensor', 'Rim, Valve Stem, Tire', 'Oil Pump, Timing Belt, Engine Mount',
 'Timing Belt, Oil Pump, Engine Mount', 'Tire, Valve Stem, Rim', 'Engine Mount, Timing Belt, Oil Pump', 'Rotors, Calipers, Brake Fluid',
 'TPMS Sensor, Rim, Tire', 'Battery Terminal, Battery, Battery Holder', 'Timing Belt, Engine Mount, Oil Pump', 'Spark Plugs, Engine Mount, Oil Pump',
 'Battery, Battery Terminal, Battery Holder', 'Tire, Rim, TPMS Sensor', 'Timing Belt, Spark Plugs, Oil Pump', 'Gearbox Seal, Clutch Kit, Transmission Fluid',
 'Transmission Fluid, Gearbox Seal, Clutch Kit', 'Tire, TPMS Sensor, Valve Stem', 'Brake Pads, Brake Fluid, Rotors',
 'Brake Pads, Calipers, Brake Fluid', 'Rotors, Brake Fluid, Brake Pads', 'Battery Holder, Battery, Battery Terminal',
 'Engine Mount, Spark Plugs, Timing Belt', 'Tire, Valve Stem, TPMS Sensor', 'Oil Filter, Oil Drain Plug, Synthetic Oil',
 'Brake Fluid, Brake Pads, Calipers', 'Spark Plugs, Timing Belt, Engine Mount', 'Calipers, Brake Fluid, Brake Pads',
 'Oil Pump, Engine Mount, Timing Belt', 'Rim, Valve Stem, TPMS Sensor', 'TPMS Sensor, Tire, Valve Stem', 'Brake Pads, Rotors, Calipers',
 'Brake Fluid, Brake Pads, Rotors', 'Valve Stem, TPMS Sensor, Rim', 'Brake Fluid, Calipers, Brake Pads', 'Engine Mount, Oil Pump, Spark Plugs',
 'TPMS Sensor, Valve Stem, Tire', 'Timing Belt, Spark Plugs, Engine Mount', 'Calipers, Brake Pads, Rotors', 'Oil Pump, Spark Plugs, Engine Mount',
 'Clutch Kit, Transmission Fluid, Gearbox Seal', 'Brake Fluid, Calipers, Rotors', 'Brake Pads, Brake Fluid, Calipers',
 'Calipers, Brake Pads, Brake Fluid', 'Timing Belt, Engine Mount, Spark Plugs', 'Spark Plugs, Oil Pump, Engine Mount',
 'TPMS Sensor, Valve Stem, Rim', 'Valve Stem, Tire, Rim', 'Spark Plugs, Timing Belt, Oil Pump', 'Calipers, Rotors, Brake Fluid',
 'Brake Pads, Rotors, Brake Fluid', 'Valve Stem, Rim, TPMS Sensor', 'Engine Mount, Spark Plugs, Oil Pump', 'Rotors, Calipers, Brake Pads',
 'Brake Fluid, Rotors, Brake Pads', 'Spark Plugs, Engine Mount, Timing Belt', 'Timing Belt, Oil Pump, Spark Plugs',
 'TPMS Sensor, Tire, Rim', 'Oil Pump, Engine Mount, Spark Plugs', 'Engine Mount, Timing Belt, Spark Plugs',
 'Tire, TPMS Sensor, Rim', 'Oil Pump, Spark Plugs, Timing Belt', 'Tire, Rim, Valve Stem', 'TPMS Sensor, Rim, Valve Stem'],
 ['s', 'In Progress'], ['No', 'Yes'], ['Standard', 'Emergency', 'Scheduled'], ['Individual', 'Business', 'Fleet'], 0, 0, 0,
 ['Harley-Davidson', 'Ford', 'Honda', 'Yamaha', 'Dodge', 'Chevrolet', 'BMW', 'Toyota'], ['No parts used',
 'Engine Mount, Oil Pump, Timing Belt', 'TPMS Sensor, Tire, Valve Stem', 'Oil Drain Plug, Oil Filter, Synthetic Oil',
 'Brake Fluid, Calipers, Rotors', 'Brake Fluid, Brake Pads, Rotors', 'Oil Pump, Spark Plugs, Timing Belt',
 'Clutch Kit, Gearbox Seal, Transmission Fluid', 'Brake Pads, Calipers, Rotors', 'Battery, Battery Holder, Battery Terminal',
 'Rim, Tire, Valve Stem', 'Rim, TPMS Sensor, Valve Stem', 'Rim, TPMS Sensor, Tire', 'Engine Mount, Oil Pump, Spark Plugs',
 'Brake Fluid, Brake Pads, Calipers', 'Engine Mount, Spark Plugs, Timing Belt'], ['200k-300k', '0-50k', '50k-100k', '100k-200k']]

with col1: 
    entrada[f'{cat[0]}'] = st.selectbox(cat[0], options[0])
    entrada[f'{cat[2]}'] = st.selectbox(cat[2], options[2])
    entrada[f'{cat[5]}'] = st.selectbox(cat[5], options[5])
    entrada[f'{cat[7]}'] = st.selectbox(cat[7], options[7])

with col2:
    entrada[f'{cat[1]}'] = st.selectbox(cat[1], options[1])
    entrada[f'{cat[3]}'] = st.selectbox(cat[3], options[3])
    entrada[f'{cat[14]}'] = st.selectbox(cat[14], options[14])
    entrada[f'{cat[11]}'] = st.number_input(cat[11], 10000, 300000, step=1000)

with col3:
    entrada[f'{cat[9]}'] = st.selectbox(cat[9], options[9])
    entrada[f'{cat[6]}'] = st.selectbox(cat[6], options[6])
    entrada[f'{cat[8]}'] = st.selectbox(cat[8], options[8])
    entrada[f'{cat[12]}'] = st.number_input(cat[12], 0.0, 100.0, step=0.1)

with col4: 
    entrada[f'{cat[13]}'] = st.selectbox(cat[13], options[13])
    entrada[f'{cat[10]}'] = st.number_input(cat[10], 0.5, 10.1, step=0.5)
    entrada[f'{cat[4]}'] = str(st.date_input(cat[4], min_value=datetime.date(2020, 1, 1), 
                                         max_value=datetime.date(2024, 12, 31), format='YYYY-MM-DD'))
    entrada[f'{cat[15]}'] = st.selectbox(cat[15], options[15])

# Data processing (encoder & scaling)
entrada = pd.DataFrame([entrada])  # una fila

for column in columns:
    # asegurarse de pasar array 1D al encoder
    entrada[f'{column}_n'] = encoders[column].transform(entrada[column].astype(str).values)
    entrada.drop(column, axis=1, inplace=True)

# reordenar columnas según espera el scaler/modelo
entrada = entrada[['service_duration_hours', 'mileage_at_service', 'tow_distance_miles', 'vehicle_type_n', 'make_and_model_n',
       'service_type_n', 'service_description_n', 'repair_date_n', 'parts_used_n', 'service_status_n',
       'follow_up_needed_n', 'urgency_level_n', 'customer_type_n', 'fabricant_n', 'parts_used_normalized_n',
       'mileage_bin_n']]

entrada = scaler.transform(entrada)

# Predict
if st.button('Predice el precio con los parametros seleccionados'):
    precio = float(grid.predict(entrada)[0])
    st.text(f'La predicción del precio es {round(precio, 2)}€')

