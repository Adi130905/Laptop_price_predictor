import streamlit as st
import pickle
import numpy as np

# Load the model and the dataframe
# Ensure these files are in the same folder as app4.py
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.set_page_config(page_title="Laptop Price Predictor", layout="wide")

st.title("💻 Laptop Price Predictor")
st.markdown("Enter the laptop specifications to estimate the price.")

# Use columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox('Brand', df['Company'].unique())
    type = st.selectbox('Type', df['TypeName'].unique())
    ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
    weight = st.number_input('Weight of the Laptop (kg)', format="%.2f")
    cpu = st.selectbox('CPU Brand', df['Cpu Brand'].unique())
    gpu = st.selectbox('GPU Brand', df['Gpu Brand'].unique())

with col2:
    touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
    ips = st.selectbox('IPS Panel', ['No', 'Yes'])
    screen_size = st.number_input('Screen Size (Inches)', value=15.6)
    resolution = st.selectbox('Screen Resolution', [
        '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', 
        '2880x1800', '2560x1600', '2560x1440', '2304x1440'
    ])
    hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
    ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
    os = st.selectbox('Operating System', df['os'].unique())

if st.button('Estimate Price'):
    # Preprocessing categorical inputs to numeric
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    # Calculate PPI (Pixels Per Inch)
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size
    
    # Create the input array
    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os], dtype=object)
    query = query.reshape(1, 12)

    # Make prediction (Assuming model was trained on log-transformed price)
    prediction = np.exp(pipe.predict(query)[0])
    
    st.success(f"### The estimated price for this configuration is ₹{int(prediction):,}")