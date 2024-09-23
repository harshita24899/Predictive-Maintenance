import streamlit as st
from snowflake.snowpark.context import get_active_session
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from snowflake.snowpark import Session
from snowflake.ml.registry import Registry
import pandas as pd



#################################################################################################
#                                      SETTING UP THE SESSION                                   #
#################################################################################################

# Get the Current credentials
session = get_active_session()

#################################################################################################
#                                      MODEL                                                    #
#################################################################################################

# Class to interact with the Snowflake registry and predict Operational status
class ManufacturingPredictor:
    def __init__(self, session, registry_db, registry_schema, model_name):
        self.session = session
        self.registry_db = registry_db
        self.registry_schema = registry_schema
        self.model_name = model_name
        
    
    # Retrieve the machine learning model from the Snowflake model registry
    def get_model(self):
        try:
            reg = Registry(session=self.session, database_name=self.registry_db, schema_name=self.registry_schema)
            #st.write(f"Accessing registry: {self.registry_db}.{self.registry_schema}")
            model = reg.get_model(self.model_name)
            return model.default
        except Exception as e:
            return None
    
    # Predict the Operational Status based on input features from Streamlit
    def predict(self, input_data_frame):
        model = self.get_model()
        if model is None:
            st.error("Model retrieval failed.")
            return None
        
        try:
            prediction_df = model.run(input_data_frame, function_name="predict")
            #st.write(f"Prediction result: {prediction_df}")
            return prediction_df['output_feature_0'][0]
        except Exception as e:
            #st.error(f"Error during prediction: {e}")
            return None

#################################################################################################
#                                      USER INTERFACE                                           #
#################################################################################################


# Class for Streamlit app interface
class ManufacturingApp:
    def __init__(self, session):
        self.session = session
        #st.write("ManufacturingApp initialized")

    # Method to display input fields and collect data from the user
    def display_and_get_data(self):
        #st.subheader("Enter Sensor Data")

        try:
            # Create two columns for the Temperature input and badges
            # First row: Temperature input and badges
            col1, col2 = st.columns([4, 3])
            with col1:
                
                Temperature = st.number_input("Temperature (°C)",  value=0.0,min_value=-100.0, max_value=500.0, step=0.1)
            with col2:
                st.markdown(html_code_TEMP, unsafe_allow_html=True)
                #st.markdown(html_code_1, unsafe_allow_html=True)

            # Add padding between rows
            st.markdown("<div style='padding: 5px 0;'></div>", unsafe_allow_html=True)
            
            # Second row: Vibration input and badges
            col3, col4 = st.columns([4, 3])
            with col3:
                Vibration = st.number_input("Vibration (mm/s)",min_value=0.0, max_value=100.0, step=0.1)
            with col4:
                st.markdown(html_code_VIB, unsafe_allow_html=True)
                #st.markdown(html_code_1, unsafe_allow_html=True)

            # Add padding between rows
            st.markdown("<div style='padding: 5px 0;'></div>", unsafe_allow_html=True)
            
            # Third row: Pressure input and badges
            col5, col6 = st.columns([4, 3])
            with col5:
                Pressure = st.number_input("Pressure", min_value=0.0, max_value=500000.0, step=0.1)
            with col6:
                st.markdown(html_code_PRESS, unsafe_allow_html=True)
                #st.markdown(html_code_1, unsafe_allow_html=True)

            # Add padding between rows
            st.markdown("<div style='padding: 5px 0;'></div>", unsafe_allow_html=True)
            
            # Fourth row: Current input and badges
            col7, col8 = st.columns([4, 3])
            with col7:
                Current = st.number_input("Current (A)", min_value=0.0, max_value=1000.0, step=0.1)
            with col8:
                st.markdown(html_code_Current, unsafe_allow_html=True)
                #st.markdown(html_code_1, unsafe_allow_html=True)
            
            input_data = pd.DataFrame([[Temperature, Vibration, Pressure, Current]],
                                      columns=['Temperature', 'Vibration', 'Pressure', 'Current'])

            input_data.columns = ['input_feature_0', 'input_feature_1', 'input_feature_2', 'input_feature_3']

            return input_data

        except Exception as e:
            st.error(f"Error fetching or processing data: {e}")
            return pd.DataFrame()  


#################################################################################################
#                                     VISUALIZATIONS                                           # 
#################################################################################################

def load_data_from_snowflake():
    query = "SELECT * FROM MANUFACTURING.PUBLIC.IOT_DATA_PROD"
    session = get_active_session()  
    df = session.sql(query).to_pandas()  
    return df

def load_balanced_data():
    query = "SELECT * FROM MANUFACTURING.PUBLIC.IOT_DATA_PROD_BALANCED"
    session = get_active_session()  
    df = session.sql(query).to_pandas()  
    return df
    
iot_pandas = load_data_from_snowflake()
iot_pandas= iot_pandas[iot_pandas['Status'] != 'idle']
iot_balanced=load_balanced_data()

 # Defining means of our features

mean_temp_operational = iot_pandas[iot_pandas['Status'] == 'Operational']['Temperature'].mean()
mean_temp_error = iot_pandas[iot_pandas['Status'] == 'Error']['Temperature'].mean()
mean_press_operational = iot_pandas[iot_pandas['Status'] == 'Operational']['Pressure'].mean()
mean_press_error = iot_pandas[iot_pandas['Status'] == 'Error']['Pressure'].mean()
mean_vib_operational = iot_pandas[iot_pandas['Status'] == 'Operational']['Vibration'].mean()
mean_vib_error = iot_pandas[iot_pandas['Status'] == 'Error']['Vibration'].mean()
mean_current_operational = iot_pandas[iot_pandas['Status'] == 'Operational']['Current'].mean()
mean_current_error = iot_pandas[iot_pandas['Status'] == 'Error']['Current'].mean()

# Function to dynamically calculate thresholds based on mean ± 3 standard deviations
def calculate_thresholds(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_threshold = Q1 - 1.5 * IQR
    upper_threshold = Q3 + 1.5 * IQR
    return lower_threshold, upper_threshold

# Dynamically calculate thresholds for each sensor
temp_lower_threshold, temp_upper_threshold = calculate_thresholds(iot_pandas['Temperature'])
vib_lower_threshold, vib_upper_threshold = calculate_thresholds(iot_pandas['Vibration'])
press_lower_threshold, press_upper_threshold = calculate_thresholds(iot_pandas['Pressure'])
Current_lower_threshold, Current_upper_threshold = calculate_thresholds(iot_pandas['Current'])


def visualize_data(iot_pandas):
    st.subheader("Visualizations")

    # Time Series Plot for each sensor in a dropdown
    with st.expander("Time Series Plots"):
        fig, axs = plt.subplots(4, 1, figsize=(10, 15), sharex=True)
         
        Operational_data = iot_pandas[iot_pandas['Status'] == 'Operational']
        Error_data = iot_pandas[iot_pandas['Status'] == 'Error']
        
        axs[0].plot(iot_pandas['Timestamp'], iot_pandas['Temperature'], color='#003545', label='Temperature (°C)')
        axs[0].axhline(temp_upper_threshold, color='#d45b90', linestyle='--', label='Upper Threshold')
        axs[0].axhline(temp_lower_threshold, color='#ff9f36', linestyle='--', label='Lower Threshold')
        axs[0].axhline(76.52, color='black', linestyle='--', label='Upper DT')
        axs[0].axhline(86.58, color='black', linestyle='--', label='Lower DT')
        axs[0].set_title('Temperature Over Time')
        axs[0].legend(loc='upper right')

        axs[1].plot(iot_pandas['Timestamp'], iot_pandas['Vibration'], color='#008b9b', label='Vibration (mm/s)')
        axs[1].axhline(vib_upper_threshold, color='#d45b90', linestyle='--', label='Upper Threshold')
        axs[1].axhline(vib_lower_threshold, color='#ff9f36', linestyle='--', label='Lower Threshold')
        axs[1].axhline(19.46, color='black', linestyle='--', label='Upper DT')
        axs[1].axhline(26.51, color='black', linestyle='--', label='Lower DT')
        axs[1].set_title('Vibration Over Time')
        axs[1].legend(loc='upper right')

        axs[2].plot(iot_pandas['Timestamp'], iot_pandas['Pressure'], color='#00a9c4', label='Pressure')
        axs[2].axhline(press_upper_threshold, color='#d45b90', linestyle='--', label='Upper Threshold')
        axs[2].axhline(press_lower_threshold, color='#ff9f36', linestyle='--', label=
                       'Lower Threshold')
        axs[2].axhline(102033.15, color='black', linestyle='--', label='Upper DT')
        axs[2].axhline(114649.64, color='black', linestyle='--', label='Lower DT')
        axs[2].set_title('Pressure Over Time')
        axs[2].legend(loc='upper right')

        axs[3].plot(iot_pandas['Timestamp'], iot_pandas['Current'], color='#71d3dc', label='Current (A)')
        axs[3].axhline(Current_upper_threshold, color='#d45b90', linestyle='--', label='Upper Threshold')
        axs[3].axhline(Current_lower_threshold, color='#ff9f36', linestyle='--', label='Lower Threshold')
        axs[3].axhline(9.24, color='black', linestyle='--', label='Upper DT')
        axs[3].axhline(18.58, color='black', linestyle='--', label='Lower DT') 
        axs[3].set_title('Current Over Time')
        axs[3].legend(loc='upper right')

        plt.tight_layout()
        st.pyplot(fig)
        
    # Distribution Plot in a dropdown
    with st.expander("Sensor Data Distribution"):
        st.subheader("Sensor Data Distribution: Unbalanced vs Balanced")
    
        # Create a figure for 4X2 grid, 2 columns: one for unbalanced, one for balanced
        fig, axs = plt.subplots(4, 2, figsize=(20, 10), sharey=False)
    
        ## UNBALANCED DATA (iot_pandas)
        # Temperature distribution for Operational and Error statuses (Unbalanced)
        sns.histplot(
            iot_pandas[iot_pandas['Status'] == 'Operational']['Temperature'], 
            kde=True, ax=axs[0, 0], color='#003545', label='Operational'
        )
        sns.histplot(
            iot_pandas[iot_pandas['Status'] == 'Error']['Temperature'], 
            kde=True, ax=axs[0, 0], color='red', label='Error'
        )
        axs[0, 0].set_title('Temperature (Unbalanced)')
        axs[0, 0].legend()
    
        # Vibration distribution for Operational and Error statuses (Unbalanced)
        sns.histplot(
            iot_pandas[iot_pandas['Status'] == 'Operational']['Vibration'], 
            kde=True, ax=axs[1, 0], color='#008b9b', label='Operational'
        )
        sns.histplot(
            iot_pandas[iot_pandas['Status'] == 'Error']['Vibration'], 
            kde=True, ax=axs[1, 0], color='red', label='Error'
        )
        axs[1, 0].set_title('Vibration (Unbalanced)')
        axs[1, 0].legend()
    
        # Pressure distribution for Operational and Error statuses (Unbalanced)
        sns.histplot(
            iot_pandas[iot_pandas['Status'] == 'Operational']['Pressure'], 
            kde=True, ax=axs[2, 0], color='#00a9c4', label='Operational'
        )
        sns.histplot(
            iot_pandas[iot_pandas['Status'] == 'Error']['Pressure'], 
            kde=True, ax=axs[2, 0], color='red', label='Error'
        )
        axs[2, 0].set_title('Pressure (Unbalanced)')
        axs[2, 0].legend()
    
        # Current distribution for Operational and Error statuses (Unbalanced)
        sns.histplot(
            iot_pandas[iot_pandas['Status'] == 'Operational']['Current'], 
            kde=True, ax=axs[3, 0], color='#71d3dc', label='Operational'
        )
        sns.histplot(
            iot_pandas[iot_pandas['Status'] == 'Error']['Current'], 
            kde=True, ax=axs[3, 0], color='red', label='Error'
        )
        axs[3, 0].set_title('Current (Unbalanced)')
        axs[3, 0].legend()
    
        ## BALANCED DATA (iot_balanced)
        # Temperature distribution for Operational and Error statuses (Balanced)
        sns.histplot(
            iot_balanced[iot_balanced['STATUS'] == 'Operational']['TEMP'], 
            kde=True, ax=axs[0, 1], color='#003545', label='Operational'
        )
        sns.histplot(
            iot_balanced[iot_balanced['STATUS'] == 'Error']['TEMP'], 
            kde=True, ax=axs[0, 1], color='red', label='Error'
        )
        axs[0, 1].set_title('Temperature (Balanced)')
        axs[0, 1].legend()
    
        # Vibration distribution for Operational and Error statuses (Balanced)
        sns.histplot(
            iot_balanced[iot_balanced['STATUS'] == 'Operational']['VIB'], 
            kde=True, ax=axs[1, 1], color='#008b9b', label='Operational'
        )
        sns.histplot(
            iot_balanced[iot_balanced['STATUS'] == 'Error']['VIB'], 
            kde=True, ax=axs[1, 1], color='red', label='Error'
        )
        axs[1, 1].set_title('Vibration (Balanced)')
        axs[1, 1].legend()
    
        # Pressure distribution for Operational and Error statuses (Balanced)
        sns.histplot(
            iot_balanced[iot_balanced['STATUS'] == 'Operational']['PRESS'], 
            kde=True, ax=axs[2, 1], color='#00a9c4', label='Operational'
        )
        sns.histplot(
            iot_balanced[iot_balanced['STATUS'] == 'Error']['PRESS'], 
            kde=True, ax=axs[2, 1], color='red', label='Error'
        )
        axs[2, 1].set_title('Pressure (Balanced)')
        axs[2, 1].legend()
    
        # Current distribution for Operational and Error statuses (Balanced)
        sns.histplot(
            iot_balanced[iot_balanced['STATUS'] == 'Operational']['CURRENT'], 
            kde=True, ax=axs[3, 1], color='#71d3dc', label='Operational'
        )
        sns.histplot(
            iot_balanced[iot_balanced['STATUS'] == 'Error']['CURRENT'], 
            kde=True, ax=axs[3, 1], color='red', label='Error'
        )
        axs[3, 1].set_title('Current (Balanced)')
        axs[3, 1].legend()
    
        plt.tight_layout()
        st.pyplot(fig)

 
    # Operations vs Error Plot in a dropdown
    with st.expander("Operational States"):
        st.subheader("Operations vs. Error vs Idle")
        
        fig, axs = plt.subplots(2, 2, figsize=(12, 20))
        sns.boxplot(x='Status', y='Temperature', data=iot_pandas, ax=axs[0,0],color='#003545')
        axs[0,0].set_title('Temperature Distribution by Status')
        axs[0,0].set_xlabel('Operational Status')
        axs[0,0].set_ylabel('Temperature (°C)')
        
        sns.boxplot(x='Status', y='Pressure', data=iot_pandas, ax=axs[0,1],color='#008b9b')
        axs[0,1].set_title('Pressure Distribution by Operational Status')
        axs[0,1].set_xlabel('Operational Status')
        axs[0,1].set_ylabel('Pressure')
        
        sns.boxplot(x='Status', y='Vibration', data=iot_pandas, ax=axs[1,0], color='#00a9c4')
        axs[1,0].set_title('Vibration Distribution by Status')
        axs[1,0].set_xlabel('Operational Status')
        axs[1,0].set_ylabel('Vibration (mm/s)')
        
        sns.boxplot(x='Status', y='Current', data=iot_pandas, ax=axs[1,1],color='#71d3dc')
        axs[1,1].set_title('Current Distribution by Operational Status')
        axs[1,1].set_xlabel('Operational Status')
        axs[1,1].set_ylabel('Current (°C)')
        st.pyplot(fig)

#################################################################################################
#                                     HEADER PAGE SETUP                                                # 
#################################################################################################

col1, col2, col3,col4 = st.columns(4)
with col1:
    st.write(' ')

with col3:
    #st.image("https://drive.google.com/uc?export=view&id=1hrkL_ONLfbjrdjej-WVyGS8USBprDCME")
    st.image('https://cdn.prod.website-files.com/63d0bbc000931f731a555e1d/63d0be4de122277588c37e5c_pandata-logo-v2.png')

with col2:
    st.write(' ')

with col4:
    st.write(' ')

# Write directly to the app
st.header("Predictive Maintenance :robot_face:",divider='grey')
st.subheader("About Predictive Maintenance for Sensor Data")
st.write(
    """
    In the age of smart industries and advanced manufacturing, predictive 
    maintenance has become a vital solution for optimizing machine performance 
    and reducing downtime. By harnessing the power of sensor data and machine 
    learning, we can foresee potential equipment failures and take action before 
    they disrupt operations.
    """
)
st.subheader("What is Predictive Maintenance?")
st.write("""
Predictive maintenance is an approach that leverages real-time data from 
sensors to predict when equipment failure might occur. Rather than relying 
on a fixed maintenance schedule or waiting for an issue to happen, 
predictive maintenance allows you to act proactively, saving costs and 
improving efficiency.
""")
st.subheader("Our Solution")
st.write("""
Our platform utilizes advanced machine learning model to analyze sensor 
data such as Temperature, Vibration, Pressure, and Current. By continuously 
monitoring these key parameters, our model can detect patterns and anomalies, 
providing valuable insights into the Operational health of your machines.
""")

# HTML and CSS for two badges side by side
html_code_TEMP = f"""
<div style="display: flex; flex-direction: column; gap: 10px;">
    <div style="display: flex; justify-content: space-between; align-items: center; background-color: #4b4b4b; color: white; padding: 2px 8px; border-radius: 5px; font-size: 14px; width: 100%;">
        <span>Mean Temp for Operational</span>
        <span style="background-color: #0f1c1d1; color: white; padding: 2px 8px; border-radius: 5px;">{mean_temp_operational:.6f}</span>
    </div>
    <div style="display: flex; justify-content: space-between; align-items: center; background-color: #4b4b4b; color: white; padding: 2px 8px; border-radius: 5px; font-size: 14px; width: 100%;">
        <span>Mean Temp for Error</span>
        <span style="background-color: #0f1c1d1; color: white; padding: 2px 8px; border-radius: 5px;">{mean_temp_error:.6f}</span>
    </div>
</div>
"""

html_code_VIB = f"""
<div style="display: flex; flex-direction: column; gap: 10px;">
    <div style="display: flex; justify-content: space-between; align-items: center; background-color: #4b4b4b; color: white; padding: 2px 8px; border-radius: 5px; font-size: 14px; width: 100%;">
        <span>Mean Vib for Operational</span>
        <span style="background-color: #0f1c1d1; color: white; padding: 5px 10px; border-radius: 5px;">{mean_vib_operational:.6f}</span>
    </div>
    <div style="display: flex; justify-content: space-between; align-items: center; background-color: #4b4b4b; color: white; padding: 2px 8px; border-radius: 5px; font-size: 14px; width: 100%;">
        <span>Mean Vib for Error</span>
        <span style="background-color: #0f1c1d1; color: white; padding: 5px 10px; border-radius: 5px;">{mean_vib_error:.6f}</span>
    </div>
</div>
"""

html_code_PRESS = f"""
<div style="display: flex; flex-direction: column; gap: 10px;">
    <div style="display: flex; justify-content: space-between; align-items: center; background-color: #4b4b4b; color: white; padding: 2px 8px; border-radius: 5px; font-size: 14px; width: 100%;">
        <span>Mean press for Operational</span>
        <span style="background-color: #0f1c1d1; color: white; padding: 2px 8px; border-radius: 5px;">{mean_press_operational:.3f}</span>
    </div>
    <div style="display: flex; justify-content: space-between; align-items: center; background-color: #4b4b4b; color: white; padding: 2px 8px; border-radius: 5px; font-size: 14px; width: 100%;">
        <span>Mean press for Error</span>
        <span style="background-color: #0f1c1d1; color: white; padding: 2px 8px; border-radius: 5px;">{mean_press_error:.2f}</span>
    </div>
</div>
"""

html_code_Current = f"""
<div style="display: flex; flex-direction: column; gap: 10px;">
    <div style="display: flex; justify-content: space-between; align-items: center; background-color: #4b4b4b; color: white; padding: 2px 8px; border-radius: 5px; font-size: 14px; width: 100%;">
        <span>Mean Current for Operational</span>
        <span style="background-color: #0f1c1d1; color: white; padding: 5px 10px; border-radius: 5px;">{mean_current_operational:.6f}</span>
    </div>
    <div style="display: flex; justify-content: space-between; align-items: center; background-color: #4b4b4b; color: white; padding: 2px 8px; border-radius: 5px; font-size: 14px; width: 100%;">
        <span>Mean Current for Error</span>
        <span style="background-color: #0f1c1d1; color: white; padding: 5px 10px; border-radius: 5px;">{mean_current_error:.6f}</span>
    </div>
</div>
"""
#################################################################################################
#                                      RUN PREDUCTIONS                                          #
#################################################################################################

# Main method for running the Streamlit app
if __name__ == "__main__":
    # Define configuration variables
    registry_db = "MANUFACTURING"
    registry_schema = "PUBLIC"
    model_name = "xgboost_prod"
    #st.write("Starting app...")

#################################################################################################
#                                      TAB LAYOUT                                              #
#################################################################################################


 # Two tabs: one for prediction and one for visualization
    tab1, tab2 = st.tabs(["Predict Machine Status", "Visualize Data"])

    with tab1:
        app = ManufacturingApp(session)
        data = app.display_and_get_data() 
      
        predictor = ManufacturingPredictor(session, registry_db=registry_db, registry_schema=registry_schema, model_name=model_name)

        # Predict and display the output when the user clicks the "Predict" button
        if st.button("Predict"):
            prediction = predictor.predict(data)
            if prediction is not None:
                if prediction == 0:
                    st.error("Model indicates Error: Machine is not Operational")
                elif prediction == 1:
                    st.success("Model indicates Operational: The machine is functioning properly")
                else:
                    st.write(f"Unexpected prediction result: {prediction}")
            else:
                st.write("No prediction available")

    with tab2:
        iot_pandas
        if 'iot_pandas' in locals(): 
            visualize_data(iot_pandas)
        else:
            st.error("dataset not found for visualizations.")
