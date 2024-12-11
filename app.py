import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import numpy as np


import category_encoders as ce
import json
import tensorflow as tf
from sklearn.metrics import mean_squared_error



data = {"Angul": ["Angul", "Athamalik", "Banarpal", "Chhendipada", "Kishorenagar", "Pallahara", "Talcher"],
        "Balangir": ["Agalpur", "Balangir", "Bangomunda", "Belpara", "Deogaon", "Gudvella", "Khaprakhol", "Loisingha", "Muribahal", "Patnagarh", "Saintala", "Tureikela"],
        "Balasore": ["Balasore", "Baliapal", "Basta", "Bhogarai", "Jaleswar", "Khaira", "Nilgiri", "Oupada", "Remuna", "Simulia", "Soro"],
        "Bargarh": ["Ambabhona", "Attabira", "Bargarh", "Barpali", "Bheden", "Bijepur", "Gaisilet", "Jharbandh", "Paikmal", "Sohela"],
        "Bhadrak": ["Basudevpur", "Bhadrak", "Bonth", "Chandabali", "Dhamnagar", "Tihidi"],
        "Boudh": ["Boudh", "Harabhanga", "Kantamal"],
        "Cuttack": ["Athagad", "Badamba", "Banki", "Baramba", "Cuttack Sadar", "Damapada", "Kantapada", "Mahanga", "Narasinghpur", "Niali", "Salepur", "Tigiria"],
        "Debagarh (Deogarh)": ["Barkote", "Deogarh", "Reamal"],
        "Dhenkanal": ["Bhuban", "Dhenkanal Sadar", "Gondia", "Hindol", "Kamakhyanagar", "Kankadahad", "Odapada", "Parjang"],
        "Gajapati": ["Chandragiri", "Mohana", "Nuagada", "R. Udayagiri"],
        "Ganjam": ["Aska", "Bellaguntha", "Beguniapada", "Buguda", "Chhatrapur", "Chikiti", "Dharakote", "Digapahandi", "Ganjam", "Hinjilicut", "Jagannath Prasad", "Khallikote", "Kabisuryanagar", "Kukudakhandi", "Patrapur", "Polasara", "Purushottampur", "Rangeilunda", "Sanakhemundi", "Sheragada", "Sorada"],
        "Jagatsinghpur": ["Balikuda", "Biridi", "Erasama", "Jagatsinghpur", "Kujang", "Naugaon", "Raghunathpur", "Tirtol"],
        "Jajpur": ["Badachana", "Bari", "Binjharpur", "Dasarathpur", "Dharmasala", "Jajpur", "Korei", "Rasulpur", "Sukinda"],
        "Jharsuguda": ["Jharsuguda", "Kirmira", "Kolabira", "Lakhanpur", "Laikera"],
        "Kalahandi": ["Bhawanipatna", "Dharmagarh", "Golamunda", "Jaipatna", "Junagarh", "Kalampur", "Karlamunda", "Kesinga", "Lanjigarh", "Madanpur Rampur", "Narla", "Thuamul Rampur"],
        "Kandhamal": ["Balliguda", "Chakapad", "Daringbadi", "G. Udayagiri", "K. Nuagaon", "Kotagarh", "Phiringia", "Phulbani", "Raikia", "Tikabali", "Tumudibandha"],
        "Kendrapara": ["Aul", "Derabish", "Garadpur", "Kendrapara", "Mahakalapada", "Marsaghai", "Pattamundai", "Rajkanika", "Rajnagar"],
        "Keonjhar (Kendujhar)": ["Anandapur", "Banspal", "Champua", "Ghasipura", "Harichandanpur", "Hatadihi", "Jhumpura", "Joda", "Keonjhar Sadar", "Patna", "Saharpada", "Telkoi"],
        "Khordha": ["Balianta", "Balipatna", "Banapur", "Begunia", "Bhubaneswar", "Bolagarh", "Chilika", "Jatni", "Khordha", "Tangi"],
        "Koraput": ["Bandhugaon", "Borigumma", "Dasamantapur", "Koraput", "Kotpad", "Lamtaput", "Laxmipur", "Nandapur", "Pottangi", "Semiliguda"],
        "Malkangiri": ["Kalimela", "Khairaput", "Korukonda", "Kudumuluguma", "Malkangiri", "Mathili", "Podia"],
        "Mayurbhanj": ["Bangiriposi", "Baripada", "Betnoti", "Bijatala", "Bisoi", "Jashipur", "Kaptipada", "Khunta", "Karanjia", "Kusumi", "Morada", "Rairangpur", "Raruan", "Samakhunta", "Saraskana", "Sukruli", "Thakurmunda", "Udala"],
        "Nabarangpur": ["Chandahandi", "Dabugam", "Jharigam", "Kosagumuda", "Nabarangpur", "Papadahandi", "Raighar", "Tentulikhunti"],
        "Nayagarh": ["Bhapur", "Dasapalla", "Gania", "Khandapada", "Nayagarh", "Nuagaon", "Odagaon", "Ranpur"],
        "Nuapada": ["Boden", "Khariar", "Komna", "Nuapada", "Sinapali"],
        "Puri": ["Astaranga", "Brahmagiri", "Delang", "Gop", "Kakatpur", "Kanas", "Krushnaprasad", "Nimapada", "Puri Sadar", "Satyabadi"],
        "Rayagada": ["Bissam Cuttack", "Chandrapur", "Gudari", "Gunupur", "Kashipur", "Kolnara", "Muniguda", "Padmapur", "Rayagada"],
        "Sambalpur": ["Bamra", "Dhankauda", "Jujomura", "Kuchinda", "Maneswar", "Naktideul", "Rairakhol"],
        "Subarnapur (Sonepur)": ["Birmaharajpur", "Dunguripali", "Sonepur", "Tarbha", "Ullunda"],
        "Sundargarh": ["Balishankara", "Bargaon", "Bisra", "Bonai", "Gurundia", "Hemgir", "Koida", "Kuarmunda", "Kutra", "Lahunipada", "Lephripara", "Rajgangpur", "Subdega", "Sundargarh", "Tangarpali"]
    }

# List of columns to fetch
soil_data = [
    'Nitrogen - High', 'Nitrogen - Medium', 'Nitrogen - Low', 'Phosphorous - High', 'Phosphorous - Medium',
    'Phosphorous - Low', 'Potassium - High', 'Potassium - Medium', 'Potassium - Low', 'OC - High', 'OC - Medium', 
    'OC - Low', 'EC - Saline', 'EC - Non Saline', 'pH - Acidic', 'pH - Neutral', 'pH - Alkaline', 
    'Copper - Sufficient', 'Copper - Deficient', 'Boron - Sufficient', 'Boron - Deficient', 'S - Sufficient',
    'S - Deficient', 'Fe - Sufficient', 'Fe - Deficient', 'Zn - Sufficient', 'Zn - Deficient', 
    'Mn - Sufficient', 'Mn - Deficient'
]
# File paths for CSV files
CSV_FILE_PATH = "odisha_dataset_DNN.csv"  # Main dataset file
OTHER_CSV_FILE_PATH = "soil_data.csv"  # Secondary dataset file

# Predefined values for sowing and harvesting
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
weeks = ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5"]

def main():
    st.title("Crop Yield Prediction")

    # Dropdown: District
    district = st.selectbox("Select a District", options=["--Select--"] + list(data.keys()), key="district")

    # Dropdown: Block
    block_options = ["--Select a district --"] if district == "--Select--" else ["--Select--"] + data[district]
    block = st.selectbox("Select a Block", options=block_options, key="block")

    # Load the main dataset
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        df=df.dropna()
    except Exception as e:
        st.error(f"Error loading the dataset: {e}")
        df = pd.DataFrame()

    # Dropdown: Crop
    if district != "--Select--" and block != "--Select--" and not df.empty:
        filtered_crops = df[(df["District"].str.upper() == district.upper()) & 
                            (df["Block"].str.upper() == block.upper())]["Crop"].unique()
        crop_options = ["--Select--"] + list(filtered_crops)
    else:
        crop_options = ["--Select a block--"]
    crop = st.selectbox("Select a Crop", options=crop_options, key="crop")

    # Dropdown: Variety
    if crop != "--Select--" and not df.empty:
        filtered_varieties = df[df["Crop"] == crop]["Variety"].unique()
        variety_options = ["--Select--"] + list(filtered_varieties)
    else:
        variety_options = ["--Select a valid crop--"]
    variety = st.selectbox("Select a Variety", options=variety_options, key="variety")

    # Dropdown: Crop Pattern
    if crop != "--Select--" and not df.empty:
        crop_pattern_options = ["--Select--"] + list(df["cropPattern"].unique())
    else:
        crop_pattern_options = ["--Select a valid crop--"]
    crop_pattern = st.selectbox("Select Crop Pattern", options=crop_pattern_options, key="crop_pattern")

    # Dropdown: Type of Sowing
    if crop != "--Select--" and not df.empty:
        sowing_type_options = ["--Select--"] + list(df["typeOfSowing"].unique())
    else:
        sowing_type_options = ["--Select a valid crop--"]
    type_of_sowing = st.selectbox("Select Type of Sowing", options=sowing_type_options, key="type_of_sowing")

    # Dropdown: Sowing Month
    sowing_month = st.selectbox("Select Sowing Month", options=["--Select--"] + months, key="sowing_month")

    # Dropdown: Sowing Week
    sowing_week = st.selectbox("Select Sowing Week", options=["--Select--"] + weeks, key="sowing_week")

    # Dropdown: harvesting month
    harvesting_month = st.selectbox("Select harvesting month", options=["--Select--"] + months, key="harvesting_month")

    # Dropdown: Harvesting Week
    harvesting_week = st.selectbox("Select Harvesting Week", options=["--Select--"] + weeks, key="harvesting_week")

    # Number Input: Registered Area
    registered_area = st.number_input(
        "Enter Registered Area (in hectares)",
        min_value=0.0,
        step=0.1,
        format="%.2f",
        key="registered_area"
    )

    # Number Input: Sowing Quantity
    sowing_quantity = st.number_input(
        "Enter Sowing Quantity (in Qtl)",
        min_value=0.0,
        step=0.1,
        format="%.2f",
        key="sowing_quantity"
    )

    # Dropdown: Source Class
    if not df.empty:
        source_class_options = ["--Select--"] + list(df["Source Class"].dropna().unique())
    else:
        source_class_options = ["--No source class data--"]
    source_class = st.selectbox("Select Source Class", options=source_class_options, key="source_class")

    # Dropdown: Destination Class
    if not df.empty:
        destination_class_options = ["--Select--"] + list(df["Destination Class"].dropna().unique())
    else:
        destination_class_options = ["--No destination class data--"]
    destination_class = st.selectbox("Select Destination Class", options=destination_class_options, key="destination_class")

    # Submit Button
    if st.button("Submit"):
        # Collect user inputs
        user_data = {
            "District": district,
            "Block": block,
            "Crop": crop,
            "Variety": variety,
            "cropPattern": crop_pattern,
            "typeOfSowing": type_of_sowing,
            "Sowing Month": sowing_month.upper(),
            "Sowing Week": sowing_week,
            "harvesting month": harvesting_month.upper(),
            "Harvesting week": harvesting_week,
            "Registered Area": registered_area,
            "Sowing Quantity": sowing_quantity,
            "Source Class": source_class,
            "Destination Class": destination_class
        }
        st.json(user_data)
        

        # Find the first matching row for District and Block
        if district != "--Select--" and block != "--Select--" and not df.empty:
            matched_row = df[
                (df["District"].str.upper() == district.upper()) &
                (df["Block"].str.upper() == block.upper())
            ].index # Get the first match only
            # st.write(matched_row)
            if matched_row.empty:
                st.warning("No matching rows found in the main dataset.")
            else:
                st.success(" ")

              

                # Get the index of the matched row
                matched_index = int(matched_row[0])
                # st.write(matched_index)
                
                # Load the secondary dataset
                try:
                    other_df = pd.read_csv(OTHER_CSV_FILE_PATH)
                    other_df=other_df[soil_data]
                    other_df = other_df.iloc[matched_index:matched_index+1]

                    # st.warning('testing')
                    # st.write(other_df.head())
                    selected_column=['Variety','Source Class','Destination Class','Registered Area','Sowing Month','Sowing Week',
                         'typeOfSowing','cropPattern','Harvesting week','harvesting month']
                    user_data_column = {key: user_data[key] for key in selected_column}
                    user_df = pd.DataFrame([user_data_column])
                    # st.write(user_df)
                    columns_encode = ['Source Class', 'Destination Class', 'Sowing Week', 'Sowing Month','Harvesting week', 'harvesting month',
                      'cropPattern', 'typeOfSowing']
                    
                    user_df = pd.get_dummies(user_df, columns=columns_encode, drop_first=False)
                    # st.write(user_df)

                    encoder_for_variety = ce.BinaryEncoder(cols=['Variety'])
                    encoded_variety = encoder_for_variety.fit_transform(user_df['Variety'])
                    print(encoded_variety)
                     # Drop the original column and concatenate the encoded columns
                    user_df = user_df.drop('Variety', axis=1)
                    user_df = pd.concat([user_df, encoded_variety], axis=1)
                    # st.write(other_df)
                    # st.write(other_df.columns)
                    # st.write(user_df)
                    # combined_df=pd.concat([other_df,user_df],axis=1)
                    # st.write(combined_df)
                   
                    # st.write('testing combined')
                    combined_df = pd.concat([user_df.reset_index(drop=True), other_df.reset_index(drop=True)], axis=1)
                    combined_df=combined_df.rename(columns={'Registered Area': 'Certified Area'})

                    # st.write(combined_df)
                    # st.write('shape of combined')
                    # st.write(combined_df.shape)

                    with open('training_columns.json', 'r') as file:
                        training_columns = json.load(file)
                    for col in combined_df:
                        if col not in training_columns:
                            st.warning(col)
                    # st.write('')
                    for col in training_columns:
                        if col not in combined_df:
                            combined_df[col]=0
                    # st.write(combined_df.columns)
                    # st.write(combined_df)
                    # st.write(len(training_columns))
                    model_path = 'new_model.h5'

                    # doing scaling of features
                    # print(len(num_columns))
                    # soil_data=soil_data.extend(['Sowed Quantity','Certified Area'])
                    # num_columns=soil_data
                    # st.write(soil_data)
                    num_columns= ['Copper - Sufficient', 'Copper - Deficient', 'Boron - Sufficient',
       'Boron - Deficient',
       'Fe - Sufficient', 'Fe - Deficient', 'Zn - Sufficient',
       'Zn - Deficient', 'Mn - Sufficient', 'Mn - Deficient',
       'Nitrogen - High', 'Nitrogen - Medium', 'Nitrogen - Low',
       'Phosphorous - High', 'Phosphorous - Medium', 'Phosphorous - Low',
       'Potassium - High', 'Potassium - Medium', 'Potassium - Low',
        'EC - Saline', 'EC - Non Saline',
       'pH - Acidic', 'pH - Neutral', 'pH - Alkaline','OC - Medium','OC - High','OC - Low', 'Sowed Quantity','Certified Area']

                    complement_columns = list(set(combined_df.columns) - set(num_columns))
                    # st.write(complement_columns)

                    # Define transformations
                    numerical_transformer = StandardScaler()
                    categorical_transformer = 'passthrough'  # Do nothing for one-hot encoded features

                    # Combine transformations
                    preprocessor = ColumnTransformer(
                    transformers=[
                    ('num', numerical_transformer, num_columns),
                    ('cat', categorical_transformer, complement_columns)
                        ]
                            )

                    processed_data = preprocessor.fit_transform(combined_df).astype(np.float32)
                    # print(processed_data)
                    # print(processed_data.shape)
                    processed_df = pd.DataFrame(processed_data, columns=num_columns + complement_columns)
                    print(processed_df.head())

                    # Load the model
                    loaded_model = tf.keras.models.load_model(model_path, custom_objects={'mse': mean_squared_error})
                    
                    prediction = loaded_model.predict(processed_df)
                    st.write('Predicted Yield')
                    st.write(prediction)






                except Exception as e:
                    st.error(f"Error loading the secondary dataset: {e}")
                  
             
                    
if __name__ == "__main__":
    main()
