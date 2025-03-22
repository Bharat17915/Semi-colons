import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


app = FastAPI()

# Initialize model variables
model = LinearRegression()
trained_model = None  # Ensuring a global trained model reference


def preprocess_data(data: pd.DataFrame, reference_columns=None):
    """
    Preprocess the data by:
    - Removing holidays based on 'Remarks/Justifications'
    - Excluding Sundays
    - Converting necessary columns to numeric
    """

    if reference_columns:
        reference_column = ['footfall', 'additional lunch', 'total lunch (e+f)', 'lunch ordered', 'difference lunch(h-g)',
              'total snacks','snacks ordered','additional snacks','date','day','difference snacks(s-r)' ]
        for col in reference_column:
            if col not in data.columns:
                data[col] = 0  # Assign a default value (can be changed if needed)
        data = data[reference_column]

    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data = data.dropna(subset=['date'])
    data = data[data['date'].dt.dayofweek != 6]  # 6 = Sunday

    for col in data.select_dtypes(include=['datetime64[ns]']):
        data[col] = data[col].astype(int) // 10**9

    # Remove holidays (if "Holiday" is mentioned in "Remarks/Justifications")
    if reference_columns:
        pass
    else:
        if 'remarks' or' justification' in data.columns:
            data['remarks'] = data['remarks'].astype(str)
            data['justification'] = data['justification'].astype(str)
            data = data[~data['remarks'].str.contains("Holiday", case=False, na=False)]

    numeric_cols = [
        'footfall', 'lunch ordered', 'additional lunch', 'total lunch (e+f)',
        'difference lunch(h-g)',
        'snacks ordered', 'additional snacks', 'difference snacks(s-r)'
    ]

    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    categorical_cols = ['day', 'facility', 'menu']  # Add other categorical columns if needed
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype('category').cat.codes 

    data = data.dropna()
    return data


def train_model(data: pd.DataFrame):
    """
    Train the linear regression model using the given data and store per-facility predictions.
    """
    global trained_model
    data = preprocess_data(data)

    if data.empty:
        raise ValueError("No valid data available for training after preprocessing.")

    # Keep 'Facility' for later grouping but exclude it from training
    facilities = data[['facility']]  

    # Select predictive features (keep important ones)
    X = data[['footfall', 'additional lunch', 'total lunch (e+f)', 'lunch ordered', 'difference lunch(h-g)',
              'total snacks','snacks ordered','additional snacks','date','day','difference snacks(s-r)' ]]
    
    # Target variables (Lunch and Snacks Actuals)
    y = data[['lunch actual', 'snacks actual']]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    trained_model = model

@app.post("/upload_excel/")
async def upload_excel(
    file: UploadFile = File(...), 
    sheet_name: str = Form(...)
):
    """ Upload an Excel file, train the model, predict and return results with heatmaps. """
    global trained_model
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents), sheet_name=sheet_name)
        required_cols = {'facility', 'difference snacks(s-r)', 'footfall', 'dry veg', 
                         'dal', 'difference lunch(h-g)', 'total snacks', 'snacks actual', 
                         'lunch actual', 'rice', 'additional snacks', 'remarks', 'menu', 
                         'lunch ordered', 'snacks ordered', 'justification', 'total lunch (e+f)', 
                         'sweet', 'gravy veg', 'day', 'additional lunch', 'date',}
       
        # Normalize the column names in the DataFrame
        df_columns_normalized = {col.lower() for col in df.columns}
        facilities = df[['facility']]
        # Check if all required columns are present
        if not required_cols.issubset(df_columns_normalized):
            raise HTTPException(status_code=400, detail="Excel sheet is missing required columns.")
        
        train_model(df)
        X = df[['footfall', 'lunch ordered', 'snacks ordered','facility']]
        x = preprocess_data(X,reference_columns=df_columns_normalized)
        predictions = trained_model.predict(x)
        x["Predicted_Consumption"] = predictions[:, 0]
        x["Predicted_Wastage"] = predictions[:, 1]
        x["facility"] = X.loc[x.index, "facility"]
        generate_heatmap(X=x, col1="Predicted_Consumption", col2="Predicted_Wastage")
        response = {
            "message": "Model trained successfully!",
            "predictions": predictions.tolist(),
            "heatmaps": ["lunch_heatmap.png", "snacks_heatmap.png"]
        }
        return response
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid sheet name: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

def generate_heatmap(X, col1, col2):
    facilities = X["facility"]
    values1 = X[col1]
    values2 = X[col2]
    
    x = np.arange(len(facilities))
    width = 0.4
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, values1, width, label=col1, color="blue")
    ax.bar(x + width/2, values2, width, label=col2, color="red")
    
    ax.set_xlabel("Facility")
    ax.set_ylabel("Values")
    ax.set_title(f"{col1} & {col2} per Facility")
    ax.set_xticks(x)
    ax.set_xticklabels(facilities, rotation=45, ha="right")
    ax.legend()
    
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@app.post("/predict-file/")
async def predict_file(
    file: UploadFile = File(...),
    sheet_name: str = Form(...)
):
    """
    Endpoint to predict food consumption per facility using an uploaded Excel file and sheet name.
    """
    global trained_model

    if trained_model is None:
        raise HTTPException(status_code=400, detail="Model is not trained. Please upload an Excel file first.")

    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents), sheet_name=sheet_name)
        
        if 'Facility' not in df.columns:
            raise HTTPException(status_code=400, detail="Excel file must contain 'Facility' column.")
        
        df = preprocess_data(df)
        
        # Select features
        X = df[['footfall', 'lunch ordered', 'additional lunch', 'total lunch (e+f)', 'snacks ordered', 'additonal snacks']]
        
        predictions = trained_model.predict(X)
        df[['Lunch Prediction', 'Snacks Prediction']] = predictions
        
        # Group by facility
        result = df.groupby("Facility")[['Lunch Prediction', 'Snacks Prediction']].mean().reset_index()
        
        # Generate heatmaps
        lunch_heatmap = generate_heatmap(result, 'Lunch Prediction')
        snacks_heatmap = generate_heatmap(result, 'Snacks Prediction')
        
        return {
            "predictions": result.to_dict(orient="records"),
            "heatmaps": {
                "lunch": lunch_heatmap,
                "snacks": snacks_heatmap
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/predict/")
async def predict(
    footfall: int = Form(None),
    lunch_ordered_prev: int = Form(None),
    additional_order: int = Form(None),
    snacks_ordered_prev: int = Form(None),
    additional_order2: int = Form(None)
):
    """
    Endpoint to predict food consumption based on direct input values.
    """
    global trained_model

    if trained_model is None:
        raise HTTPException(status_code=400, detail="Model is not trained. Please upload an Excel file first.")

    if any(v is not None for v in [footfall, lunch_ordered_prev, additional_order, snacks_ordered_prev, additional_order2]):
        input_data = np.array([[footfall or 0, lunch_ordered_prev or 0, additional_order or 0, snacks_ordered_prev or 0, additional_order2 or 0]])
        prediction = trained_model.predict(input_data)
        
        lunch_actual, snacks_actual = prediction[0][:2]  # Adjust indices if more predictions are added
        lunch_wastage, snacks_wastage = (prediction[0][2:4] if len(prediction[0]) > 2 else (0, 0))
        
        response = {
            "prediction": {
                "lunch_actual": float(lunch_actual),
                "snacks_actual": float(snacks_actual),
                "lunch_wastage": float(lunch_wastage),
                "snacks_wastage": float(snacks_wastage),
            }
        }
        
        single_df = pd.DataFrame([{
            "Facility": "Single Input",
            "Lunch Prediction": lunch_actual,
            "Snacks Prediction": snacks_actual,
            "Lunch Wastage Prediction": lunch_wastage,
            "Snacks Wastage Prediction": snacks_wastage
        }])
        
        response["heatmaps"] = {
            "lunch": generate_heatmap(single_df, "Lunch Prediction"),
            "snacks": generate_heatmap(single_df, "Snacks Prediction"),
            "lunch wastage": generate_heatmap(single_df, "Lunch Wastage Prediction"),
            "snacks wastage": generate_heatmap(single_df, "Snacks Wastage Prediction")
        }
        
        return response
    
    else:
        raise HTTPException(status_code=400, detail="Provide input values for prediction.")



@app.post("/admin/train_full_dataset/")
async def admin_train_full_dataset(file: UploadFile = File(...)):
    """
    Admin-only function to train the model with the entire dataset, without removing holidays or Sundays.
    This function does not impact the regular training or prediction process.
    """
    try:    
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        required_cols = {
            'Facility', 'Date', 'Day', 'Access Control Data (Footfall)', 'Lunch Ordered Previous Day',
            'Additional Order', 'Total Order(F+G)', 'Lunch Actual', 'Difference (H-G)', 'Dry Veg',
            'Gravy Veg', 'Rice', 'Dal', 'Sweet', 'Remarks/Justifications', 'Snacks Ordered Previous day',
            'Additional order2', 'Snacks Actual', 'DIffrence2 (J-L)', 'Menu', 'Remarks/Justifications2'
        }
        if not required_cols.issubset(df.columns):
            raise HTTPException(status_code=400, detail="Excel sheet is missing required columns.")
        X = df.drop(columns=['Facility', 'Date', 'Menu', 'Remarks/Justifications', 'Remarks/Justifications2'])
        y = df[['Lunch Actual', 'Snacks Actual', 'Difference (H-G)', 'DIffrence2 (J-L)']]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_full = LinearRegression()
        model_full.fit(X_train, y_train)
        y_pred = model_full.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f'Mean Squared Error (Full Dataset Training): {mse}')
        return {"message": "Model trained successfully with full dataset! (Admin only)"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")