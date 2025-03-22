import pandas as pd
import numpy as np
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import io

app = FastAPI()

# Initialize model variables
model = LinearRegression()
trained_model = None  # Ensuring a global trained model reference


def preprocess_data(data: pd.DataFrame):
    """
    Preprocess the data by:
    - Removing holidays based on 'Remarks/Justifications'
    - Excluding Sundays
    - Converting necessary columns to numeric
    """

    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data = data.dropna(subset=['Date'])
    data = data[data['Date'].dt.dayofweek != 6]  # 6 = Sunday

    # Remove holidays (if "Holiday" is mentioned in "Remarks/Justifications")
    if 'Remarks/Justifications' in data.columns:
        data = data[~data['Remarks/Justifications'].str.contains("Holiday", case=False, na=False)]

    numeric_cols = [
        'Access Control Data (Footfall)', 'Lunch Ordered Previous Day', 'Additional Order', 'Total Order(F+G)',
        'Lunch Actual', 'Difference (H-G)', 'Dry Veg', 'Gravy Veg', 'Rice', 'Dal', 'Sweet',
        'Snacks Ordered Previous day', 'Additional order2', 'Snacks Actual', 'DIffrence2 (J-L)'
    ]

    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna()
    return data

def verify_admin(user_role: str = "user"):
    if user_role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")

def train_model(data: pd.DataFrame):
    """
    Train the linear regression model using the given data.
    """
    global trained_model
    data = preprocess_data(data)

    if data.empty:
        raise ValueError("No valid data available for training after preprocessing.")

    # Define features (X) and targets (y)
    X = data.drop(columns=['Facility', 'Date', 'Remarks/Justifications', 'Remarks/Justifications2'])
    y = data[['Lunch Actual', 'Snacks Actual']]  # Predicting actual lunch/snacks consumption

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    trained_model = model  # Store the trained model


@app.post("/upload_excel/")
async def upload_excel(
    file: UploadFile = File(...), 
    sheet_name: str = Form(...)
):
    """
    Endpoint to upload an Excel file and train the model using a specific sheet.
    """
    global trained_model

    try:
        contents = await file.read()
        # Read the Excel file with the given sheet name
        df = pd.read_excel(io.BytesIO(contents), sheet_name=sheet_name)

        required_cols = {
            'Facility', 'Date', 'Day', 'Access Control Data (Footfall)', 'Lunch Ordered Previous Day',
            'Additional Order', 'Total Order(F+G)', 'Lunch Actual', 'Difference (H-G)', 'Dry Veg',
            'Gravy Veg', 'Rice', 'Dal', 'Sweet', 'Remarks/Justifications', 'Snacks Ordered Previous day',
            'Additional order2', 'Snacks Actual', 'DIffrence2 (J-L)', 'Menu', 'Remarks/Justifications2'
        }
        if not required_cols.issubset(df.columns):
            raise HTTPException(status_code=400, detail="Excel sheet is missing required columns.")

        train_model(df)
        return {"message": "Model trained successfully!"}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid sheet name: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/predict/")
async def predict(
    footfall: int = None,
    lunch_ordered_prev: int = None,
    additional_order: int = None,
    snacks_ordered_prev: int = None,
    additional_order2: int = None,
    predict_wastage: bool = False
):
    """
    Endpoint to predict food consumption and optionally wastage based on input features.
    """
    global trained_model

    if trained_model is None:
        raise HTTPException(status_code=400, detail="Model is not trained. Please upload an Excel file first.")

    if all(v is None for v in [footfall, lunch_ordered_prev, additional_order, snacks_ordered_prev, additional_order2]):
        raise HTTPException(status_code=400, detail="At least one feature must be provided for prediction.")

    input_data = np.array([[
        footfall or 0,
        lunch_ordered_prev or 0,
        additional_order or 0,
        snacks_ordered_prev or 0,
        additional_order2 or 0
    ]])

    prediction = trained_model.predict(input_data)
    lunch_actual, snacks_actual, lunch_wastage, snacks_wastage = prediction[0]
    response = {
        "prediction": {
            "lunch_actual": float(lunch_actual),
            "snacks_actual": float(snacks_actual),
        }
    }
    if predict_wastage:
        response["prediction"]["lunch_wastage"] = float(lunch_wastage)
        response["prediction"]["snacks_wastage"] = float(snacks_wastage)

    return response

@app.post("/admin/train_full_dataset/")
async def admin_train_full_dataset(file: UploadFile = File(...), user_role: str = Depends(verify_admin)):
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