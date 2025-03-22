import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, Query
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


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
        'Lunch Actual', 'Difference (H-G)',
        'Snacks Ordered Previous day', 'Additional order2', 'Snacks Actual', 'DIffrence2 (J-L)'
    ]

    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
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
    facilities = data[['Facility']]  

    # Select predictive features (keep important ones)
    X = data[['Access Control Data (Footfall)', 'Lunch Ordered Previous Day', 
              'Additional Order', 'Total Order(F+G)', 'Snacks Ordered Previous day', 'Additional order2']]
    
    # Target variables (Lunch and Snacks Actuals)
    y = data[['Lunch Actual', 'Snacks Actual']]

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
        required_cols = {
            'Facility', 'Date', 'Day', 'Access Control Data (Footfall)', 'Lunch Ordered Previous Day',
            'Additional Order', 'Total Order(F+G)', 'Lunch Actual', 'Difference (H-G)', 'Dry Veg',
            'Gravy Veg', 'Rice', 'Dal', 'Sweet', 'Remarks/Justifications', 'Snacks Ordered Previous day',
            'Additional order2', 'Snacks Actual', 'DIffrence2 (J-L)', 'Menu', 'Remarks/Justifications2'
        }
        if not required_cols.issubset(df.columns):
            raise HTTPException(status_code=400, detail="Excel sheet is missing required columns.")
        train_model(df)
        X = df[['Access Control Data (Footfall)', 'Lunch Ordered Previous Day', 'Snacks Ordered Previous day']]
        predictions = trained_model.predict(X)
        generate_heatmap(df, predictions)
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

@app.post("/predict/")
async def predict(
    file: UploadFile = File(None),
    footfall: int = Query(None),
    lunch_ordered_prev: int = Query(None),
    additional_order: int = Query(None),
    snacks_ordered_prev: int = Query(None),
    additional_order2: int = Query(None),
    predict_wastage: bool = False
):
    """
    Endpoint to predict food consumption.
    - If a file is uploaded, predicts per facility (bulk mode).
    - If values are provided directly, predicts for those inputs.
    """
    global trained_model

    if trained_model is None:
        raise HTTPException(status_code=400, detail="Model is not trained. Please upload an Excel file first.")

    if file:
        try:
            contents = await file.read()
            df = pd.read_excel(io.BytesIO(contents))
            if 'Facility' not in df.columns:
                raise HTTPException(status_code=400, detail="Excel file must contain 'Facility' column.")
            df = preprocess_data(df)

            # Select features
            X = df[['Access Control Data (Footfall)', 'Lunch Ordered Previous Day',
                    'Additional Order', 'Total Order(F+G)', 'Snacks Ordered Previous day', 'Additional order2']]

            predictions = trained_model.predict(X)
            df[['Lunch Prediction', 'Snacks Prediction']] = predictions

            # Group results by facility
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

    # ✅ **Single Prediction (if values are provided)**
    elif any(v is not None for v in [footfall, lunch_ordered_prev, additional_order, snacks_ordered_prev, additional_order2]):
        input_data = np.array([[footfall or 0, lunch_ordered_prev or 0, additional_order or 0, snacks_ordered_prev or 0, additional_order2 or 0]])
        prediction = trained_model.predict(input_data)

        lunch_actual, snacks_actual = prediction[0]  # If predicting wastage, adjust accordingly
        response = {
            "prediction": {
                "lunch_actual": float(lunch_actual),
                "snacks_actual": float(snacks_actual),
            }
        }
        if predict_wastage:
            lunch_wastage, snacks_wastage = prediction[0][2:4] if len(prediction[0]) > 2 else (0, 0)
            response["prediction"]["lunch_wastage"] = float(lunch_wastage)
            response["prediction"]["snacks_wastage"] = float(snacks_wastage)
        
        single_df = pd.DataFrame([{
        "Facility": "Single Input",
        "Lunch Prediction": response["prediction"]["lunch_actual"],
        "Snacks Prediction": response["prediction"]["snacks_actual"]
        }])

        response["heatmaps"] = {
            "lunch": generate_heatmap(single_df, "Lunch Prediction"),
            "snacks": generate_heatmap(single_df, "Snacks Prediction")
        }

        return response

    else:
        raise HTTPException(status_code=400, detail="Provide either an Excel file or input values for prediction.")



def generate_heatmap(df, value_col):
    """
    Generates a heatmap for the given dataframe and value column.
    Returns the heatmap as a base64-encoded image.
    """
    plt.figure(figsize=(10, 6))
    pivot_df = df.pivot(index="Facility", columns=None, values=value_col)
    sns.heatmap(pivot_df, annot=True, cmap="coolwarm", fmt=".1f")
    plt.title(f"{value_col} Heatmap")
    
    # Convert plot to image
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


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