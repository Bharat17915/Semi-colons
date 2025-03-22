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
        'facility', 'date', 'day', 'access control data (footfall)', 'lunch ordered previous day',
        'additional order', 'total order(f+g)', 'lunch actual', 'difference (h-g)', 'dry veg',
        'gravy veg', 'rice', 'dal', 'sweet', 'remarks/justifications', 'snacks ordered previous day',
        'additional order2', 'snacks actual', 'diffrence2 (j-l)', 'menu', 'remarks/justifications2'
        }

        # Normalize the column names in the DataFrame
        df_columns_normalized = {col.lower() for col in df.columns}

        # Check if all required columns are present
        if not required_cols.issubset(df_columns_normalized):
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

def generate_heatmap(df, column):
    facilities = df['Facility']
    values = df[column]
    
    x = np.arange(len(facilities))
    width = 0.4
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, values, width, label=column)
    
    ax.set_xlabel('Facility')
    ax.set_ylabel(column)
    ax.set_title(f'{column} per Facility')
    ax.set_xticks(x)
    ax.set_xticklabels(facilities, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.post("/predict/")
async def predict(
    file: UploadFile = File(None),
    sheet_name: str = Form(None),
    footfall: int = Query(None),
    lunch_ordered_prev: int = Query(None),
    additional_order: int = Query(None),
    snacks_ordered_prev: int = Query(None),
    additional_order2: int = Query(None),
):
    global trained_model

    if trained_model is None:
        raise HTTPException(status_code=400, detail="Model is not trained. Please upload an Excel file first.")

    if file:
        try:
            contents = await file.read()
            df = pd.read_excel(io.BytesIO(contents))
            if 'Facility' not in df.columns:
                raise HTTPException(status_code=400, detail="Excel file must contain 'Facility' column.")
            
            # Preprocess data (assuming preprocess_data exists)
            df = preprocess_data(df)
            
            # Select features for prediction
            X = df[['Access Control Data (Footfall)', 'Lunch Ordered Previous Day',
                    'Additional Order', 'Total Order(F+G)', 'Snacks Ordered Previous day', 'Additional order2']]
            
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
    
    elif any(v is not None for v in [footfall, lunch_ordered_prev, additional_order, snacks_ordered_prev, additional_order2]):
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
        raise HTTPException(status_code=400, detail="Provide either an Excel file or input values for prediction.")


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