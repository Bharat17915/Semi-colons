# Sustainable Dining — Food Wastage Prediction

> *Plan Better, Waste Less — Predict tomorrow's food needs & minimize waste.*

An AI-powered full-stack application that helps corporate and institutional canteens
predict daily food demand, so kitchens order closer to what's actually consumed instead
of guessing — cutting both food waste and shortages.

---

## Problem

Canteens order lunch and snacks a day in advance based on rough estimates. This leads to:

- **Over-ordering** → food is cooked and thrown away.
- **Under-ordering** → shortages and unhappy staff.

By learning from historical consumption patterns (footfall, past orders, day of week),
the model predicts how much food each facility will actually need.

---

## Features

- **Train Model** — Upload an Excel sheet of past consumption data; the model learns demand patterns.
- **Prediction** — Enter expected orders (or upload a file) to get predicted lunch/snacks consumption and wastage.
- **Per-facility insights** — Predictions and bar-chart visualizations grouped by facility.
- **Data cleaning built in** — Automatically excludes Sundays, filters holidays, and normalizes messy spreadsheet data.

---

## Tech Stack

| Layer      | Technology                                             |
| ---------- | ------------------------------------------------------ |
| Frontend   | React (Create React App), Material UI, Chart.js        |
| Backend    | Python, FastAPI, Uvicorn                               |
| ML / Data  | scikit-learn (Linear Regression), pandas, NumPy        |
| Charts     | Matplotlib (server-side), @mui/x-charts (client-side)  |
| Data I/O   | openpyxl / SheetJS (xlsx)                              |

---

## Architecture

```
┌─────────────────────────┐         ┌──────────────────────────────┐
│   React Frontend         │         │   FastAPI Backend            │
│   (Food_waste_app/)      │         │   (main.py)                  │
│                          │  HTTP   │                              │
│  • Home                  │ ──────▶ │  • /upload_excel/  (train)   │
│  • Train Model (upload)  │         │  • /predict-file/  (batch)   │
│  • Prediction (form)     │ ◀────── │  • /predict/       (single)  │
│                          │  JSON   │  • /admin/train_full_dataset/│
└─────────────────────────┘         └──────────────┬───────────────┘
                                                    │
                                         ┌──────────▼───────────┐
                                         │  LinearRegression    │
                                         │  (multi-output:      │
                                         │   lunch + snacks)    │
                                         └──────────────────────┘
```

The model is **multi-output** — it predicts lunch and snacks demand simultaneously from
features like footfall (turnstile headcount), previous-day orders, additional orders, and day of week.

---

## Project Structure

```
Semi-colons/
├── main.py                     # FastAPI backend + ML model
├── requirements.txt            # Python dependencies
├── data_files/
│   ├── data.xlsx               # Historical canteen data (Pune, 2023–25)
│   ├── processed_data.xlsx     # Cleaned dataset
│   └── presentation.txt        # Notes on data-quality issues / future scope
├── Food_waste_app/             # React frontend
│   ├── src/
│   │   ├── App.js              # Tab layout (Home / Train / Predict)
│   │   └── Components/
│   │       ├── Home.js
│   │       ├── FileUpload.js
│   │       ├── OrderForm.js
│   │       └── BarChartComponent.js
│   └── package.json
├── Semicolon /             # Pitch deck
└── LICENSE                     # Apache-2.0
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js 16+ and npm

### 1. Backend (FastAPI)

```bash
# From the repo root
pip install -r requirements.txt

# Start the API (default: http://localhost:8000)
uvicorn main:app --reload
```

Interactive API docs are available at `http://localhost:8000/docs`.

> **Note:** `main.py` imports `seaborn` and `matplotlib`. If they aren't already installed,
> add them: `pip install matplotlib seaborn`.

### 2. Frontend (React)

```bash
cd Food_waste_app
npm install
npm start        # opens http://localhost:3000
```

---

## API Endpoints

| Method | Endpoint                        | Description                                                        |
| ------ | ------------------------------- | ----------------------------------------------------------------- |
| POST   | `/upload_excel/`                | Upload Excel data, train the model, return predictions + heatmap. |
| POST   | `/predict-file/`                | Predict demand for an uploaded file, grouped by facility.         |
| POST   | `/predict/`                     | Predict for a single manual input.                                |
| POST   | `/admin/train_full_dataset/`    | Admin-only: train on the raw dataset (no holiday/Sunday filter).  |

Both `/upload_excel/` and `/predict-file/` expect a file upload plus a `sheet_name` form field.

---

## Data Format

The input Excel is expected to contain columns such as:

`Facility`, `Date`, `Day`, `Access Control Data (Footfall)`, `Lunch Ordered Previous Day`,
`Additional Order`, `Lunch Actual`, `Snacks Ordered Previous Day`, `Snacks Actual`, `Menu`, `Remarks`, `Justification`.

During preprocessing the backend:

- Parses dates and **drops Sundays** (canteen closed).
- **Removes holidays** flagged in the Remarks column.
- Converts order/footfall columns to numeric and encodes categorical fields (day, facility, menu).

---

## Future Scope

From the team's data-quality notes (`data_files/presentation.txt`):

- **Standardize menu naming** — items like *Varan* / *Dal fry* appear in several spellings.
- **Capture per-item counts** — e.g. watermelon demand spikes in summer but isn't tracked individually.
- **Improve data clarity** — record the actual event behind "low turnout", justifications for fasting days, etc.
- **Add external signals** — weather / rain advisories and special occasions, consistently across locations.
- **Model upgrades** — try tree-based models (Random Forest / XGBoost) for seasonal, non-linear demand, and add proper cross-validation.

---

## License

Licensed under the **Apache License 2.0**. See [LICENSE](./LICENSE) for details.
