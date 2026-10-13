# 🧬 LabMind-MCP

**Diagnostic Test Interpreter & Anomaly Detection MCP Server**

An [MCP (Model Context Protocol)](https://modelcontextprotocol.io) server that connects AI agents to Laboratory Information Systems (LIS) and clinical ML models — enabling intelligent lab result interpretation, anomaly detection, sepsis risk prediction, and temporal trend analysis.

---

## Architecture

```
 [ AI Agent / LLM Client ]         (Claude Desktop, custom agents, etc.)
          │
          │  JSON-RPC over Stdio / SSE
          ▼
   ┌───────────────────┐
   │   LabMind-MCP     │
   │     Server        │           6 Tools · 3 Resources · 2 Prompts
   └───────┬───────────┘
           │
 ┌─────────┼────────────────────────┬───────────────────────┐
 │         ▼                        ▼                       ▼
 │ [ ML Inference Engine ]   [ Database Layer ]      [ Config/LOINC ]
 │ • Isolation Forest        • SQLite LIS DB         • Reference ranges
 │ • XGBoost Classifier      • Patient records       • LOINC code directory
 │ • Trend Analyzer          • Lab result history    • Risk thresholds
 │ • SHAP Explainability     • Reference ranges
 └──────────────────────────────────────────────────────────────────
```

---

## Features

### 🔧 Tools (6)

| Tool | Description |
|------|-------------|
| `fetch_patient_lab_history` | Retrieve lab results with demographics and abnormality flags |
| `run_lab_anomaly_detector` | Isolation Forest multi-parameter anomaly detection |
| `predict_sepsis_risk` | XGBoost sepsis risk prediction with SHAP explainability |
| `analyze_lab_trend` | Temporal trend analysis with linear regression & projections |
| `search_patients` | Search patients by name or MRN |
| `get_critical_alerts` | Prioritized critical value alerts combining rules + ML |

### 📊 Resources (3)

| URI | Description |
|-----|-------------|
| `lis://reference-ranges` | Full reference ranges with gender-specific and critical thresholds |
| `lis://lab-codes` | LOINC code directory with test descriptions and categories |
| `lis://patient-roster` | Patient discovery list with IDs, names, and demographics |

### 💬 Prompts (2)

| Prompt | Description |
|--------|-------------|
| `clinical_review` | Comprehensive patient review workflow (fetch → detect → predict → trend) |
| `shift_handoff` | Multi-patient shift handoff report generator |

---

## Quick Start

### 1. Clone & Setup

```bash
cd labmind-mcp
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
pip install -e ".[dev]"
```

### 2. Seed the Database

Generates 60 synthetic patients with 6,000+ lab results, including sepsis and AKI trajectories:

```bash
python -m labmind.data.seed_database
```

### 3. Train ML Models

Trains the Isolation Forest anomaly detector and XGBoost sepsis classifier:

```bash
python -m labmind.ml.train_models
```

### 4. Run the Server

```bash
python -m labmind.server
```

---

## Claude Desktop Configuration

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "labmind": {
      "command": "python",
      "args": ["-m", "labmind.server"],
      "cwd": "/path/to/labmind-mcp",
      "env": {
        "PYTHONPATH": "/path/to/labmind-mcp"
      }
    }
  }
}
```

Or, if installed as a package:

```json
{
  "mcpServers": {
    "labmind": {
      "command": "labmind-mcp"
    }
  }
}
```

---

## Example Interactions

### Fetch Lab History
```
User: "Show me the latest labs for patient P001"
Agent calls: fetch_patient_lab_history(patient_id="P001")
→ Returns formatted table with demographics, lab values, reference ranges, and abnormality flags
```

### Anomaly Detection
```
User: "Are there any unusual patterns in P003's labs?"
Agent calls: run_lab_anomaly_detector(patient_id="P003")
→ Returns Isolation Forest analysis with z-scores and flagged parameters
```

### Sepsis Risk Prediction
```
User: "What's the sepsis risk for P001? HR=110, BP=85, Temp=38.9, RR=24"
Agent calls: predict_sepsis_risk(patient_id="P001", heart_rate=110, systolic_bp=85, temperature=38.9, respiratory_rate=24)
→ Returns probability, risk level, SHAP feature importances, and clinical recommendation
```

### Trend Analysis
```
User: "How has P005's creatinine been trending?"
Agent calls: analyze_lab_trend(patient_id="P005", test_code="2160-0")
→ Returns direction, slope, R², projected values, and clinical significance
```

---

## Technology Stack

| Layer | Tool / Library |
|-------|----------------|
| **MCP Protocol** | Python MCP SDK (`mcp`), FastMCP |
| **ML Models** | `scikit-learn` (Isolation Forest), `xgboost` (Sepsis prediction), `shap` (Explainability) |
| **Data Processing** | `pandas`, `numpy` |
| **Database** | `sqlite3` (Mock LIS database) |
| **Transport** | `stdio` (default) or `SSE` |

---

## Project Structure

```
labmind-mcp/
├── pyproject.toml              # Project configuration & dependencies
├── README.md                   # This file
├── .gitignore
└── labmind/
    ├── __init__.py             # Package init (version)
    ├── config.py               # Paths, hyperparams, LOINC codes, thresholds
    ├── server.py               # MCP Server (tools, resources, prompts)
    ├── data/
    │   ├── __init__.py
    │   ├── seed_database.py    # Synthetic data generator & DB seeder
    │   └── labmind.db          # SQLite database (generated)
    ├── db/
    │   ├── __init__.py
    │   └── database.py         # Typed database access layer
    └── ml/
        ├── __init__.py         # ML package exports
        ├── anomaly_detector.py # Isolation Forest anomaly detection
        ├── sepsis_predictor.py # XGBoost sepsis risk prediction
        ├── trend_analyzer.py   # Temporal trend analysis
        ├── train_models.py     # Model training pipeline
        └── models/             # Saved model artifacts (generated)
            ├── anomaly_detector.joblib
            └── sepsis_predictor.joblib
```

---

## ML Models

### Isolation Forest (Anomaly Detection)
- **Purpose:** Detect multi-parameter lab panel anomalies
- **Features:** WBC, Hemoglobin, Platelets, Creatinine, Lactate, BUN, Glucose, Potassium
- **Output:** Anomaly flag, score, z-scores per parameter, flagged outliers
- **Contamination:** 5% expected anomaly rate

### XGBoost (Sepsis Risk)
- **Purpose:** Predict sepsis probability from labs + vitals
- **Features:** 8 lab values + 4 vital signs (12 total)
- **Output:** Probability, risk level (LOW/MODERATE/HIGH/CRITICAL), SHAP explanations
- **Training:** 2,000 synthetic samples (75/25 split healthy/sepsis)

### Trend Analyzer (Linear Regression)
- **Purpose:** Assess temporal trajectories of lab values
- **Output:** Slope, direction, R², projections (7/14/30 days), clinical significance

---

## License

MIT
