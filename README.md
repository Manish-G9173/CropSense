# CropSense

CropSense is a small Streamlit application with two workflows: crop recommendations from soil nutrients and climate, and crop-yield estimation from historical production records. The crop recommender reports held-out accuracy, a confusion matrix, crop counts, and feature importance.

The sidebar language selector switches the dashboard between English and Hindi, including input guidance, model charts, and localized crop names. A short bilingual Soil Health Card guide explains which card values to enter and which local weather values to provide separately.

## Dataset setup

The Kaggle CSV is kept out of Git. A local copy is currently in `data/Crop_recommendation.csv`. To set up another checkout, download the dataset from [Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset), extract `Crop_recommendation.csv`, and place it here:

```text
data/Crop_recommendation.csv
```

The expected columns are `N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`, and `label`. CropSense will show a setup message until the CSV is available; it does not create sample training data or display made-up model metrics.

You can download with Kaggle CLI after configuring your Kaggle API token, or download the ZIP from the dataset page in a browser and extract the CSV into `data`.

Optional Kaggle CLI commands from PowerShell:

```powershell
python -m pip install kaggle
New-Item -ItemType Directory -Force data
kaggle datasets download -d atharvaingle/crop-recommendation-dataset -p data --unzip
```

## Install and run on Windows

Run these commands in PowerShell from the project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, open Command Prompt and run:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## Train/evaluate from the command line

The app trains and caches the model for the running session. To print a fresh held-out evaluation separately:

```powershell
python train_model.py
```

Or specify another CSV path:

```powershell
python train_model.py --dataset .\data\Crop_recommendation.csv
```

The fixed random seed is 42. The pipeline uses a stratified 80/20 split when the class counts permit it, median imputation fitted on training rows only, and a Random Forest classifier. The app maps `predict_proba` values using the fitted model's class order before selecting the top three.

To compare classifiers with stratified cross-validation, run:

```powershell
python train_model.py --cross-validate
```

This compares K-Nearest Neighbors, Decision Tree, Logistic Regression, Gaussian Naive Bayes, RBF SVM, and Random Forest with five folds by default. CV runs on the stratified 80% training split; the remaining 20% stays separate as a final test split. It reports mean and standard deviation for accuracy and macro-F1, each fold's accuracy, and test-set scores. Imputation and scaling are fitted inside each fold to avoid validation leakage. Change the fold count with `--cv-folds 3` (or another valid count).

## Yield estimation

The second workflow uses the provided `crop_yield.csv.zip` dataset (19,689 historical records, 1997–2020). It is available in this workspace as `data/crop_yield.csv`; the CSV and ZIP are excluded from Git. For another checkout, download the Kaggle [Agricultural Crop Yield in Indian States dataset](https://www.kaggle.com/datasets/akshatgupta7/crop-yield-in-indian-states-dataset) and place `crop_yield.csv` or `crop_yield.csv.zip` under `data`.

The target is `Yield`. Inputs are crop, year, season, state, area, annual rainfall, fertilizer, and pesticide. `Production` is excluded because it contains the harvested output and would leak information about yield. The model comparison includes Linear Regression, Ridge Regression, KNN Regressor, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor, and Linear SVR. Logistic Regression is a classifier, so the yield workflow uses numeric regression models instead.

Evaluation uses forward-chaining cross-validation on earlier years and keeps the latest years as a future holdout. After scoring, inference models are refit on all available records so predictions can use the full history. The leaderboard reports cross-validation and holdout MAE/RMSE in the dataset's own `Yield` scale; lower errors are better. To print a fresh comparison in PowerShell:

```powershell
python train_yield_model.py
```

Pass another CSV/ZIP with `python train_yield_model.py --dataset .\data\crop_yield.csv.zip`. The app sidebar switches between **Crop recommendation** and **Yield estimation**.

## Plant disease check

The third workflow uses the public Hugging Face model [`rarfileexe/Plant-Disease-Detector`](https://huggingface.co/rarfileexe/Plant-Disease-Detector), a MobileNetV2 CNN fine-tuned for 38 PlantVillage classes. The model card says it was trained on an augmented PlantVillage dataset and reports 98.75% test accuracy; that is the model author's reported result, not a metric independently reproduced in this app. The first diagnosis downloads the model into the Hugging Face cache and runs inference locally. The disease extra requires a network connection on first use and a CPU-compatible TensorFlow installation.

Install the optional dependencies in PowerShell:

```powershell
python -m pip install -r requirements-disease.txt
```

The uploader accepts JPG, JPEG, PNG, WEBP, and PDF up to 20 MB. For a PDF, only the first page is converted to an image and classified. Use a clear leaf photo; this model only has 38 PlantVillage labels covering 14 crops, and field photos can be misclassified.

Gemini care advice is optional. **The API key previously pasted into chat is exposed; revoke it and create a replacement before using Gemini.** Do not put the key in Python source or commit it. Copy the ignored local secrets file and add the replacement key:

```powershell
Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml
notepad .streamlit\secrets.toml
```

The file should contain `GEMINI_API_KEY = "your-new-key"`. The app supports Gemini 3.5 Flash-Lite, 3.8 Flash, and 3.6 Flash. If the selected model is unavailable, it tries the remaining requested models, then Gemini 3.7 Flash, 3.5 Flash, and 3.1 Flash-Lite. Gemini receives the classifier's crop, predicted class, and score—not the uploaded image. Advice avoids pesticide brands/doses and should be confirmed with local agricultural experts.

## Notes

- N, P, and K are treated as nonnegative values up to 300; temperature as -20 to 60 °C; humidity as 0–100%; pH as 0–14; and rainfall as 0–5000 mm. Out-of-range feature values are treated as missing and median-imputed from the training split. Rows without a crop label or any usable feature values are removed.
- Input values are bounded in the interface and predictions are based on the selected dataset's patterns.
- Class probabilities are model outputs, not calibrated guarantees of crop success or yield. Local climate, season, variety, irrigation, soil condition, and farm management also matter. Confirm planting decisions with local agricultural expertise.
- The confusion matrix is rendered with Plotly, so Matplotlib and Seaborn are not required.
