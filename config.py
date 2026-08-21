import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

CLASSIFIER_PATH = os.path.join(MODELS_DIR, "addiction_risk_classifier_v1.pkl")
REGRESSOR_PATH = os.path.join(MODELS_DIR, "addiction_risk_regressor_v1.pkl")
PIPELINE_PATH = os.path.join(MODELS_DIR, "preprocessing_pipeline_v1.pkl")
SCHEMA_PATH = os.path.join(MODELS_DIR, "feature_schema.json")

# Confirm this mapping matches your classifier's actual class order —
# check classifier.classes_ in Colab before deploying
RISK_TIER_LABELS = {0: "Low", 1: "Moderate", 2: "High"}

GENDER_OPTIONS = ["Female", "Male", "Other"]
SCHOOL_GRADE_OPTIONS = ["7th", "8th", "9th", "10th", "11th", "12th"]
PURPOSE_OPTIONS = ["Social Media", "Gaming", "Education", "Browsing", "Other"]