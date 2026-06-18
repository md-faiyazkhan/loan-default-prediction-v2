import shap
import pandas as pd

feature_labels = {
    'no_of_dependents': 'number of dependents',
    'education': 'education level',
    'self_employed': 'self-employment status',
    'income_annum': 'annual income',
    'loan_amount': 'loan amount requested',
    'loan_term': 'loan term',
    'cibil_score': 'CIBIL score',
    'residential_assets_value': 'residential assets value',
    'commercial_assets_value': 'commercial assets value',
    'luxury_assets_value': 'luxury assets value',
    'bank_asset_value': 'bank assets value'
}

recommendation_map = {
    'cibil_score': 'Improving your CIBIL score could significantly increase your approval chances.',
    'loan_amount': 'Consider requesting a lower loan amount relative to your income and assets.',
    'income_annum': 'A higher annual income would strengthen your application.',
    'loan_term': 'Adjusting your loan term may improve your approval likelihood.',
    'no_of_dependents': 'A lower number of dependents relative to your income works in your favor.',
    'residential_assets_value': 'Having stronger residential assets could improve your profile.',
    'commercial_assets_value': 'Stronger commercial assets would support your application.',
    'luxury_assets_value': 'Your luxury assets value is currently working against your application.',
    'bank_asset_value': 'Higher bank assets would strengthen your financial profile.',
    'education': 'Your education status is a factor in this decision.',
    'self_employed': 'Your employment type is a factor in this decision.'
}

def get_explanation(model, input_df, feature_names, top_n=3):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    shap_row = shap_values[0]

    contributions = list(zip(feature_names, shap_row))
    positive = sorted([c for c in contributions if c[1] > 0], key=lambda x: x[1], reverse=True)[:top_n]
    negative = sorted([c for c in contributions if c[1] < 0], key=lambda x: x[1])[:top_n]

    return positive, negative

def explain_in_words(positive, negative):
    pos_text = [f"Your {feature_labels[f]} positively supported the decision" for f, v in positive]
    neg_text = [f"Your {feature_labels[f]} negatively impacted the decision" for f, v in negative]
    return pos_text, neg_text

def generate_recommendations(negative):
    return [recommendation_map[f] for f, v in negative]