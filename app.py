import streamlit as st
import torch
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from recipe_preprocessing import preprocess_recipe
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

def categorize_recipe_svc(name, ingredients, instructions):
    with open("svc_recipes.pkl", "rb") as f:
        model = load(f)
    with open("mlb.pkl", "rb") as f:
        mlb = load(f)
    cleaned_text = preprocess_recipe(name, ingredients, instructions)
    return mlb.inverse_transform(model.predict([cleaned_text]))

@st.cache_resource
def categorize_recipe_distilbert(name, ingredients, instructions):
    text = preprocess_recipe(name, ingredients, instructions)
    path = '/Users/davidmayrhofer/Documents/Python/Machine Learning/Recipes Project/checkpoint-9480';
    base_model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name,
                                                           num_labels=56,
                                                           problem_type="multi_label_classification")
    model = PeftModel.from_pretrained(base_model, path)
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    with open("mlb.pkl", "rb") as f:
        mlb = load(f)
    return mlb.inverse_transform((logits>0).numpy().astype(int))
def main():
    st.title("Recipe Categorizer")
    st.write("Upload a recipe and let the model categorize it for you!")
    recipe_name = st.text_input('Recipe Name')
    recipe_ingredients = st.text_area('Recipe Ingredients (one per line)')
    recipe_instructions = st.text_area('Recipe Instructions')
    classifier = st.radio('Use',['Support Vector Classifier','Distilbert Neural Network'], horizontal=True)
    if st.button('Categorize Recipe'):
        if recipe_name and recipe_ingredients and recipe_instructions:
            # Here you would call your model to categorize the recipe
            if classifier == 'Support Vector Classifier':
                category = list(categorize_recipe_svc(recipe_name, recipe_ingredients, recipe_instructions))
            else:
                category = list(categorize_recipe_distilbert(recipe_name, recipe_ingredients, recipe_instructions))
            st.success(f'The recipe "{recipe_name}" has been categorized as: ' + ', '.join(category[0]).title().replace('-',' '))
        else:
            st.error('Please fill in all fields before categorizing.')

if __name__ == "__main__":
    main()