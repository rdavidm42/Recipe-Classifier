import streamlit as st
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from recipe_preprocessing import preprocess_recipe

def categorize_recipe(name, ingredients, instructions):
    with open("logistic_recipes.pkl", "rb") as f:
        model = load(f)
    with open("mlb.pkl", "rb") as f:
        mlb = load(f)
    cleaned_text = preprocess_recipe(name, ingredients, instructions)
    return mlb.inverse_transform(model.predict([cleaned_text]))

def main():
    st.title("Recipe Categorizer")
    st.write("Upload a recipe and let the model categorize it for you!")
    recipe_name = st.text_input('Recipe Name')
    recipe_ingredients = st.text_area('Recipe Ingredients (one per line)')
    recipe_instructions = st.text_area('Recipe Instructions')
    if st.button('Categorize Recipe'):
        if recipe_name and recipe_ingredients and recipe_instructions:
            # Here you would call your model to categorize the recipe
            category = list(categorize_recipe(recipe_name, recipe_ingredients, recipe_instructions))
            st.success(f'The recipe "{recipe_name}" has been categorized as: ' + ', '.join(category[0]).title())
        else:
            st.error('Please fill in all fields before categorizing.')

if __name__ == "__main__":
    main()