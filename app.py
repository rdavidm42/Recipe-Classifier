import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from heapq import nlargest
from recipe_preprocessing import preprocess_recipe
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from wordcloud import WordCloud

def categorize_recipe_svc(name, ingredients, instructions):
    with open("svc_recipes.pkl", "rb") as f:
        model = load(f)
    with open("mlb.pkl", "rb") as f:
        mlb = load(f)
    cleaned_text = preprocess_recipe(name, ingredients, instructions)
    return mlb.inverse_transform(model.predict([cleaned_text]))

def feature_importance(term):
        with open("svc_recipes.pkl", "rb") as f:
            model = load(f)
        with open("mlb.pkl", "rb") as f:
            mlb = load(f)
        feat_impts = [] 
        for clf in model[1].estimators_:
            feat_impts.append(clf.coef_)

        classes_dictionary = dict(zip(mlb.classes_,np.arange(len(mlb.classes_))))
        i = classes_dictionary[term.lower().replace(' ','-')]

        features=dict(zip(model[0].get_feature_names_out(),feat_impts[i].flatten()))
        # print(mlb.classes_[i])

        wordcloud = WordCloud(relative_scaling=0,background_color='white').generate(' '.join(nlargest(25,features, key=features.get)))
        return wordcloud
@st.cache_resource
def categorize_recipe_distilbert(name, ingredients, instructions):
    text = preprocess_recipe(name, ingredients, instructions)
    path = 'lora_weights';
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
    tab1, tab2 = st.tabs(["Classifier", "Explore the Model"])
    with tab1:
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
    with tab2:
        st.write("Explore what the model thinks are important words in each category")
        with open("mlb.pkl", "rb") as f:
            mlb = load(f)
        tags = [x.title().replace('-',' ') for x in mlb.classes_]
        selection = st.selectbox('Categories',tags)
        # if st.button('Get Model Importance'):
        wordcloud = feature_importance(selection)
        fig = plt.figure()
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.margins(x=0, y=0)
        st.pyplot(fig)

if __name__ == "__main__":
    main()