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

def categorize_recipe_svc(name, ingredients, instructions, model, mlb):
    """
    Categorizes recipe with the SVC model. First loads in the trained SVC model and the MultiLabelBinarizer (MLB).
    Text is then cleaned and combined, then the SVC model predicts the tags. 
    The MLB is used to transform the binarized labels back into human readable tags.

    Parameters
    ----------
    name : str
        Recipe title.
    ingredients : str
        Ingredients used in recipe.
    instructions : str
        Instructions to make recipe.
    model : sklearn.pipeline.Pipeline
        SVC model used to predict the recipe tags
    mlb : sklearn.preprocessing._label.MultiLabelBinarizer
        MultiLabelBinarizer used to get label names
    """
    cleaned_text = preprocess_recipe(name, ingredients, instructions)
    return mlb.inverse_transform(model.predict([cleaned_text]))

def feature_importance(term,model,mlb):
    """
    Getting feature importance for each tag. SVC and MLB are loaded, 
    then 25 words with highest feature importantance are obtained.
    A wordcloud function of these 25 words is then returned.

    Parameters
    ----------
    term : str
       Tag that function gets feature importance for.
    model : sklearn.pipeline.Pipeline
        SVC model used to predict the recipe tags
    mlb : sklearn.preprocessing._label.MultiLabelBinarizer
        MultiLabelBinarizer used to get label names
    """
    clfs = np.array(model[1].estimators_)

    feat_impts = clfs[mlb.classes_==term][0].coef_.flatten()
    features=dict(zip(model[0].get_feature_names_out(),feat_impts))

    wordcloud = WordCloud(relative_scaling=0,background_color='white').generate(' '.join(nlargest(25,features, key=features.get)))
    return wordcloud

@st.cache_resource
def categorize_recipe_distilbert(name, ingredients, instructions,mlb):
    """
    Categorizes recipe with the DistilBERT model. First loads in the trained DistilBERT model
    and the MultiLabelBinarizer (MLB).Text is then cleaned and combined, then the SVC model predicts the tags. 
    The MLB is used to transform the binarized labels back into human readable tags.

    Parameters
    ----------
    name : str
        Recipe title.
    ingredients : str
        Ingredients used in recipe.
    instructions : str
        Instructions to make recipe.
    model : sklearn.pipeline.Pipeline
        SVC model used to predict the recipe tags
    mlb : sklearn.preprocessing._label.MultiLabelBinarizer
        MultiLabelBinarizer used to get label names
    """
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
    
    return mlb.inverse_transform((logits>0).numpy().astype(int))
def main():
    """
    Main function to initiate the streamlit app. There are two tabs, 
    the first is used to categorize the recipes with either the SVC model or the DistilBERT model. 
    The second tab is used to explore feature importance in the SVC model.
    """
    # Define the SVC model and the MLB, save to session state
    if 'svc_model' not in st.session_state:
        with open("svc_recipes.pkl", "rb") as f:
            st.session_state.svc_model = load(f)
    if 'mlb' not in st.session_state:
        with open("mlb.pkl", "rb") as f:
            st.session_state.mlb = load(f)
    st.title("Recipe Categorizer")
    #Create two tabs - one for the classifier and one for exploring feature importance
    tab1, tab2 = st.tabs(["Classifier", "Explore the SVC Model"])
    with tab1:
        st.write("Upload a recipe and let the model categorize it for you!")

        # Gets the recipe information from the user
        recipe_name = st.text_input('Recipe Name')
        recipe_ingredients = st.text_area('Recipe Ingredients (one per line)')
        recipe_instructions = st.text_area('Recipe Instructions')

        # Choose the SVC or DistilBERT classifier
        classifier = st.radio('Use',['Support Vector Classifier','Distilbert Neural Network'], horizontal=True)
        if st.button('Categorize Recipe'):
            if recipe_name and recipe_ingredients and recipe_instructions:
                if classifier == 'Support Vector Classifier':
                    category = list(categorize_recipe_svc(recipe_name, recipe_ingredients, recipe_instructions,st.session_state.svc_model,st.session_state.mlb))
                else:
                    category = list(categorize_recipe_distilbert(recipe_name, recipe_ingredients, recipe_instructions,st.session_state.mlb))
                
                st.success(f'The recipe "{recipe_name}" has been categorized as: ' + ', '.join(category[0]).title().replace('-',' '))
            else:
                st.error('Please fill in all fields before categorizing.')

    # Feature Importance Tag
    with tab2:
        st.write("Explore what the SVC model thinks are important words in each category")
        # Get list of all tags
        tags = [x.title().replace('-',' ') for x in st.session_state.mlb.classes_]
        selection = st.selectbox('Categories',tags)

        # Create Wordcloud from the feature importances
        wordcloud = feature_importance(selection,st.session_state.svc_model,st.session_state.mlb)
        fig = plt.figure()
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.margins(x=0, y=0)
        st.pyplot(fig)

if __name__ == "__main__":
    main()