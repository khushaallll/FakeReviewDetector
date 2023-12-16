#LIBRARIES
import streamlit as st
import pickle
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import  PorterStemmer 
import re


#LOAD PICKLE FILES
model = pickle.load(open('data and pickle files/ensemble_model.pkl','rb')) 
vectorizer = pickle.load(open('data and pickle files/count_vectorizer.pkl','rb')) 

#FOR STREAMLIT
nltk.download('stopwords')

#TEXT PREPROCESSING
sw = set(stopwords.words('english'))
def text_preprocessing(text):
    txt = TextBlob(text)
    result = txt.correct()
    removed_special_characters = re.sub("[^a-zA-Z]", " ", str(result))
    tokens = removed_special_characters.lower().split()
    stemmer = PorterStemmer()
    
    cleaned = []
    stemmed = []
    
    for token in tokens:
        if token not in sw:
            cleaned.append(token)
            
    for token in cleaned:
        token = stemmer.stem(token)
        stemmed.append(token)

    return " ".join(stemmed)

#TEXT CLASSIFICATION
def text_classification(text):
    if len(text) < 1:
        st.write("  ")
    else:
        with st.spinner("Classification in progress..."):
            cleaned_review = text_preprocessing(text)
            process = vectorizer.transform([cleaned_review]).toarray()
            prediction = model.predict(process)
            print(prediction)
            p = ''.join(str(i) for i in prediction)
        
            if p == 'True':
                st.success("The review entered is Legitimate.")
            if p == 'False':
                st.error("The review entered is Fraudulent.")

#PAGE FORMATTING AND APPLICATION
def main():
    st.title("Fake Review Detection System")
    
    
    # --EXPANDERS--    
    abstract = st.expander("Abstract")
    if abstract:
        abstract.write("As the Internet continues to grow in size and importance, the number and impact of online reviews is increasing continuously. Reviews can influence people across a large range of industries, but they are particularly important in e-commerce, where comments and reviews on products and services are often the foremost convenient, if not the sole way for a buyer to come to a decision whether to shop for them. As such, the credibility of online reviews is crucial for businesses and can directly affect a company s reputation and profitability. That is why some businesses are paying spammers to post fake reviews. These fake reviews exploit consumer purchasing decisions. Fake review detection has attracted considerable attention in recent years. Most review sites, however, still do not filter fake reviews publicly. We aim to develop a system to identify fake reviews. We propose a Machine Learning approach to identify fake reviews. We plan on the features extraction process of the reviews, and apply several features engineering to extract various behaviors of the reviewers. We will be using Yelp dataset of restaurant reviews with and without features extracted from users behaviors. Classifiers like KNN, Random Forest, tend to show prominent results in similar kinds of problems and we intend to explore them as well as a few others. This research aims to detect fake reviews for a product by using the text and rating property from a review. In short, the proposed system will measure the honesty value of a review, the trustiness value of the reviewers and the reliability value of a product and hence will come to a conclusion.")
        #st.write(abstract)
    
    links = st.expander("Related Links")
    if links:
        links.write("[Dataset utilized](https://www.kaggle.com/akudnaver/amazon-reviews-dataset)")
        # links.write("[Github](https://github.com/kntb0107/Fraud-Detection-in-Online-Consumer-Reviews-Using-Machine-Learning-Techniques)")
        
    # --CHECKBOXES--
    st.subheader("Information on the Classifier")
    if st.checkbox("About Classifer"):
        st.markdown('**Model:** Logistic Regression')
        st.markdown('**Vectorizer:** Count')
        st.markdown('**Test-Train splitting:** 40% - 60%')
        st.markdown('**Spelling Correction Library:** TextBlob')
        st.markdown('**Stemmer:** PorterStemmer')
        
    if st.checkbox("Evaluation Results"):
        st.markdown('**Accuracy:** 85%')
        st.markdown('**Precision:** 80%')
        st.markdown('**Recall:** 92%')
        st.markdown('**F-1 Score:** 85%')


    #--IMPLEMENTATION OF THE CLASSIFIER--
    st.subheader("Fake Review Classifier")
    review = st.text_area("Enter Review: ")
    if st.button("Check"):
        text_classification(review)

#RUN MAIN        
main()
