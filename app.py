import streamlit as st
import time, torch, tokenizers
from zeroshot_bart import Model

@st.cache(hash_funcs={torch.nn.parameter.Parameter: hash, tokenizers.Tokenizer: hash, tokenizers.AddedToken: hash}, suppress_st_warning=True, allow_output_mutation=True)
def Net(ckpt):
    return Model(ckpt)

st.title("Model Card:")
ckpt = st.selectbox("Checkpoint", ("facebook/bart-large-mnli", "joeddav/bart-large-mnli-yahoo-answers", "roberta-large-mnli", "roberta-base"))

labels = st.text_input("Label Set").strip()    # input labels as csv
premise = st.text_input("Input Text")
hypothesis = st.text_input("Prompt")
if st.button("Predict"):
    if labels!="" and premise!="" and hypothesis!="":
        with st.spinner("Loading and Evaluating..."):
            labels = labels.split(',') 
            m = Net(ckpt)
            prediction, confidence = m(premise, hypothesis, labels)
            st.success("Output: " + str(prediction) + "  \nConfidence: " + str(confidence))
            
#             st.markdown('<p style="font-family:sans-serif; color:Red; font-size: 14px;">Output: </p>' + '<p style="font-family:sans-serif; color:White; font-size: 14px;">Test Accuracy: ' + str(prediction) + '</p>', unsafe_allow_html=True)

    else:
        st.error("Fields cannot be empty!")