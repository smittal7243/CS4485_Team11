import os
import streamlit as st
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
import json


openai_api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(temperature=0.5, openai_api_key=openai_api_key)


template = (
    "Extract the food items, quantities, and customizations from this order: '{order_text}'. "
    "Return the result as JSON with 'name', 'quantity', and 'customizations' fields for each item."
)
prompt = PromptTemplate(input_variables=["order_text"], template=template)
llm_chain = LLMChain(llm=llm, prompt=prompt)


st.title("In-N-Out Order AI Agent")
st.write("Type your order below: ")

order_text = st.text_area("Enter your order:", placeholder="e.g., I'd like 2 cheeseburgers with extra cheese and 3 fries with no salt.")

if st.button("Analyze Order"):
    if order_text.strip():
        with st.spinner("Analyzing your order..."):
            try:
                
                result = llm_chain.run(order_text)
                
                
                st.subheader("Order Analysis")
                try:
                    result_json = json.loads(result)
                    st.json(result_json)  # Pretty-print the JSON
                except json.JSONDecodeError:
                    st.error("Failed to parse the result as JSON. Please refine your input.")
                    st.text(result)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter your order before clicking 'Analyze Order'.")