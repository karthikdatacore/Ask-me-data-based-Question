import streamlit as st
from google import genai
import pandas as pd

st.set_page_config(page_title="Ask a question!", layout="wide")
st.title("Ask a question!")
st.write("Upload your dataset")

with st.sidebar:
    st.header("Configuration")
    i_need_key=st.text_input("Enter your goddamn Key", type="password")

    st.header("Upload data")
    uploaded_file = st.file_uploader("Choose a file", type="csv")

if i_need_key and uploaded_file is not None:
    i_need_key=genai.Client(api_key=i_need_key)
    df=pd.read_csv(uploaded_file)

    tab1,tab2=st.tabs(
        ["Automated EDA Report","Conversational Data Chat"]
    )

    with tab1:
        st.subheader("Automated EDA Report")
        col1,col2,col3= st.columns(3)
        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())

        st.dataframe(df.head())

        if st.button("Generate Report"):
            with st.spinner("Generating report..."):
                buffer_summary=f"""Shape:{df.shape}
                                   Column Names: {df.dtypes.to_dict()}
                                   Missing Values: {df.isnull().sum().to_dict()}
                                   Statistical Summary: {df.describe().to_dict()}
                                   """
                eda_prompt=f"""You are a VERY EXPERIENCED DATA ANALYST. You are given a dataset with the following summary: {buffer_summary}
                Please provide a detailed exploratory data analysis (EDA) report based on this summary. Include insights, patterns, and any recommendations for further analysis.
                KEEP IT PROFESSIONAL AND DETAILED. DO NOT REPEAT THE SUMMARY. FOCUS ON INSIGHTS AND RECOMMENDATIONS."""

                response=i_need_key.models.generate_content(model="gemini-3.5-flash-lite",contents=eda_prompt)

                st.success("Report generated successfully!")
                st.write(response.content[0].text)

    with tab2:
        st.subheader("Conversational Data Chat")
        user_question = st.text_input("Ask a question about your dataset:")
        if user_question:
            with st.spinner("Generating response..."):
                chat_prompt=f"""You are a VERY EXPERIENCED DATA ANALYST. You are given a dataset with the following summary: {buffer_summary}
                A user has asked the following question: {user_question}
                Please provide a detailed and professional answer to the user's question based on the dataset summary. Include insights, patterns, and any recommendations for further analysis.
                KEEP IT PROFESSIONAL AND DETAILED. DO NOT REPEAT THE SUMMARY. FOCUS ON INSIGHTS AND RECOMMENDATIONS."""

                response=i_need_key.models.generate_content(model="gemini-3.1-pro",contents=chat_prompt)

                st.success("Response generated successfully!")
                st.write(response.content[0].text)

        try:
            response=i_need_key.models.generate_content(model="gemini-3.5-flash-lite",contents=chat_prompt)
            generated_code=(response.content[0].text.strip().replace("```python","").replace("```","").strip())

            st.text("Generated Python code:")
            st.code(generated_code, language="python")
            result=eval(generated_code)

            st.success("Code generated successfully!")
            st.metric(label="Result",value=result)

        except Exception as e:
            st.error(f"Error generating code: {e}") 
