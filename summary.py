import streamlit as st
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, find_dotenv

# Set your Pinecone API key
load_dotenv(find_dotenv(), override=True)

# Function to summarize text using langchain
def summarize_text(text, chunk_size=10000):
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
    chunks = text_splitter.create_documents([text])

    chain = load_summarize_chain(llm, chain_type='map_reduce', verbose=False)
    output_summary = chain.run(chunks)

    return output_summary

# Streamlit app
def main():
    st.title("Document Summarizer")

    # Sidebar with file upload and chunk size control
    uploaded_file = st.sidebar.file_uploader("Upload a document", type=["txt", "pdf"])
    chunk_size = st.sidebar.number_input("Enter chunk size (default: 10,000)", value=10000)

    if st.sidebar.button("Summarize"):
        if uploaded_file is not None:
            # Read the file content
            # Read the content of the uploaded file
            with uploaded_file as f:
                text = f.read().decode('utf-8')

            # Summarize the text
            summary = summarize_text(text, chunk_size)

            # Display the summary
            st.subheader("Summary")
            st.write(summary)
        else:
            st.warning("Please upload a document first.")

if __name__ == "__main__":
    main()
