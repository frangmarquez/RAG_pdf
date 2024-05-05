import os
import pickle
import streamlit as st
from PyPDF2 import PdfReader
from streamlit_chat import message
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains.question_answering import load_qa_chain

# Content in the sidebar
with st.sidebar:
    st.title(" Chat with your pdf's!  ")
    st.markdown('''
    This app has been constructed using:
    - [streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [Hugging Face](https://huggingface.com/)

    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by Francisco Javier Marquez Gonzalez')


def main():
    st.header("Chat with your pdf's!")

    # to upload PDF file
    pdf = st.file_uploader("Upload your PDF file", type="pdf")

    if pdf:
        pdf_reader = PdfReader(pdf)

        text = "".join([page.extract_text() for page in pdf_reader.pages])

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                    chunk_overlap=200,
                                                    length_function=len
                                                    )
        chunks = text_splitter.split_text(text=text)

        # embeddings

        store_name = pdf.name.strip(".pdf")
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            st.write('Embeddings loaded')
        else:
            embeddings = HuggingFaceBgeEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
            st.write('Embeddings computed')

        if 'responses' not in st.session_state:
            st.session_state['responses'] = []
        if 'requests' not in st.session_state:
            st.session_state['requests'] = []

        # Starting chat
        responsecontainer = st.container(height=350, border=False)
        textcontainer = st.container()            
        
        with textcontainer:
            query = st.text_input("Enter your query: ",value=None,key="question")
            if query:
                with st.spinner("typing..."):
                    docs = VectorStore.similarity_search(query=query, k=3)
                    llm = Ollama(model="llama3",temperature=0)
                    chain = load_qa_chain(llm=llm, chain_type="stuff")
                    response = chain.run(input_documents=docs,question=query)
                st.session_state.requests.append(query)
                st.session_state.responses.append(response)
        with responsecontainer:
            if st.session_state['requests']:
                for i in range(len(st.session_state['requests'])):
                    message(st.session_state['requests'][i],is_user=True, key=str(i) + '_user')
                    if i < len(st.session_state['responses']):
                        message(st.session_state['responses'][i],key=str(i))    

if __name__ == '__main__':
    main()
