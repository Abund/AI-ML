import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import os
from dotenv import load_dotenv, find_dotenv
import pinecone 

# Set your Pinecone API key
load_dotenv(find_dotenv(), override=True)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def delete_pinecone_index(index_name='all'):
    import pinecone
    st.write(print(f'deleting database'))
    pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))
    #pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV')) 

    if index_name == 'all':
        indexes = pinecone.list_indexes()
        for i in indexes:
            print(f'Deleting index {i} ...', end='')
            pinecone.delete_index(i)
            print('Ok')
    else:
        print(f'Deleting index {index_name} ...', end='')
        if pinecone.index_exists(index_name):
            pinecone.delete_index(index_name)
            print('Ok')
        else:
            print(f'Index {index_name} does not exist.')


# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    import os
    name, extension = os.path.splitext(file)
    st.write(print(f'loading docs'))
    # Call the delete_pinecone_index function before loading a new document
    delete_pinecone_index()
    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data

# splitting data in chunks
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    st.write(f'data chunked')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

# create embeddings using OpenAIEmbeddings() and save them in a Pinecone vector store
def create_embeddings(chunks):
    import pinecone
    from langchain.vectorstores import Pinecone
    embeddings = OpenAIEmbeddings()
    api_key = os.environ.get('PINECONE_API_KEY')
    environment = os.environ.get('PINECONE_ENV')
    index_name = "docs"

    # Initialize Pinecone
    vector_store = pinecone.init(api_key=api_key, environment=environment)

    # Check if the index exists
    if index_name not in pinecone.list_indexes():
        # If the index does not exist, create it
        print(f'Creating index {index_name} and embeddings ...', end='')
        pinecone.create_index(index_name, dimension=1536, metric='cosine', pods=1, pod_type='p1.x1')
        print('Done')

    # Index each chunk's embedding into Pinecone
    #for chunk in chunks:
        #vector = embeddings.encode(chunk.page_content)

        # Add the vector to the index
        #vector_store.insert(item_id=str(chunk.page_num), vector=vector, index_name=index_name)
    vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)

    return vector_store



def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.run(q)
    return answer

def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_tokens, total_tokens / 1000 * 0.0004

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

if __name__ == "__main__":
    st.image('img.png')
    st.subheader('LLM Question-Answering Application 🤖')
    with st.sidebar:
        api_key = st.text_input('OpenAI API Key:', type='password')
        if api_key:
            st.write('checking keys')
            os.environ['OPENAI_API_KEY'] = api_key

        
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)
        add_data = st.button('Add Data', on_click=clear_history)
        
        if uploaded_file and add_data:
            st.write('got to here3')
            with st.spinner('Reading, chunking and embedding file ...'):
                try:
                    bytes_data = uploaded_file.read()
                    file_name = os.path.join('./', uploaded_file.name)
                    with open(file_name, 'wb') as f:
                        f.write(bytes_data)

                    st.write('File read successfully.')

                    data = load_document(file_name)
                    if data is None:
                        st.error('Error loading the document. Please make sure the document format is supported.')
                        st.stop()

                    st.write('Document loaded successfully.')

                    chunks = chunk_data(data, chunk_size=chunk_size)
                    st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                    tokens, embedding_cost = calculate_embedding_cost(chunks)
                    st.write(f'Embedding cost: ${embedding_cost:.4f}')

                    vector_store = create_embeddings(chunks)
                    st.session_state.vs = vector_store
                    st.success('File uploaded, chunked and embedded successfully.')

                except Exception as e:
                    st.error(f'An error occurred: {str(e)}')
                    
                    st.stop()
        
    q = st.text_input('Ask a question about the content of your file:')
    if q:
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            st.write(f'k: {k}')
            answer = ask_and_get_answer(vector_store, q, k)
            st.text_area('LLM Answer: ', value=answer)
            st.divider()

            if 'history' not in st.session_state:
                st.session_state.history = ''

            value = f'Q: {q} \nA: {answer}'
            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history
            st.text_area(label='Chat History', value=h, key='history', height=400)
