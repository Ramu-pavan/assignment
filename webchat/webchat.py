import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Initialize session state for conversation history
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

if 'user_input' not in st.session_state:
    st.session_state.user_input = ""  # Initialize input storage



def fetch_web_text(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract text from the page (you can adjust the extraction logic if needed)
            text = soup.get_text(separator=" ", strip=True)
            return text
        else:
            st.error(f"Failed to fetch content from the URL. Status code: {response.status_code}")
            return ""
    except Exception as e:
        st.error(f"Error fetching the URL: {e}")
        return ""


# FUNCTION TO DIVIDE THE TEXT INTO SMALLER CHUNKS
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


# GET TEXT EMBEDDINGS USING GEMINI MODEL & PREPARING THE VECTOR DATABASE
def get_embeddings(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectors = FAISS.from_texts(chunks, embedding=embeddings)
    vectors.save_local("faiss_index")



# PREPARING THE CONVERSATIONAL CHAIN
def get_conversational_chain():
    prompt_temp = '''
    Answer the question from the provided context. Try to answer in as detailed manner as possible from the provided context.
    If the answer to the question is not known from the provided context, then don't provide wrong answers, in that case just say,
    'Answer to the question is not available in the provided document. Feel free to ask question from the provided context.'
    Context:\n{context}?\n
    Question:\n{question}\n
    '''
    prompt = PromptTemplate(
        template=prompt_temp,
        input_variables=['context', 'question']
    )

    model = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.5)

    chain = load_qa_chain(model, chain_type='stuff', prompt=prompt)
    return chain


# PREPARING THE MODEL'S RESPONSE AND MAINTAINING CONVERSATION HISTORY
def get_response(user_input):
    embedding = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    new_db = FAISS.load_local('faiss_index', embedding, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_input)
    chain = get_conversational_chain()
    response = chain(
        {'input_documents': docs, 'question': user_input},
        return_only_outputs=True
    )

    # Append the user's question and the bot's response to conversation history
    st.session_state.conversation.append({
        "user": user_input,
        "bot": response['output_text']
    })


# CREATING THE FRONT END APPLICATION
def main():
    st.title('Chat With Web Pages Using Gemini 🤖')

    # Create a container for the input at the top
    with st.container():
        st.session_state.user_input = st.text_input(
            label='Ask a question related to the web page.',
            label_visibility='hidden',
            placeholder='Type your question here...',
            value=st.session_state.user_input
        )

        if st.session_state.user_input and st.button("Send"):
            # Process the user's query and get response
            get_response(st.session_state.user_input)
            st.session_state.user_input = ""  # Clear input after the query is processed
            st.rerun()  # Re-run the app to update the conversation history

    # Display conversation history (Scrollable)
    chat_container = st.container()
    with chat_container:
        if st.session_state.conversation:
            for msg in st.session_state.conversation:
                # Display user's question
                st.chat_message("user").write(f"*User*: {msg['user']}")
                # Display bot's reply
                st.chat_message("bot").write(f"*Bot*: {msg['bot']}")
        else:
            st.write("No conversation yet.")

    with st.sidebar:
        st.title('Provide Web URL 🌐')
        url = st.text_input(
            label='Enter a URL to fetch content from.',
            label_visibility='hidden',
            placeholder='Enter URL here...'
        )
        if st.button('Submit and Process'):
            with st.spinner('In Process...'):
                # Fetch text from the URL
                text = fetch_web_text(url)
                if text:
                    chunks = get_text_chunks(text)
                    get_embeddings(chunks)
                    st.success('DONE!')


if __name__ == '__main__':
    main()
