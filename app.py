import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from sentence_transformers import SentenceTransformer
#from sklearn.metrics.pairwise import cosine_similarity
import ollama

@st.cache_resource
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def create_embeddings(chunks):
    model = load_embedding_model()
    return model.encode(chunks)

st.title("AI Study Assistant")

if "notes" not in st.session_state:
    st.session_state.notes = ""

if "mcqs" not in st.session_state:
    st.session_state.mcqs = ""

uploaded_file = st.file_uploader("Upload your PDF", type= ["pdf"])
if uploaded_file is not None:
    st.write("File is uploaded")

    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        text += page.extract_text()

    st.write(text[:1000])

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)
    # for i, chunk in enumerate(chunks[:3]):
    #     st.write(f"chunk {i+1}:")
    #     st.write(chunk)

    #model = SentenceTransformer("all-MiniLM-L6-v2")
    model = load_embedding_model()
    embeddings = create_embeddings(chunks)
    #embeddings = model.encode(chunks)
    st.write(embeddings)
    # st.write(type(embeddings))
    # st.write(len(embeddings))
    # st.write(embeddings[0])
    question = st.text_input("Ask a question about the document:")
    
    if question:
        from sklearn.metrics.pairwise import cosine_similarity

        question_embedding = model.encode(question)
        similarities = cosine_similarity([question_embedding], embeddings)
        
        best_match_index = similarities.argmax()
        st.subheader("Answer:")
        st.write("Best Match Index:", best_match_index)
        st.write("Similarity Score:", similarities[0][best_match_index])
        # st.write(chunks[best_match_index])
        context = chunks[best_match_index]

        prompt = f"""
        You are an AI Study Assistant.
        Answer the user's question ONLY using the information below.

        Context:
        {context}

        Question:
        {question}

        Give a simple answer.
        """

        response = ollama.chat(
            model="qwen2.5:1.5b", 
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        st.subheader("Answer")
        st.write(response["message"]["content"])
        

    if st.button("Generate Notes"):
        st.write("Generating Notes...")

        prompt = f""" Convert the following text into short exam notes.

        Rules:
        - Use bullet points.
        - Keep the notes short.
        - Do not copy exactly.
        - Explain in simple language.
        - Also give minimum 10 points.

        Text:

        {context}
        """

        response = ollama.chat(
            model="qwen2.5:1.5b", 
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        ) 
      
        st.session_state.notes = (response["message"]["content"])

    if st.session_state.notes:
        st.subheader("Short Notes 📚")
        st.write(st.session_state.notes)


    if st.button("Generate MCQs"):
     st.write("Generating MCQs...")

     prompt = f""" 
     You are an AI teacher.

     create EXACTLY 5 multiple choice questions from the text below.

     IMPORTANT FORMATTING RULES:
     - Write each question on its own line.
     - NEVER put an option on the same line as the question.
     - A), B), C), and D) MUST each start on a NEW LINE.
     - Leave one blank line after every complete question.
     - Each question must have exactly 4 options.
     - Only ONE option can be correct.
     - Write the correct answer usng ONLY A, B, C, or D.
     - The correct answer MUST match one of the four options.
     - Do NOT write explanations.
     - Do NOT combine question and options on the same line.
     - Generate ALL 5 questions. Do not stop early.

     FOLLOW THIS EXACT FORMAT:

     1. Question goes here?
     A) Option 1
     B) Option 2
     C) Option 3
     D) Option 4
     Correct Answer: B

     2. Question goes here?
     A) Option 1
     B) Option 2
     C) Option 3
     D) Option 4
     Correct Answer: A

     Text:

     {context}
     """

     response = ollama.chat(
            model="qwen2.5:1.5b", 
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options = {
                "temperature": 0.3,
                "num_predict": 800
            }
     ) 

     st.session_state.mcqs = (response["message"]["content"])
    if st.session_state.mcqs:
        st.subheader("MCQs 📝") 
        st.write(st.session_state.mcqs)  