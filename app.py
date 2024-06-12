import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

# 1. Vectorise the sales response csv data
loader = CSVLoader(file_path="final_startup_ideas_dataset.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["API_KEY"])
db = FAISS.from_documents(documents, embeddings)

# 2. Function for similarity search


def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)

    page_contents_array = [doc.page_content for doc in similar_response]

    # print(page_contents_array)

    return page_contents_array


# 3. Setup LLMChain & prompts
llm = ChatOpenAI(temperature=0, model="gpt-4o", api_key=st.secrets["API_KEY"])

template = """
You are a world class business idea innovator.
I will share a prospect's input with you and you will give me the best business idea
based on examples of business_ideas,
and you will follow ALL of the rules below:

1/ Response should be very similar or even identical to the past business_ideas,
in terms of length, tone of voice, logical arguments and other details

2/ Think of unique ideas that are fitting for the input. The business_ideas are meant as a guideline on
the kind of ideas prospects want and not to copy them.

3/ Answer in a simple way with a clear and revolutionizing idea.

Below is the user's input:
{user_input}

Here is a list of business_ideas of how you can convert a user_input into an idea that is useful:
{business_ideas}

Based on the user's input and the examples provided, please generate a unique and innovative business idea. 
Limit your response to 50-75 words of a clear, concise and most importantly, a useful idea.
"""



prompt = PromptTemplate(
    input_variables=["user_input", "business_ideas"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)


# 4. Retrieval augmented generation
def generate_response(user_input):
    business_ideas = retrieve_info(user_input)
    prompt_input = {
        "user_input": user_input,
        "business_ideas": "\n".join(business_ideas)
    }
    response = chain.invoke(input=prompt_input)
    idea = response['text'].split('Here is a list of business_ideas of how you can convert a user_input into an idea that is useful:')[0].strip()
    # Remove the 'idea:' prefix
    if idea.lower().startswith('idea:'):
        idea = idea[5:].strip()
    return idea





# 5. Build an app with streamlit
def main():
    st.set_page_config(
        page_title="Business Idea Generator", page_icon=":bulb:")

    st.header("Business Idea Generator :bulb:")
    message = st.text_area("Chat with StartSmart")

    if message:
        st.write("Generating best idea...")

        result = generate_response(message)

        st.info(result)


if __name__ == '__main__':
    main()
