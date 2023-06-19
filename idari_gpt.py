from youtube_transcript_api import YouTubeTranscriptApi

# LangChain Dependencies
from langchain import ConversationChain, PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.llms import OpenAI
from langchain.memory import VectorStoreRetrieverMemory

# StreamLit Dependencies
import streamlit as st
from streamlit_chat import message

# Environment Dependencies
# from dotenv import load_dotenv
import os


def asd():
    loader = DirectoryLoader(
        path="./", glob="**/*.txt", loader_cls=TextLoader, show_progress=True
    )
    embeddings = OpenAIEmbeddings(
        openai_api_key="********************************"
    )
    index = VectorstoreIndexCreator(embedding=embeddings).from_loaders([loader])

    retriever = index.vectorstore.as_retriever(search_kwargs=dict(k=5))
    memory = VectorStoreRetrieverMemory(retriever=retriever)

    _DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

    Relevant pieces of previous conversation:
    {history}

    (You do not need to use these pieces of information if not relevant)

    Current conversation:
    Human: {input}
    AI:"""
    PROMPT = PromptTemplate(
        input_variables=["history", "input"], template=_DEFAULT_TEMPLATE
    )

    llm = OpenAI(
        temperature=0.7,
        openai_api_key="********************************",
    )  # Can be any valid LLM

    conversation_with_summary = ConversationChain(
        llm=llm,
        prompt=PROMPT,
        # We set a very low max_token_limit for the purposes of testing.
        memory=memory,
    )

    st.header("İdariYasaGPT")

    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []

    def get_text():
        input_text = st.text_input(
            "You: ",
            "Merhaba, İdari Usül kanunu hakkında bir sorum var.",
            key="input",
        )
        return input_text

    user_input = get_text()

    if user_input:
        output = conversation_with_summary.predict(input=user_input)

        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")


if __name__ == "__main__":
    asd()
