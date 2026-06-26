from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os

def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.environ.get("MISTRAL_API_KEY"),
        temperature=0.3,
    )

def split_transcript(transcript:str)->list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )
    return splitter.split_text(transcript)

def summarize(transcript:str)->str:
    llm = get_llm()
    map_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that summarizes transcripts."),
            (
                "human",
                "Summarize the following transcript in a concise manner, highlighting key points and important details:\n\n{transcript}",
            )
        ]
    )
    map_chain = map_prompt | llm | StrOutputParser()
    chunks = split_transcript(transcript)
    chunk_summaries = [map_chain.invoke({'transcript': chunk}) for chunk in chunks]
    combined = "\n\n".join(chunk_summaries)
    combined_prompt = ChatPromptTemplate.from_messages([(
        "system", "You are an expert summarizer that combines multiple summaries into a cohesive final professional video summary in bullet points."
    ),(
        "human","{text}"
    )])
    combined_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x:{"text":x})| combined_prompt | llm | StrOutputParser()
    )
    return combined_chain.invoke(combined)

def generate_title(transcript:str)-> str:
    llm = get_llm()
    title_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that generates titles for video summaries (max 10 words) Only return the title, nothing else."),
            ("human", "Generate a concise and engaging title for the following transcript:\n\n{text}")
        ])
        | llm | StrOutputParser()
    )
    print("Transcript type:", type(transcript))
    print("Transcript length:", len(transcript) if transcript else 0)
    print("Transcript preview:", str(transcript)[:200])
    result =  title_chain.invoke(transcript[:2000])
    print("Generated title:", result)
    return result