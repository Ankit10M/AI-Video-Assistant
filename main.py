from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_decisions, extract_questions
from core.rag_engine import build_rag_chain, answer_question

load_dotenv()

def run_pipeline(sourcestr: str, language: str = "en") -> dict:
    print("Video Assisting Pipeline Started...")
    # Process the input audio
    chunks = process_input(sourcestr)

    # Transcribe the audio
    transcription = transcribe_all(chunks, language)
    print("raw transcription(first 300 characters):", transcription[:300])

    # Generate a title
    title = generate_title(transcription)

    # Generate a summary
    summary = summarize(transcription)

    # Extract action items
    action_items = extract_action_items(transcription)

    # Extract decisions
    decisions = extract_decisions(transcription)

    # Extract questions
    questions = extract_questions(transcription)

    rag_chain = build_rag_chain(transcription)

    return {
        "title": title,
        "transcription": transcription,
        "summary": summary,
        "action_items": action_items,
        "decisions": decisions,
        "questions": questions,
        "rag_chain": rag_chain
    }
if __name__ == "__main__":
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Enter language (english/hinglish): ").strip() or "english"
    results = run_pipeline(source, language)
    print("\n" + "="*50)
    print(f"Title: {results['title']}\n")
    print(f"Summary: {results['summary']}\n")
    print(f"Action Items: {results['action_items']}\n")
    print(f"Decisions: {results['decisions']}\n")
    print(f"Questions: {results['questions']}\n")
    print("="*50)

    print("\n Chat with your meeting (type 'exit' to quit)\n")
    rag_chain = results["rag_chain"]
    while True:
        question = input("You: ").strip()
        if question.lower() in ['exit', 'quit','q']:
            print("Exiting chat. Goodbye!")
            break
        if not question:
            continue
        answer = answer_question(rag_chain, question)
        print(f"\nAI: {answer}\n")
