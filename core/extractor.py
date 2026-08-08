#Actionableitems , decision , questions 

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os 


def get_llm():
    return ChatMistralAI(model = "mistral-small-latest", mistral_api_key = os.getenv("MISTRAL_API_KEY"),temperature=0.2)


def _split_transcript(transcript: str) -> list:
    """Split long transcripts into manageable chunks for the LLM."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )
    return splitter.split_text(transcript)


def _extract_with_map_reduce(transcript: str, chunk_prompt: str, combine_prompt: str) -> str:
    """
    Map-reduce extraction: extract from each chunk, then combine results.
    This prevents token limit issues with long transcripts.
    """
    llm = get_llm()
    
    # Map phase: extract from each chunk
    map_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            ("system", chunk_prompt),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )

    chunks = _split_transcript(transcript)
    print(f"  Extracting from {len(chunks)} chunk(s)...")
    
    chunk_results = []
    for i, chunk in enumerate(chunks):
        result = map_chain.invoke(chunk)
        # Only keep results that actually found something
        if result.strip() and "none" not in result.strip().lower()[:30]:
            chunk_results.append(result.strip())

    if not chunk_results:
        return None  # Signal that nothing was found

    # Reduce phase: combine all chunk results
    combined = "\n\n".join(chunk_results)
    
    combine_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            ("system", combine_prompt),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )

    return combine_chain.invoke(combined)


def extract_action_items(transcript: str) -> str:
    chunk_prompt = (
        "You are an expert content analyst. From the following transcript portion "
        "(which may be a meeting, lecture, tutorial, hearing, or any video), "
        "extract all action items, recommendations, tasks, or steps the audience should take. "
        "For each provide:\n"
        "- Task or recommendation description\n"
        "- Owner or audience (who should do it)\n"
        "- Deadline or priority (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. Be thorough - look for implicit actions too. "
        "If genuinely none exist in this portion, say 'None found in this section.'"
    )
    combine_prompt = (
        "You are an expert content analyst. Below are action items extracted from "
        "different portions of a transcript. Merge, deduplicate, and produce a single "
        "clean numbered list of all action items. Remove duplicates but keep all unique items. "
        "Do NOT include any preamble, introductory text, or headers - start directly with the numbered list."
    )
    result = _extract_with_map_reduce(transcript, chunk_prompt, combine_prompt)
    return result if result else "No action items found."


def extract_key_decisions(transcript: str) -> str:
    chunk_prompt = (
        "You are an expert content analyst. From the following transcript portion "
        "(which may be a meeting, lecture, tutorial, hearing, or any video), "
        "extract all key decisions, conclusions, important points established, "
        "proposals, rulings, agreements, or definitive statements made. "
        "Format as a numbered list. "
        "If genuinely none exist in this portion, say 'None found in this section.'"
    )
    combine_prompt = (
        "You are an expert content analyst. Below are key decisions/conclusions extracted "
        "from different portions of a transcript. Merge, deduplicate, and produce a single "
        "clean numbered list. Remove duplicates but keep all unique items. "
        "Do NOT include any preamble, introductory text, or headers - start directly with the numbered list."
    )
    result = _extract_with_map_reduce(transcript, chunk_prompt, combine_prompt)
    return result if result else "No key decisions found."


def extract_questions(transcript: str) -> str:
    chunk_prompt = (
        "You are an expert content analyst. From the following transcript portion "
        "(which may be a meeting, lecture, tutorial, hearing, or any video), "
        "extract all unresolved questions, open issues, topics needing follow-up, "
        "or questions raised by speakers/participants. "
        "Format as a numbered list. "
        "If genuinely none exist in this portion, say 'None found in this section.'"
    )
    combine_prompt = (
        "You are an expert content analyst. Below are open questions extracted "
        "from different portions of a transcript. Merge, deduplicate, and produce a single "
        "clean numbered list. Remove duplicates but keep all unique items. "
        "Do NOT include any preamble, introductory text, or headers - start directly with the numbered list."
    )
    result = _extract_with_map_reduce(transcript, chunk_prompt, combine_prompt)
    return result if result else "No open questions found."