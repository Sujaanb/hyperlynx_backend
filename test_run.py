from util.llm_factory import LLMFactory
from util.system_prompt import prompt_generate_summary


from typing import Optional

def generate_summary(text: str, local_llm = False, document_content: Optional[str] = None) -> Optional[str]:
    """
    Generates a summary of the given text using the LLM.

    Args:
        text: The input text to summarize
        local_llm: Whether to use local LLM
        document_content: Optional document content to use as context

    Returns:
        The generated summary or None if generation fails
    """
    if not text or not text.strip():
        return None

    try:
        # Prepare the human message with document context if provided
        human_message = text
        if document_content:
            human_message = f"Document Context:\n{document_content}\n\nQuestion: {text}"
        
        response = LLMFactory.invoke(
            system_prompt=prompt_generate_summary,
            human_message=human_message,
            temperature=0.7,
            local_llm=local_llm,
        )
        summary = response.content.strip()
        return summary
    except Exception as e:
        # Log the error appropriately
        print(f"Error generating summary: {e}")
        return None