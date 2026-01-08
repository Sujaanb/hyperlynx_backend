from util.llm_factory import LLMFactory
from util.system_prompt import prompt_generate_summary, prompt_rag_compliance_analysis
from rag_service import get_rag_service
import re
import logging
from typing import Optional

# Configure logging
logger = logging.getLogger(__name__)


def generate_summary(text: str, local_llm=False, document_content: Optional[str] = None) -> Optional[str]:
    """
    Generates a comprehensive response using RAG pipeline.

    This function:
    1. Uses the RAG service to retrieve relevant compliance documents from Chroma Cloud
    2. If document_content is provided, finds similar compliance documents for comparison
    3. Builds context from retrieved documents and user content
    4. Uses the LLM to generate a comprehensive response

    Args:
        text: The input question/query from the user
        local_llm: Whether to use local LLM
        document_content: Optional document content from uploaded file

    Returns:
        The generated response or None if generation fails
    """
    if not text or not text.strip():
        return None

    try:
        # Get the RAG service
        rag_service = get_rag_service()
        
        # Build RAG context from Chroma Cloud
        logger.info("Building RAG context from Chroma Cloud...")
        context, sources = rag_service.build_rag_context(
            question=text,
            uploaded_document_content=document_content,
            top_k=5
        )
        
        # Determine which prompt to use
        if document_content or sources:
            # Use RAG compliance analysis prompt when we have context
            system_prompt = prompt_rag_compliance_analysis
            human_message = f"""
Based on the following context, please answer the user's question.

{context}

---
USER QUESTION: {text}
"""
        else:
            # Fall back to general prompt if no context retrieved
            system_prompt = prompt_generate_summary
            human_message = text
        
        logger.info(f"Generating response with {len(sources)} source documents...")
        
        # Generate response using LLM
        response = LLMFactory.invoke(
            system_prompt=system_prompt,
            human_message=human_message,
            temperature=0.7,
            local_llm=local_llm,
        )
        summary = response.content.strip()
        
        # Add source references if available
        if sources:
            source_list = "\n".join([f"- {src}" for src in set(sources)])
            summary += f"\n\n---\n**Sources Referenced:**\n{source_list}"
        
        # Replace citation numbers with markdown links if available
        if hasattr(response, 'additional_kwargs') and 'citations' in response.additional_kwargs:
            citations = response.additional_kwargs['citations']
            for i, citation in enumerate(citations, 1):
                summary = re.sub(f'\\[{i}\\]', f'[{i}]({citation})', summary)
        
        logger.info("Response generated successfully")
        return summary
        
    except Exception as e:
        logger.error(f"Error generating summary with RAG: {e}")
        # Log the full traceback for debugging
        import traceback
        logger.error(traceback.format_exc())
        return None