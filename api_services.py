from fastapi import APIRouter, HTTPException, status, Form, File, UploadFile
from typing import Optional
import logging
import tempfile
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_router = APIRouter(tags=["HL API Services"])

from base_requests import GenerateContentRequest, GenerateContentResponse
from test_run import generate_summary
from util.utility import Utility


@api_router.post(
    "/generate",
    response_model=GenerateContentResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"description": "Invalid Question"},
        422: {"description": "Unprocessable Question"},
    },
)
async def generate_content(
    question: str = Form(...),
    local_llm: bool = Form(False),
    file: Optional[UploadFile] = File(None)
) -> GenerateContentResponse:
    temp_file_path = None
    try:
        logger.info("Generating content with Question")
        
        document_content = None
        if file:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                temp_file_path = temp_file.name
                content = await file.read()
                temp_file.write(content)
            
            # Read file content using utility function
            try:
                documents = Utility.read_file_content(temp_file_path)
                document_content = "\n".join([doc.page_content for doc in documents])
                document_content = Utility.clean_text(document_content, preserve_paragraphs=True)
            except Exception as e:
                logger.error(f"Error reading file content: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Error processing uploaded file: {str(e)}"
                )
        
        summary = generate_summary(
            text=question, 
            local_llm=local_llm, 
            document_content=document_content
        )
        
        if summary is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate content. Please check your input and try again."
            )

        return GenerateContentResponse(
            status="success", message="Content generated successfully", data=summary
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error generating content: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating content: {str(e)}",
        )


    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {temp_file_path}: {str(e)}")