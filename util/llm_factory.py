import os
import sys
from dotenv import load_dotenv
import re
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_perplexity import ChatPerplexity

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_openai import OpenAIEmbeddings

import util.constants as constants

class LLMFactory:

    @staticmethod
    def get_model_name():
        """
        Returns the model name based on the environment variable.
        """
        load_dotenv(override=True)
        env_name = os.getenv("llm_provider", "")

        model_mapping = {
            "mistral": constants.mistral_llm,
            "gemini": constants.gemini_llm,
            "openai": constants.openai_llm,
            "groq": constants.groq_llm,
            "perplexity": constants.perplexity_llm,
        }
        try:
            return model_mapping[env_name]
        except KeyError:
            raise ValueError(
                "Invalid environment name. Must be 'mistral', 'gemini', 'openai'."
            )

    @staticmethod
    def get_api_key():
        """
        Returns the API key based on the model/environment.
        """
        load_dotenv(override=True)
        env_name = os.getenv("llm_provider", "")

        api_key_mapping = {
            "mistral": os.getenv("mistral_api_key"),
            "gemini": os.getenv("gemini_api_key"),
            "openai": os.getenv("openai_api_key"),
            "groq": os.getenv("groq_api_key"),
            "perplexity": os.getenv("perplexity_api_key"),
        }

        try:
            return api_key_mapping[env_name]
        except KeyError:
            raise ValueError(
                "Invalid environment name. Must be 'mistral', 'gemini', or 'openai'."
            )

    @staticmethod
    def create_llm_instance(temperature=0.3, local_llm=False):
        """
        Creates and returns an instance of the appropriate LLM.
        """
        if local_llm:
            # If local_llm is True, use the local LLM instance.
            return ChatOllama(
                model=constants.local_llm,  # Use exact model name from 'ollama list'
                base_url=os.getenv("local_model_url"),  # Ollama server URL
            )
        model_name = LLMFactory.get_model_name()
        api_key = LLMFactory.get_api_key()

        if model_name == constants.mistral_llm:
            return ChatMistralAI(
                api_key=api_key, model_name=model_name, temperature=temperature
            )
        elif model_name == constants.openai_llm:
            return ChatOpenAI(
                api_key=api_key, model=model_name, temperature=temperature
            )
        elif model_name == constants.gemini_llm:
            return ChatGoogleGenerativeAI(
                api_key=api_key, model=model_name, temperature=temperature, transport="rest"
            )
        elif model_name == constants.groq_llm:
            return ChatGroq(
                api_key=api_key, model=model_name, temperature=temperature
            )
        elif model_name == constants.perplexity_llm:
            return ChatPerplexity(
                api_key=api_key, model=model_name, temperature=temperature
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    @staticmethod
    def invoke(
        system_prompt: str = None,
        human_message: str = None,
        temperature=0.3,
        local_llm=False
    ):
        """
        Invokes the LLM with given prompts using ChatPromptTemplate.
        """
        llm = LLMFactory.create_llm_instance(temperature, local_llm)

        if system_prompt and human_message:
            system_prompt = system_prompt.replace("{", "{{").replace("}", "}}")
            human_message = human_message.replace("{", "{{").replace("}", "}}")
            prompt_obj = ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(system_prompt),
                    HumanMessagePromptTemplate.from_template(human_message),
                ]
            )
            formatted_messages = prompt_obj.format_messages()
            return llm.invoke(formatted_messages)

        elif human_message:
            human_message = human_message.replace("{", "{{").replace("}", "}}")
            return llm.invoke(human_message)

        else:
            raise ValueError("At least a human_message must be provided.")
        
    @staticmethod    
    def web_search (query):
        """
            Args:
                query: The search query
            Returns:
                Response of the API in markdown format along with embedded links.
            Function:
                Performs a search using the Perplexity API and returns the response or error message.
        """
        api_key = os.getenv("perplexity_api_key")
        llm = ChatPerplexity(
            api_key=api_key, model=constants.perplexity_llm, temperature=0
        )
        response = llm.invoke(query)
        response_content = response.content.strip()

        # Replace citation numbers with markdown links
        if hasattr(response, 'additional_kwargs') and 'citations' in response.additional_kwargs:
            citations = response.additional_kwargs['citations']
            for i, citation in enumerate(citations, 1):
                response_content = re.sub(f'\[{i}\]', f'[{i}]({citation})', response_content)

        return response_content
    
    @staticmethod
    def get_openai_embeddings():
        """
        Returns the OpenAI embedding model.
        """
        load_dotenv(override=True)
        api_key = os.getenv("openai_api_key")

        return OpenAIEmbeddings(model=constants.openai_embedding_model, openai_api_key=api_key)
