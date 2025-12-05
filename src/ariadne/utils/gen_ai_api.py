import os
import numpy as np
from ariadne.utils.utils import get_environment_variable
from openai import OpenAI, AzureOpenAI
from typing import List, Optional, Dict, Any, Tuple
from dotenv import load_dotenv

load_dotenv()

_PRICING_TABLE = {
    # OpenAI / Azure Standard Global
    "o3": {"input": 2.00, "output": 8.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "text-embedding-3-small": {"input": 0.02, "output": 0.00},
    "text-embedding-3-large": {"input": 0.13, "output": 0.00},
    # Local models (Free)
    "local": {"input": 0.00, "output": 0.00},
}

_TEMPERATURE_OK_MODELS = {
    "gpt-4o", "gpt-4", "gpt-35-turbo",
    "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"
}


class _AIClientFactory:

    @staticmethod
    def get_client(task_type: str = "llm") -> Tuple[Any, str, str]:
        """
        Args:
        task_type: 'llm' or 'embedding'

        Returns:
            client: Configured AI client
            model_name: Name of model
            provider_type: 'openai', 'azure', or 'local'
        """
        provider = get_environment_variable("GENAI_PROVIDER").lower()

        if task_type == "embedding":
            model_name = get_environment_variable("EMBEDDING_MODEL")
            api_key = get_environment_variable("EMBEDDING_API_KEY")
        else:
            model_name = get_environment_variable("LLM_MODEL")
            api_key = get_environment_variable("LLM_API_KEY")

        if provider == "azure":
            if task_type == "embedding":
                endpoint = get_environment_variable("AZURE_EMBEDDING_ENDPOINT")
            else:
                endpoint = get_environment_variable("AZURE_LLM_ENDPOINT")

            # Seems a bug, but must provide api key in both headers and api-key argument or we get an error:
            client = OpenAI(
                api_key=api_key,
                base_url=endpoint,
                default_query={
                    "api-version": get_environment_variable("AZURE_OPENAI_API_VERSION")
                },
                default_headers={"api-key": api_key},
            )
            return client, model_name, "azure"

        elif provider == "lm-studio":
            endpoint = get_environment_variable("LM_STUDIO_ENDPOINT")
            client = OpenAI(base_url=endpoint, api_key="lm-studio")
            return client, model_name, "local"

        else:  # OpenAI Direct
            client = OpenAI(api_key=api_key)
            return client, model_name, "openai"


def _calculate_cost(
    model_name: str, input_tok: int, output_tok: int, provider_type: str
) -> float:
    if provider_type == "local":
        return 0.0
    price_key = next((k for k in _PRICING_TABLE if k in model_name), None)
    if not price_key:
        return 0.0
    prices = _PRICING_TABLE[price_key]
    return round(
        ((input_tok / 1e6) * prices["input"]) + ((output_tok / 1e6) * prices["output"]),
        6,
    )


def get_embedding_vectors(texts: List[str]) -> Dict[str, Any]:
    """
    Generates embedding vectors for a list of texts using the embedding-specific config.

    Args:
        texts: List of texts to generate embeddings for.

    Returns:
        A dictionary containing:
            - "embeddings": A numpy array of embedding vectors.
            - "usage": A dictionary with token usage and cost details.
    """

    client, model, provider = _AIClientFactory.get_client(task_type="embedding")

    response = client.embeddings.create(input=texts, model=model)

    data = sorted(response.data, key=lambda x: x.index)
    np_vectors = np.array([item.embedding for item in data])

    usage = response.usage
    total_cost = _calculate_cost(model, usage.prompt_tokens, 0, provider)

    return {
        "embeddings": np_vectors,
        "usage": {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_cost_usd": total_cost,
            "model_used": model,
        },
    }


def get_llm_response(
    prompt: str, system_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates text response using the LLM-specific config.

    Args:
        prompt: The user prompt to send to the LLM.
        system_prompt: Optional system prompt to guide the LLM's behavior.
    Returns:
        A dictionary containing:
            - "content": The generated text response from the LLM.
            - "usage": A dictionary with token usage and cost details.
    """

    client, model, provider = _AIClientFactory.get_client(task_type="llm")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    if model in _TEMPERATURE_OK_MODELS:
        temperature = 0.0
    else:
        temperature = None

    response = client.chat.completions.create(
        model=model, messages=messages, temperature = temperature
    )

    usage = response.usage
    reasoning_tokens = 0
    if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
        reasoning_tokens = getattr(
            usage.completion_tokens_details, "reasoning_tokens", 0
        )

    total_cost = _calculate_cost(
        model, usage.prompt_tokens, usage.completion_tokens, provider
    )

    return {
        "content": response.choices[0].message.content,
        "usage": {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "total_cost_usd": total_cost,
            "model_used": model,
        },
    }


if __name__ == "__main__":
    texts = ["Acute Myocardial Infarction", "Liver Failure"]
    embeddings_result = get_embedding_vectors(texts)
    print("Embeddings Result:", embeddings_result)

    llm_result = get_llm_response(
        "What is the capital of the Netherlands?", "You are a helpful assistant."
    )
    print("LLM Response:", llm_result)
