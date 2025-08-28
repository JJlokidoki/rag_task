# Prompts module for RAG system
from .crypto_expert import CRYPTO_EXPERT_PROMPT, CRYPTO_EXPERT_REFINEMENT_PROMPT
from .crypto_verificator import CRYPTO_VERIFICATOR_PROMPT

__all__ = [
    'CRYPTO_EXPERT_PROMPT',
    'CRYPTO_EXPERT_REFINEMENT_PROMPT', 
    'CRYPTO_VERIFICATOR_PROMPT'
]
