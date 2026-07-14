import time
from dataclasses import dataclass, field
from datetime import datetime

from rag_helper import RAGBase


@dataclass
class LLMCallRecord:
    model: str
    prompt: str
    instructions: str
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_time: float
    cost: float
    timestamp: datetime = field(default_factory=datetime.now)


def calculate_cost(model, usage):
    """
    Gemini pricing is not implemented yet.
    Return 0 for now.
    """
    return 0.0


class RAGWithMetrics(RAGBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_call: LLMCallRecord = None

    def llm(self, prompt):
        start_time = time.time()

        response = self._call_llm(prompt)

        response_time = time.time() - start_time

        self._log_response(
            prompt=prompt,
            response=response,
            response_time=response_time
        )

        return response.text

    def _call_llm(self, prompt):

        full_prompt = f"""{self.instructions}

{prompt}
"""

        response = self.llm_client.models.generate_content(
            model=self.model,
            contents=full_prompt
        )

        return response

    def _log_response(self, prompt, response, response_time):

        usage = response.usage_metadata

        prompt_tokens = usage.prompt_token_count
        completion_tokens = usage.candidates_token_count
        total_tokens = usage.total_token_count

        call_record = LLMCallRecord(
            model=self.model,
            prompt=prompt,
            instructions=self.instructions,
            answer=response.text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            response_time=response_time,
            cost=calculate_cost(self.model, usage),
        )

        print(call_record)

        self.last_call = call_record