import time

from tqdm.auto import tqdm
from google.genai import types

from rag_helper import RAGBase


# --------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------

def calc_price(usage):
    input_tokens = usage.prompt_token_count or 0
    output_tokens = usage.candidates_token_count or 0

    # Gemini 2.5 Flash pricing (adjust if needed)
    input_price_per_million = 0.15
    output_price_per_million = 0.60

    input_cost = (input_tokens / 1_000_000) * input_price_per_million
    output_cost = (output_tokens / 1_000_000) * output_price_per_million

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


def calc_total_price(usages):
    total = 0.0

    for usage in usages:
        total += calc_price(usage)["total_cost"]

    return total


# --------------------------------------------------------------------
# Structured Output
# --------------------------------------------------------------------

def llm_structured(
    client,
    instructions,
    user_prompt,
    output_type,
    model="gemini-2.5-flash",
):
    full_prompt = f"""{instructions}

{user_prompt}
"""

    response = client.models.generate_content(
        model=model,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=output_type,
        ),
    )

    return response.parsed, response.usage_metadata


def llm_structured_retry(
    client,
    instructions,
    user_prompt,
    output_type,
    model="gemini-2.5-flash",
    max_retries=3,
):
    for attempt in range(max_retries):
        try:
            return llm_structured(
                client=client,
                instructions=instructions,
                user_prompt=user_prompt,
                output_type=output_type,
                model=model,
            )
        except Exception:
            if attempt == max_retries - 1:
                raise

            time.sleep(2 ** attempt)


# --------------------------------------------------------------------
# RAG with Usage Tracking
# --------------------------------------------------------------------

class RAGWithUsage(RAGBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.usages = []
        self.last_usage = None

    def reset_usage(self):
        self.usages = []
        self.last_usage = None

    def search(self, query, num_results=5):
        boost_dict = {
            "question": 1.0,
            "answer": 2.0,
            "section": 0.1,
        }

        filter_dict = {
            "course": self.course,
        }

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict,
        )

    def llm(self, prompt):
        full_prompt = f"""{self.instructions}

{prompt}
"""

        response = self.llm_client.models.generate_content(
            model=self.model,
            contents=full_prompt,
        )

        self.last_usage = response.usage_metadata
        self.usages.append(response.usage_metadata)

        return response.text

    def total_cost(self):
        return calc_total_price(self.usages)


# --------------------------------------------------------------------
# Progress Mapping
# --------------------------------------------------------------------

def map_progress(pool, seq, f):
    results = []

    with tqdm(total=len(seq)) as progress:
        futures = []

        for item in seq:
            future = pool.submit(f, item)
            future.add_done_callback(lambda _: progress.update())
            futures.append(future)

        for future in futures:
            results.append(future.result())

    return results