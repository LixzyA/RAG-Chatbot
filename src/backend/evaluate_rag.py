import asyncio
import json
import logging
import re
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from vectordb.core import init_chroma_client
from chat.core import init_llm
from chat.service import get_answers, get_relevant_files
from huggingface_hub import AsyncInferenceClient
from chromadb.errors import NotFoundError as ChromaNotFoundError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Clients (initialized in main)
# ---------------------------------------------------------------------------
vectordb_client = None
llm_client: AsyncInferenceClient = None

CRITIQUE_MODEL = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

# ---------------------------------------------------------------------------
# Critique prompts
# ---------------------------------------------------------------------------
question_groundedness_critique_prompt = """\
You will be given a context and a question.
Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here are the question and context.

Question: {question}\n
Context: {context}\n
Answer::: """

question_relevance_critique_prompt = """\
You will be given a question.
Your task is to provide a 'total rating' representing how useful this question can be to users seeking information from a knowledge base.
Give your answer on a scale of 1 to 5, where 1 means that the question is not useful at all, and 5 means that the question is extremely useful.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}\n
Answer::: """

question_standalone_critique_prompt = """\
You will be given a question.
Your task is to provide a 'total rating' representing how context-independent this question is.
Give your answer on a scale of 1 to 5, where 1 means that the question depends on additional information to be understood, and 5 means that the question makes sense by itself.
For instance, if the question refers to a particular setting, like 'in the context' or 'in the document', the rating must be 1.
The questions can contain obscure technical nouns or acronyms and still be a 5: it must simply be clear to an operator with access to documentation what the question is about.

For instance, "What is the name of the checkpoint from which the ViT model is imported?" should receive a 1, since there is an implicit mention of a context, thus the question is not independent from the context.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}\n
Answer::: """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_rating(text: str) -> float | None:
    """Extract the numeric rating from an LLM critique response."""
    match = re.search(r"Total rating:\s*([1-5](?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


async def critique_agent(question: str, context: str) -> dict:
    """
    Call the LLM judge for three critique dimensions in parallel:
      - groundedness: can the question be answered from the retrieved context?
      - relevance: is the question useful/meaningful?
      - standalone: is the question self-contained (no implicit references)?

    Returns a dict with scores and raw evaluation text for each dimension.
    """
    global llm_client

    async def _call(prompt: str) -> str:
        response = await llm_client.chat.completions.create(
            model=CRITIQUE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.1,
        )
        return response.choices[0].message.content or ""

    groundedness_raw, relevance_raw, standalone_raw = await asyncio.gather(
        _call(question_groundedness_critique_prompt.format(question=question, context=context)),
        _call(question_relevance_critique_prompt.format(question=question)),
        _call(question_standalone_critique_prompt.format(question=question)),
    )

    return {
        "groundedness": {
            "score": _parse_rating(groundedness_raw),
            "evaluation": groundedness_raw,
        },
        "relevance": {
            "score": _parse_rating(relevance_raw),
            "evaluation": relevance_raw,
        },
        "standalone": {
            "score": _parse_rating(standalone_raw),
            "evaluation": standalone_raw,
        },
    }


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

async def evaluate_rag() -> list[dict]:
    """
    For each row in evaluation.jsonl:
      1. Retrieve relevant context from the vector store.
      2. Generate an answer from the LLM.
      3. Run the critique agent to score groundedness, relevance, and standalone.
      4. Collect results.
    """
    global vectordb_client, llm_client

    eval_data: list[dict] = []

    with open("./evaluation.jsonl", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    logging.info(f"Loaded {len(lines)} evaluation sample(s).")

    for i, row in enumerate(lines, 1):
        question = row["Question"]
        reference = row.get("Answer", "")
        logging.info(f"[{i}/{len(lines)}] Evaluating: {question!r}")

        # 1. Retrieve context
        try:
            files = get_relevant_files(vectordb_client, question, top_k=3)
            docs = files.get("documents") if files else None
            relevant_contexts: list[str] = docs[0] if docs and docs[0] else []
        except (ChromaNotFoundError, Exception) as e:
            logging.warning(f"  Could not retrieve context (collection may be empty): {e}")
            relevant_contexts = []
        context = "\n\n".join(relevant_contexts)

        # 2. Generate answer
        response_chunks: list[str] = []
        async for chunk in get_answers(llm_client, question, context, "generalist"):
            response_chunks.append(chunk)
        response = "".join(response_chunks)

        # 3. Critique
        critiques = await critique_agent(question, context)

        eval_data.append({
            "user_input": question,
            "retrieved_contexts": relevant_contexts,
            "response": response,
            "reference": reference,
            "critiques": critiques,
        })

        logging.info(
            f"  groundedness={critiques['groundedness']['score']} | "
            f"relevance={critiques['relevance']['score']} | "
            f"standalone={critiques['standalone']['score']}"
        )

    return eval_data


def _compute_summary(eval_data: list[dict]) -> dict:
    """Compute average scores across all evaluation samples."""
    dims = ["groundedness", "relevance", "standalone"]
    summary: dict = {}
    for dim in dims:
        scores = [
            item["critiques"][dim]["score"]
            for item in eval_data
            if item["critiques"][dim]["score"] is not None
        ]
        summary[f"avg_{dim}"] = round(sum(scores) / len(scores), 2) if scores else None
    summary["total_samples"] = len(eval_data)
    return summary


async def main():
    global vectordb_client, llm_client

    vectordb_client = init_chroma_client()
    llm_client = init_llm()

    logging.info("Starting RAG evaluation pipeline…")
    eval_data = await evaluate_rag()

    summary = _compute_summary(eval_data)

    # Save full results to a timestamped JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"./eval_results_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": eval_data}, f, ensure_ascii=False, indent=2)

    print("\n=== Evaluation Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nFull results saved to: {output_path}")

    return eval_data


if __name__ == "__main__":
    asyncio.run(main())
