import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evals import load_dataset, mean_scores, no_context_retrieval_score


def test_load_dataset_and_mean_scores(tmp_path):
    dataset = tmp_path / "dataset.csv"
    dataset.write_text(
        "question,reference\nWhat is RAG?,A retrieval system.\n", encoding="utf-8"
    )

    assert load_dataset(dataset) == [
        {"question": "What is RAG?", "reference": "A retrieval system."}
    ]
    assert mean_scores(
        [{"scores": {"faithfulness": 1.0}}, {"scores": {"faithfulness": 0.5}}]
    ) == {"faithfulness": 0.75}
    assert no_context_retrieval_score([]) == 1.0
    assert no_context_retrieval_score(["stored chunk"]) == 0.0
