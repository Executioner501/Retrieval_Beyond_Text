import os
import json
import pytest
from scripts.make_report import update_readme, START_MARKER, END_MARKER


def test_make_report_updates_tables_and_preserves_content(tmp_path):
    """Verify make_report populates tables between markers and preserves outer content."""
    readme_path = str(tmp_path / "README.md")
    eval_dir = str(tmp_path / "eval_results")
    os.makedirs(eval_dir, exist_ok=True)

    # Write initial README with markers
    initial_content = (
        "# Test WhatsApp QA System\n\n"
        "Introduction text here.\n\n"
        "## 📊 5. Empirical Benchmark Results\n\n"
        f"{START_MARKER}\nold results\n{END_MARKER}\n\n"
        "## Next Section\n\n"
        "Unchanged footer content.\n"
    )
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(initial_content)

    # Create dummy eval fixtures
    test_summary = {
        "summary": {
            "questions": 33,
            "answerable": 28,
            "negative": 5,
            "hit@1": 0.607,
            "hit@4": 0.857,
            "mrr@k": 0.705,
            "dense_only_hit@4": 0.750,
            "hit@1_ci95": [0.429, 0.786],
            "hit@4_ci95": [0.714, 0.964],
            "mrr@k_ci95": [0.571, 0.839],
            "avg_retrieved_messages": 28.4,
            "avg_retrieved_chars": 2512.2,
            "hit@4[factual-recall] (n=15)": 0.933,
        }
    }
    with open(os.path.join(eval_dir, "qa_eval_eval_test_minilm_clean.json"), "w", encoding="utf-8") as f:
        json.dump(test_summary, f)

    ablation_text = (
        "# Ablation Study\n\n"
        "| Chunker | Retrieval | hit@1 (95% CI) | hit@4 (95% CI) | MRR (95% CI) |\n"
        "| :--- | :--- | :---: | :---: | :---: |\n"
        "| semantic | hybrid | 0.607 [0.429, 0.786] | 0.821 [0.679, 0.964] | 0.696 [0.536, 0.848] |\n"
    )
    with open(os.path.join(eval_dir, "ablation.md"), "w", encoding="utf-8") as f:
        f.write(ablation_text)

    embedder_text = (
        "# Embedder Comparison\n\n"
        "| Embedder | hit@1 | hit@4 | MRR |\n"
        "| :--- | :---: | :---: | :---: |\n"
        "| minilm | 0.607 | 0.839 | 0.701 |\n"
    )
    with open(os.path.join(eval_dir, "embedder_comparison.md"), "w", encoding="utf-8") as f:
        f.write(embedder_text)

    # First run
    success = update_readme(readme_path, eval_dir)
    assert success is True

    with open(readme_path, "r", encoding="utf-8") as f:
        content_after_first = f.read()

    assert START_MARKER in content_after_first
    assert END_MARKER in content_after_first
    assert "# Test WhatsApp QA System" in content_after_first
    assert "Unchanged footer content." in content_after_first
    assert "0.857" in content_after_first  # from test_summary hit@4
    assert "semantic" in content_after_first  # from ablation
    assert "minilm" in content_after_first  # from embedder


def test_make_report_idempotence(tmp_path):
    """Verify running make_report twice yields byte-identical README.md."""
    readme_path = str(tmp_path / "README.md")
    eval_dir = str(tmp_path / "eval_results")
    os.makedirs(eval_dir, exist_ok=True)

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"Header\n\n{START_MARKER}\n{END_MARKER}\n\nFooter\n")

    update_readme(readme_path, eval_dir)
    with open(readme_path, "rb") as f:
        bytes_run1 = f.read()

    update_readme(readme_path, eval_dir)
    with open(readme_path, "rb") as f:
        bytes_run2 = f.read()

    assert bytes_run1 == bytes_run2


def test_make_report_missing_inputs_writes_todo_without_crashing(tmp_path):
    """Verify missing evaluation artifacts generate marked TODOs without failing."""
    readme_path = str(tmp_path / "README.md")
    empty_eval_dir = str(tmp_path / "empty_evals")
    os.makedirs(empty_eval_dir, exist_ok=True)

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"Header\n\n{START_MARKER}\n{END_MARKER}\n\nFooter\n")

    # Should not raise exception
    success = update_readme(readme_path, empty_eval_dir)
    assert success is True

    with open(readme_path, "r", encoding="utf-8") as f:
        text = f.read()

    assert "TODO: run scripts/run_qa_eval.py" in text
    assert "TODO: run scripts/run_ablation.py" in text
    assert "TODO: run scripts/run_embedder_comparison.py" in text
