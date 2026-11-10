import os
import sys
import subprocess
import pytest


def test_pipeline_imports_without_generator():
    """Verify src.pipeline imports cleanly without src.generation.generator existing."""
    # Ensure generator.py does not exist
    generator_file = os.path.join("src", "generation", "generator.py")
    assert not os.path.exists(generator_file), "generator.py should have been deleted"

    # Verify importing src.pipeline succeeds
    import src.pipeline
    assert hasattr(src.pipeline, "ConversationQA")


def test_system_prompt_exists_and_valid():
    """Verify src.generation.prompts.SYSTEM_PROMPT exists and is a non-empty string."""
    from src.generation.prompts import SYSTEM_PROMPT
    assert isinstance(SYSTEM_PROMPT, str)
    assert len(SYSTEM_PROMPT.strip()) > 0
    assert "evidence excerpts" in SYSTEM_PROMPT
    assert "Later messages override earlier ones" in SYSTEM_PROMPT

    # Verify exported in __init__
    import src.generation
    assert hasattr(src.generation, "SYSTEM_PROMPT")


def test_wrapper_scripts_cli_help():
    """Verify wrapper scripts expose argparse --help via subprocess."""
    scripts = [
        "scripts/run_naive_baseline.py",
        "scripts/run_langchain.py",
        "scripts/run_llamaindex.py",
    ]
    for script in scripts:
        cmd = [sys.executable, script, "--help"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"{script} --help failed with code {res.returncode}: {res.stderr}"
        stdout = res.stdout.lower()
        assert "--eval-set" in stdout
        assert "--output" in stdout
        assert "--top-k" in stdout


@pytest.mark.network
def test_framework_baselines_message_split_configurations():
    """Verify framework_baselines supports standard and message_split configurations."""
    from src.evaluation.framework_baselines import (
        LangChainRAG,
        LlamaIndexRAG,
        LangChainMessageSplitRAG,
        LlamaIndexMessageSplitRAG,
        NaiveBaselineRAG,
        auto_score,
    )

    toy_text = (
        "[01/01/2024 10:00] Alice: Hey let's meet tomorrow at 5pm\n"
        "[01/01/2024 10:05] Bob: Sure sounds good to me\n"
        "[01/01/2024 10:10] Charlie: I can only join at 6pm though"
    )

    lc_std = LangChainRAG(raw_text=toy_text, split_mode="standard")
    assert "document-oriented defaults" in lc_std.name
    assert len(lc_std.chunk_texts) > 0

    lc_msg = LangChainMessageSplitRAG(raw_text=toy_text)
    assert "Message-Split" in lc_msg.name
    assert len(lc_msg.chunk_texts) > 0

    llama_std = LlamaIndexRAG(raw_text=toy_text, split_mode="standard")
    assert "document-oriented defaults" in llama_std.name
    assert len(llama_std.chunk_texts) > 0

    llama_msg = LlamaIndexMessageSplitRAG(raw_text=toy_text)
    assert "Message-Split" in llama_msg.name
    assert len(llama_msg.chunk_texts) > 0


def test_auto_score_logic():
    """Verify centralized auto_score evaluation heuristics."""
    from src.evaluation.framework_baselines import auto_score

    # Positive hit
    retrieved = [
        {"chunk_text": "Meeting moved to 6pm by Charlie", "score": 0.85},
        {"chunk_text": "Other unrelated text", "score": 0.3},
    ]
    assert auto_score(retrieved, "6pm", "Meeting moved to 6pm", is_neg=False) == "hit"

    # Negative control: score below 0.45 should be hit (abstain), above should be partial
    neg_retrieved_low = [{"chunk_text": "Random chat", "score": 0.25}]
    assert auto_score(neg_retrieved_low, "", "", is_neg=True) == "hit"

    neg_retrieved_high = [{"chunk_text": "Random chat", "score": 0.65}]
    assert auto_score(neg_retrieved_high, "", "", is_neg=True) == "partial"
