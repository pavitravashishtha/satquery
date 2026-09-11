import os
import sys
import io
import contextlib
import torch

sys.path.insert(0, "/home/pavitra/satquery/GeoChat")
sys.path.insert(0, "/home/pavitra/satquery")

from scripts.orchestrator import run_query
from scripts.model_registry import build_registry, GEOCHAT

def main():
    print("=" * 80)
    print("PART 2: VERIFYING GUARDRAIL LIVE IN REAL ORCHESTRATOR DISPATCH PATH")
    print("=" * 80)

    image_path = "data/SECOND/im2/00018.png"
    query = "What objects and features are visible in this satellite image?"

    manager = build_registry(device="cuda")

    class MockInterpreter:
        def interpret(self, q):
            class Interpreted:
                raw_query = q
                task_sequence = ["geochat"]
                confidence = 1.0
                needs_clarification = False
            return Interpreted()

    interpreter = MockInterpreter()

    # -------------------------------------------------------------------------
    # Sub-test 2.1: Real orchestrator dispatch call & check log line fires
    # -------------------------------------------------------------------------
    print("\n--- Sub-test 2.1: Normal Dispatch via orchestrator.run_query() ---")
    
    # Capture stdout to verify log line
    stdout_capture = io.StringIO()
    with contextlib.redirect_stdout(stdout_capture):
        res = run_query(
            raw_query=query,
            images=[image_path],
            task_type="geochat",
            interpreter=interpreter,
            lifecycle_manager=manager,
        )
    
    captured_output = stdout_capture.getvalue()
    print(captured_output)

    print("\nVerifying captured stdout for guardrail check:")
    assert "[GUARDRAIL CHECK] function invoked" in captured_output, (
        "FAILED: '[GUARDRAIL CHECK] function invoked' did NOT fire in real dispatch path!"
    )
    print("  -> PASSED: '[GUARDRAIL CHECK] function invoked' confirmed in stdout during orchestrator dispatch!")

    output_dict = res["results"][0]["output"]
    print("  Orchestrator output dict:")
    print(f"    answer: {repr(output_dict['answer'])}")
    print(f"    confidence: {output_dict['confidence']}")
    print(f"    is_degenerate: {output_dict.get('is_degenerate')}")

    assert output_dict.get("is_degenerate") is False, "Expected is_degenerate=False for normal run"
    assert output_dict["confidence"] > 0.5, "Expected normal confidence > 0.5"

    # -------------------------------------------------------------------------
    # Sub-test 2.2: Artificially forced loop through real dispatch path
    # -------------------------------------------------------------------------
    print("\n--- Sub-test 2.2: Forced Loop End-to-End Through orchestrator.run_query() ---")
    
    # Access loaded specialist from manager
    specialist = manager.load(GEOCHAT)
    assert specialist is not None, "GeoChat specialist should be loaded in manager"

    orig_generate = specialist.model.generate

    # Create a mock generate that simulates the exact unmitigated LLaMA repetition loop
    class MockOutput:
        def __init__(self, sequences, scores):
            self.sequences = sequences
            self.scores = scores

    def forced_loop_generate(*args, **kwargs):
        # Decode text that repeats "a building" consecutively 8 times
        loop_text = (
            "In this satellite image, there are several objects and features visible, "
            "including a football field, a road, a building, a building, a building, "
            "a building, a building, a building, a building."
        )
        input_ids = args[0] if args else kwargs.get("input_ids")
        tokens = specialist.tokenizer.encode(loop_text, return_tensors="pt").to(input_ids.device)
        # Prepend dummy prompt sequence length so slicing outputs.sequences[0, input_ids.shape[1]:] works
        full_seq = torch.cat([input_ids, tokens], dim=-1)
        # Mock scores for confidence computation (high prob like the original bug had: ~0.95)
        step_logits = [torch.zeros((1, 32000), device=input_ids.device) for _ in range(tokens.shape[1])]
        for s in step_logits:
            s[0, 5214] = 10.0  # high confidence logit
        return MockOutput(full_seq, tuple(step_logits))

    specialist.model.generate = forced_loop_generate

    try:
        stdout_capture2 = io.StringIO()
        with contextlib.redirect_stdout(stdout_capture2):
            res_forced = run_query(
                raw_query=query,
                images=[image_path],
                task_type="geochat",
                interpreter=interpreter,
                lifecycle_manager=manager,
            )
        captured2 = stdout_capture2.getvalue()
        print(captured2)
    finally:
        specialist.model.generate = orig_generate

    print("\nVerifying forced loop handling in orchestrator output:")
    forced_out = res_forced["results"][0]["output"]
    print(f"  Truncated Answer: {repr(forced_out['answer'])}")
    print(f"  Confidence: {forced_out['confidence']}")
    print(f"  is_degenerate: {forced_out.get('is_degenerate')}")

    assert "[GUARDRAIL CHECK] function invoked" in captured2, "Guardrail must be invoked"
    assert "Degenerate repetition loop detected! Truncating response." in captured2, "Warning must be logged"
    assert forced_out.get("is_degenerate") is True, "is_degenerate must be True"
    assert forced_out["confidence"] <= 0.40, f"Confidence must be capped at <= 0.40, got {forced_out['confidence']}"
    assert forced_out["answer"].count("a building") == 1, (
        f"Expected exactly 1 mention of 'a building' after truncation, got: {forced_out['answer']}"
    )
    print("  -> PASSED: Guardrail successfully truncated output, set is_degenerate=True, and capped confidence at 0.40 through full orchestrator dispatch!")

    manager.unload(GEOCHAT)
    print("\nALL PART 2 SUB-TESTS COMPLETED AND PASSED!")

if __name__ == "__main__":
    main()
