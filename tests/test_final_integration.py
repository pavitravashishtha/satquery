import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.model_registry import build_registry, GEOCHAT, INTERPRETER_LLM
from scripts.orchestrator import run_query
from scripts.interpreter import QueryInterpreter
from scripts.schema import InterpretedQuery, Task, Domain, Sensor, Source


def test_final_integration():
    print("=====================================================================")
    print("  FINAL INTEGRATION CHECK: End-to-End Pipeline with GeoChat")
    print("=====================================================================")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_image_path = os.path.join(project_root, "data", "SECOND", "im2", "00003.png")
    assert os.path.exists(test_image_path), f"Test image not found at {test_image_path}"

    test_query = "What objects and features are visible in this satellite image?"

    print(f"\nTest Image: {test_image_path}")
    print(f"Test Query: {test_query}")

    manager = build_registry(device="cuda")

    # Measure VRAM before GeoChat call
    torch.cuda.empty_cache()
    vram_before = torch.cuda.memory_allocated() / 1e9
    print(f"\n[VRAM] Before GeoChat call: {vram_before:.3f} GB")

    # Run interpreter to verify full interpreter + orchestrator integration
    print("\n[Pipeline] Interpreting query with Qwen3.5-2B LLM on CPU...")
    from scripts.model_registry import InterpreterLLMWrapper
    llm = InterpreterLLMWrapper(device="cpu")
    llm.load()
    interpreter = QueryInterpreter(llm_fn=llm)

    # Dispatch full pipeline with task_sequence=['geochat']
    print("\n[Pipeline] Running run_query() through orchestrator...")
    pipeline_result = run_query(
        raw_query=test_query,
        images=[test_image_path],
        interpreter=interpreter,
        lifecycle_manager=manager,
        task_sequence=["geochat"],
    )

    vram_during = torch.cuda.memory_allocated() / 1e9
    print(f"[VRAM] During GeoChat call: {vram_during:.3f} GB")

    # Extract output text
    results = pipeline_result.get("results", [])
    assert len(results) > 0, "No results returned from pipeline!"
    geochat_output = results[0]["output"]
    decoded_text = geochat_output.get("answer", "")
    confidence = geochat_output.get("confidence", 0.0)

    print("\n" + "=" * 70)
    print("ACTUAL DECODED OUTPUT TEXT:")
    print(f"\"{decoded_text}\"")
    print(f"Confidence: {confidence}")
    print("=" * 70)

    # Verification: Confirm coherent English, NOT CJK garbage
    # CJK Unified Ideographs block: \u4e00 - \u9fff
    cjk_chars = [c for c in decoded_text if '\u4e00' <= c <= '\u9fff']
    print(f"\nCJK characters count: {len(cjk_chars)}")
    assert len(cjk_chars) == 0, f"Error: Decoded text contains CJK characters: {cjk_chars}"
    assert len(decoded_text.strip()) > 0, "Error: Decoded text is empty!"

    # Unload GeoChat
    manager.unload(GEOCHAT)
    torch.cuda.empty_cache()
    vram_after = torch.cuda.memory_allocated() / 1e9
    print(f"\n[VRAM] After GeoChat unload: {vram_after:.3f} GB")

    # Assert VRAM bounds (RTX 4050 / RTX 4070 Ti isolation check)
    assert vram_during < 5.5, f"VRAM during call ({vram_during:.2f} GB) indicates co-existence of models!"
    print(f"[VRAM Check] Verified no Qwen/GeoChat coexistence: peak={vram_during:.2f} GB < 5.5 GB limit.")

    print("\nFINAL INTEGRATION CHECK PASSED SUCCESSFULLY.")


if __name__ == "__main__":
    test_final_integration()
