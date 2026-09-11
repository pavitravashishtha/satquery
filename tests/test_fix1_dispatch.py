import os
import sys
import io
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.model_registry import build_registry, GEOCHAT
from scripts.orchestrator import _dispatch_geochat, run_query
from scripts.schema import InterpretedQuery, Task, Domain, Sensor, Source


def test_fix1_dispatch():
    print("=== Testing Fix 1: GeoChat Dispatch Routing ===")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_image_path = os.path.join(project_root, "GeoChat", "demo_images", "7292.JPG")
    assert os.path.exists(test_image_path), f"Test image not found at {test_image_path}"

    query = "What objects are visible in this satellite image?"

    print("\n1. Initializing registry and loading GEOCHAT via _load_geochat()...")
    manager = build_registry(device="cuda")
    specialist_obj = manager.load(GEOCHAT)

    # Verify loaded specialist class
    assert type(specialist_obj).__name__ == "GeoChatSpecialist", (
        f"Expected GeoChatSpecialist, got {type(specialist_obj).__name__}"
    )
    print(f"Verified loaded specialist type: {type(specialist_obj).__name__}")

    # Capture stdout during dispatch call to verify the log statement
    stdout_capture = io.StringIO()
    print("\n2. Executing _dispatch_geochat directly with task_type='geochat'...")
    with redirect_stdout(stdout_capture):
        result = _dispatch_geochat(
            specialist_obj=specialist_obj,
            images=test_image_path,
            raw_query=query,
            task_type="geochat",
        )

    captured_logs = stdout_capture.getvalue()
    print("\n--- Captured stdout during dispatch ---")
    print(captured_logs)
    print("--- End of captured stdout ---")

    # Find the target log line
    matching_lines = [
        line for line in captured_logs.splitlines()
        if "[GeoChatSpecialist] Running inference with tokenizer:" in line
    ]
    assert len(matching_lines) > 0, "Log statement '[GeoChatSpecialist] Running inference with tokenizer:' was not found!"
    target_log_line = matching_lines[0]
    print(f"\nTarget log line captured: '{target_log_line}'")

    # Assertions on captured log line:
    assert "GeoChatSpecialist" in target_log_line, "Expected 'GeoChatSpecialist' in log line"
    assert ("geochat" in target_log_line.lower() or "vicuna" in target_log_line.lower()), (
        f"Expected tokenizer path to contain 'geochat' or 'vicuna', got {target_log_line}"
    )
    assert "qwen" not in target_log_line.lower(), f"Forbidden 'Qwen' found in tokenizer log line: {target_log_line}"

    print("\n3. Dispatch result:")
    print(f"Answer: {result.get('answer')}")
    print(f"Confidence: {result.get('confidence')}")
    print(f"Task Type: {result.get('task_type')}")

    # Verify orchestrator run_query end-to-end routing with task_sequence=['geochat']
    print("\n4. Testing full orchestrator run_query with task_sequence=['geochat']...")
    class MockInterpreter:
        def interpret(self, q):
            return InterpretedQuery(
                raw_query=q,
                task_sequence=[Task.GEOCHAT],
                primary_domain=Domain.URBAN,
                target_sensor=Sensor.OPTICAL,
                confidence=1.0,
                source=Source.KEYWORD_FALLBACK,
            )

    pipe_out = run_query(
        raw_query=query,
        images=[test_image_path],
        interpreter=MockInterpreter(),
        lifecycle_manager=manager,
        task_sequence=["geochat"],
    )
    print(f"Pipeline output status: needs_clarification={pipe_out.get('needs_clarification')}")
    print(f"Pipeline results: {pipe_out.get('results')}")

    # Unload to free VRAM
    manager.unload(GEOCHAT)
    print("\nFIX 1 VERIFICATION PASSED.")


if __name__ == "__main__":
    test_fix1_dispatch()
