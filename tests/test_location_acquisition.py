"""
tests/test_location_acquisition.py — Test location acquisition and raw coordinate handling.

Verifies:
1. Real place name ("Delhi") with no images -> acquires real satellite imagery tile.
2. Raw coordinates ("28.6139, 77.2090") with no images -> directly resolves without geocoding.
3. Invalid/nonsense location ("xyzqwe123randomland") -> returns explicit error, no silent fallback.
4. Out-of-range coordinates ("128.6139, 277.2090") -> returns explicit range validation error.
5. Structured coordinate inputs: tuple (lat, lon) and dict {"lat": ..., "lon": ...}.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.location_acquisition import geocode_location, get_imagery_for_query
from scripts.orchestrator import run_query
from scripts.interpreter import QueryInterpreter


class MockSpecialist:
    def run(self, **kwargs):
        img = kwargs.get("image") or kwargs.get("image_before")
        return {"answer": "Verification successful", "confidence": 0.99, "image_received": str(img)}


class MockLifecycleManager:
    def load(self, name):
        return MockSpecialist()

    def run_sequence(self, sequence, dispatch_fn):
        return [dispatch_fn(name, MockSpecialist()) for name in sequence]


class TestLocationAcquisition(unittest.TestCase):
    def setUp(self):
        self.mgr = MockLifecycleManager()
        self.interp = QueryInterpreter()

    def test_structured_coordinate_parsing(self):
        # Tuple
        res_tuple = geocode_location((28.6139, 77.2090))
        self.assertEqual(res_tuple["source"], "coordinates")
        self.assertAlmostEqual(res_tuple["lat"], 28.6139)
        self.assertAlmostEqual(res_tuple["lon"], 77.2090)

        # Dict
        res_dict = geocode_location({"lat": 28.6139, "lon": 77.2090})
        self.assertEqual(res_dict["source"], "coordinates")
        self.assertAlmostEqual(res_dict["lat"], 28.6139)
        self.assertAlmostEqual(res_dict["lon"], 77.2090)

        # Out-of-range bounds check
        with self.assertRaises(ValueError) as ctx:
            geocode_location((95.0, 77.2090))
        self.assertIn("out of sane geographic range", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            geocode_location("128.6139, 277.2090")
        self.assertIn("out of sane geographic range", str(ctx.exception))

    def test_case_a_real_place_name(self):
        out = run_query(
            raw_query="Describe the urban area in Delhi",
            images=None,
            interpreter=self.interp,
            lifecycle_manager=self.mgr,
            task_sequence=["captioning"],
        )
        self.assertIsNone(out.get("error"))
        self.assertIsNotNone(out.get("telemetry"))
        self.assertEqual(out["telemetry"]["resolved_name"], "Delhi, India")
        self.assertTrue(out["telemetry"]["is_live_acquired"])
        self.assertGreater(len(out["results"]), 0)
        img_path = out["results"][0]["output"]["image_received"]
        self.assertTrue(os.path.exists(img_path), f"Acquired image does not exist: {img_path}")

    def test_case_b_raw_coordinates(self):
        out = run_query(
            raw_query="What objects are visible at 28.6139, 77.2090?",
            images=None,
            interpreter=self.interp,
            lifecycle_manager=self.mgr,
            task_sequence=["captioning"],
        )
        self.assertIsNone(out.get("error"))
        self.assertIsNotNone(out.get("telemetry"))
        self.assertEqual(out["telemetry"]["source"], "coordinates")
        self.assertGreater(len(out["results"]), 0)
        img_path = out["results"][0]["output"]["image_received"]
        self.assertTrue(os.path.exists(img_path), f"Acquired image does not exist: {img_path}")

    def test_case_c_invalid_nonsense_location(self):
        out = run_query(
            raw_query="Describe the terrain around xyzqwe123randomland",
            images=None,
            interpreter=self.interp,
            lifecycle_manager=self.mgr,
            task_sequence=["captioning"],
        )
        self.assertIsNotNone(out.get("error"))
        self.assertIn("Could not resolve location", out["error"])
        self.assertEqual(len(out["results"]), 0)

    def test_case_d_out_of_range_coordinates(self):
        out = run_query(
            raw_query="Assess the damage at 128.6139, 277.2090",
            images=None,
            interpreter=self.interp,
            lifecycle_manager=self.mgr,
            task_sequence=["captioning"],
        )
        self.assertIsNotNone(out.get("error"))
        self.assertIn("out of sane geographic range", out["error"])
        self.assertEqual(len(out["results"]), 0)

    def test_case_e_cardinal_and_degree_formats(self):
        # Cardinal format with degrees
        res = geocode_location("28.6139° N, 77.2090° E")
        self.assertEqual(res["source"], "coordinates")
        self.assertAlmostEqual(res["lat"], 28.6139)
        self.assertAlmostEqual(res["lon"], 77.2090)

        # Southern/Western hemisphere coordinates
        res_sw = geocode_location("33.8688° S, 151.2093° E")
        self.assertEqual(res_sw["source"], "coordinates")
        self.assertAlmostEqual(res_sw["lat"], -33.8688)
        self.assertAlmostEqual(res_sw["lon"], 151.2093)

        # Query string containing coordinates in natural language
        out = run_query(
            raw_query="What is visible at 28.6139° N, 77.2090° E?",
            images=None,
            interpreter=self.interp,
            lifecycle_manager=self.mgr,
            task_sequence=["captioning"],
        )
        self.assertIsNone(out.get("error"))
        self.assertIsNotNone(out.get("telemetry"))
        self.assertEqual(out["telemetry"]["source"], "coordinates")
        self.assertIn("Raisina Hill", out["telemetry"]["resolved_name"])


if __name__ == "__main__":
    unittest.main()
