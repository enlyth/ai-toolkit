import json
import tempfile
import unittest
from pathlib import Path

import torch

from toolkit.guidance_loss import (
    GuidanceLossSchedule,
    disable_trainable_network,
    guidance_loss_weights,
    inverse_flow_shift,
)


class GuidanceLossTests(unittest.TestCase):
    def _write_schedule(self, data):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "schedule.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_inverse_flow_shift_round_trip(self):
        base_sigma = torch.tensor([0.0, 0.05, 0.2, 0.5, 0.8, 1.0])
        shift = 12.0
        shifted = shift * base_sigma / (1.0 + (shift - 1.0) * base_sigma)
        recovered = inverse_flow_shift(shifted, shift)
        torch.testing.assert_close(recovered, base_sigma, atol=1e-6, rtol=1e-6)

    def test_point_schedule_interpolates_video_and_audio(self):
        path = self._write_schedule(
            {
                "points": [
                    {"base_sigma": 0.0, "w_video": 1.0, "w_audio": 1.5},
                    {"base_sigma": 1.0, "w_video": 5.0, "w_audio": 3.0},
                ]
            }
        )
        schedule = GuidanceLossSchedule(path)
        sigma = torch.tensor([0.0, 0.25, 1.0])
        torch.testing.assert_close(
            schedule.scales(sigma, "video", fallback=9.0),
            torch.tensor([1.0, 2.0, 5.0]),
        )
        torch.testing.assert_close(
            schedule.scales(sigma, "audio", fallback=9.0),
            torch.tensor([1.5, 1.875, 3.0]),
        )

    def test_bucket_schedule_extends_end_values_and_smooths_boundaries(self):
        path = self._write_schedule(
            {
                "buckets": [
                    {"base_lo": 0.0, "base_hi": 0.5, "w_video": 1.0},
                    {"base_lo": 0.5, "base_hi": 1.0, "w_video": 5.0},
                ]
            }
        )
        schedule = GuidanceLossSchedule(path)
        scales = schedule.scales(
            torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]),
            "video",
            fallback=3.0,
        )
        torch.testing.assert_close(
            scales, torch.tensor([1.0, 1.0, 3.0, 5.0, 5.0])
        )

    def test_named_curve_and_missing_stream_fallback(self):
        path = self._write_schedule(
            {
                "image_fit": [
                    {"base": 0.0, "w_video": 2.0},
                    {"base": 1.0, "w_video": 4.0},
                ]
            }
        )
        schedule = GuidanceLossSchedule(path, curve="image_fit")
        sigma = torch.tensor([0.0, 0.5, 1.0])
        torch.testing.assert_close(
            schedule.scales(sigma, "video", fallback=9.0),
            torch.tensor([2.0, 3.0, 4.0]),
        )
        torch.testing.assert_close(
            schedule.scales(
                sigma, "audio", fallback=torch.tensor([1.0, 2.0, 3.0])
            ),
            torch.tensor([1.0, 2.0, 3.0]),
        )

    def test_disable_trainable_network_restores_state_after_error(self):
        class Network:
            is_active = True

        network = Network()
        with self.assertRaises(RuntimeError):
            with disable_trainable_network(network):
                self.assertFalse(network.is_active)
                raise RuntimeError("boom")
        self.assertTrue(network.is_active)

    def test_guidance_loss_weights(self):
        scales = torch.tensor([1.0, 2.0, 4.0])
        torch.testing.assert_close(
            guidance_loss_weights(scales, "inverse"),
            torch.tensor([1.0, 0.5, 0.25]),
        )
        torch.testing.assert_close(
            guidance_loss_weights(scales, "inverse_square"),
            torch.tensor([1.0, 0.25, 0.0625]),
        )


if __name__ == "__main__":
    unittest.main()
