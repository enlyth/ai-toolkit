import json
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union

import torch

from toolkit.paths import get_path


Number = Union[int, float]


def inverse_flow_shift(sigma: torch.Tensor, shift: Number) -> torch.Tensor:
    """Map a scheduler-shifted flow sigma back to its unshifted coordinate."""
    shift = float(shift)
    sigma = sigma.clamp(0.0, 1.0)
    if shift == 1.0:
        return sigma
    denominator = shift - (shift - 1.0) * sigma
    return (sigma / denominator.clamp_min(torch.finfo(sigma.dtype).eps)).clamp(0.0, 1.0)


@contextmanager
def disable_trainable_network(network):
    """Temporarily disable an ai-toolkit network while preserving its state."""
    if network is None:
        yield
        return

    was_active = network.is_active
    network.is_active = False
    try:
        yield
    finally:
        network.is_active = was_active


class GuidanceLossSchedule:
    """A linearly interpolated CFG-recovery schedule loaded from JSON.

    Supported JSON shapes:

    * ``{"points": [{"base_sigma": 0.0, "w_video": 1.0, ...}, ...]}``
    * ``{"buckets": [{"base_lo": 0.0, "base_hi": 0.2, ...}, ...]}``
    * a named point list such as ``image_fit`` or ``video_fit``, selected with
      ``curve``. Point lists may use ``base`` instead of ``base_sigma``.

    Bucket values are placed at bucket centres and interpolated, with the first
    and last values extended to the schedule endpoints. This avoids hard target
    jumps at bucket boundaries.
    """

    def __init__(self, path: Union[str, Path], curve: Optional[str] = None):
        expanded_path = Path(path).expanduser()
        self.path = Path(get_path(str(expanded_path)))
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Guidance loss schedule does not exist: {self.path}"
            )

        with self.path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        self._top_level = data
        self._points = self._select_points(data, curve)
        if not self._points:
            raise ValueError(
                f"Guidance loss schedule has no usable points: {self.path}"
            )

    @staticmethod
    def _point_sigma(point: dict) -> float:
        for key in ("base_sigma", "sigma", "base"):
            if key in point:
                return float(point[key])
        raise ValueError(
            "Guidance schedule points need one of: base_sigma, sigma, base"
        )

    def _select_points(self, data: dict, curve: Optional[str]) -> list:
        if curve is not None:
            if curve not in data:
                raise ValueError(
                    f"Guidance schedule curve '{curve}' was not found in {self.path}"
                )
            source = data[curve]
            if not isinstance(source, list):
                raise ValueError(
                    f"Guidance schedule curve '{curve}' must be a list"
                )
            points = [dict(point) for point in source]
        elif isinstance(data.get("points"), list):
            points = [dict(point) for point in data["points"]]
        elif isinstance(data.get("buckets"), list):
            buckets = data["buckets"]
            points = []
            for bucket in buckets:
                lo = float(bucket["base_lo"])
                hi = float(bucket["base_hi"])
                if hi <= lo:
                    raise ValueError(
                        f"Guidance schedule bucket has base_hi <= base_lo: {bucket}"
                    )
                point = dict(bucket)
                point["base_sigma"] = 0.5 * (lo + hi)
                points.append(point)

            if points:
                first = dict(points[0])
                first["base_sigma"] = float(buckets[0]["base_lo"])
                last = dict(points[-1])
                last["base_sigma"] = float(buckets[-1]["base_hi"])
                points = [first, *points, last]
        else:
            points = []

        points.sort(key=self._point_sigma)
        previous = None
        for point in points:
            sigma = self._point_sigma(point)
            if not 0.0 <= sigma <= 1.0:
                raise ValueError(
                    f"Guidance schedule sigma must be in [0, 1], got {sigma}"
                )
            if previous is not None and sigma <= previous:
                raise ValueError("Guidance schedule sigmas must be strictly increasing")
            previous = sigma
        return points

    def scales(
        self,
        sigma: torch.Tensor,
        stream: str,
        fallback: Union[Number, torch.Tensor],
    ) -> torch.Tensor:
        if stream not in ("video", "audio"):
            raise ValueError(f"Unknown guidance schedule stream: {stream}")

        key = f"w_{stream}"
        output_dtype = sigma.dtype
        work_sigma = sigma.float()
        fallback_tensor = torch.as_tensor(
            fallback, device=sigma.device, dtype=torch.float32
        )
        if fallback_tensor.numel() == 1:
            fallback_tensor = fallback_tensor.expand_as(work_sigma)
        else:
            fallback_tensor = fallback_tensor.reshape(work_sigma.shape)

        top_level_fallback = self._top_level.get(key, None)
        values = []
        for point in self._points:
            value = point.get(key, top_level_fallback)
            values.append(None if value is None else float(value))

        if all(value is None for value in values):
            return fallback_tensor.to(output_dtype)

        # A schedule that defines this stream on only SOME points fills the
        # gaps with a single fallback value. When the fallback is per-sample
        # (a randomized list target), only sample 0's value can be used here —
        # per-sample fallbacks cannot vary along the schedule axis. Fully
        # specify the stream in the JSON to avoid the approximation.
        values = [
            float(fallback_tensor.flatten()[0].item()) if value is None else value
            for value in values
        ]
        xs = torch.tensor(
            [self._point_sigma(point) for point in self._points],
            device=sigma.device,
            dtype=torch.float32,
        )
        ys = torch.tensor(values, device=sigma.device, dtype=torch.float32)

        flat_sigma = work_sigma.reshape(-1).clamp(0.0, 1.0)
        indices = torch.searchsorted(xs, flat_sigma, right=True)
        lower = (indices - 1).clamp(0, len(xs) - 1)
        upper = indices.clamp(0, len(xs) - 1)
        x0, x1 = xs[lower], xs[upper]
        y0, y1 = ys[lower], ys[upper]
        weight = torch.where(
            x1 > x0,
            (flat_sigma - x0) / (x1 - x0).clamp_min(torch.finfo(xs.dtype).eps),
            torch.zeros_like(flat_sigma),
        )
        return (y0 + weight * (y1 - y0)).reshape(sigma.shape).to(output_dtype)


def guidance_loss_weights(scales: torch.Tensor, mode: str) -> torch.Tensor:
    """Return loss weights that compensate for CFG target extrapolation."""
    if mode == "none":
        return torch.ones_like(scales)
    safe_scales = scales.abs().clamp_min(1.0)
    if mode == "inverse":
        return safe_scales.reciprocal()
    if mode == "inverse_square":
        return safe_scales.square().reciprocal()
    raise ValueError(
        "guidance_loss_weighting must be one of: none, inverse, inverse_square"
    )
