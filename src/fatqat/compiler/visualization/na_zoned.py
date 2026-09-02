"""Read-only Matplotlib animation for neutral-atom zoned plans.

The visual behavior is derived in part from the bundled ZAP animator,
copyright (c) 2026 北京量子信息科学研究院-量子操作系统组, used under the
MIT License.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from matplotlib.artist import Artist
from matplotlib.patches import Circle, Rectangle

from ...operations.fixed_gates import CZGate
from ..algorithms.zap import architecture_sites
from ..errors import UnsupportedFeatureError
from ..dialects.na_zoned import (
    CrosstalkEvent,
    GateBatch,
    MoveEvent,
    Position,
    TransferEvent,
    ZonedPlan,
    verify_zoned_plan,
)

_CANVAS_PADDING = 10.0
_RYDBERG_PADDING = 5.0
_POINTS_PER_MICRON = 8.0
_TRAP_RADIUS = 1.0
_HIGHLIGHT_RADIUS = 2.0


@dataclass(frozen=True, slots=True)
class _FrameSpan:
    event_index: int | None
    kind: str
    begin: int
    end: int


def create_na_animation(
    plan: ZonedPlan,
    architecture: Mapping[str, object],
    *,
    fps: int = 30,
) -> FuncAnimation:
    """Construct a read-only animation of a verified neutral-atom plan."""

    verify_zoned_plan(plan)
    _verify_fps(fps)
    return _NAZonedAnimator(plan, architecture, fps).create_animation()


def save_na_animation(
    animation: FuncAnimation,
    output_path: str | Path,
) -> None:
    """Save a neutral-atom animation to an explicitly requested MP4 path."""

    path = Path(output_path)
    if path.suffix.lower() != ".mp4":
        raise ValueError("neutral-atom animation output must be an .mp4 file")
    if not path.parent.is_dir():
        raise FileNotFoundError(
            f"animation output directory does not exist: {path.parent}"
        )
    if not writers.is_available("ffmpeg"):
        raise UnsupportedFeatureError(
            "saving neutral-atom animation requires FFmpeg on PATH"
        )
    animation.save(path, writer="ffmpeg")


def _build_frame_spans(plan: ZonedPlan, fps: int) -> tuple[_FrameSpan, ...]:
    _verify_fps(fps)
    microseconds_per_frame = 50.0 / fps
    initial_frames = max(1, fps // 5)
    spans = [_FrameSpan(None, "initial", 0, initial_frames)]
    frame = initial_frames

    for event_index, event in enumerate(plan.events):
        if type(event) is TransferEvent:
            kind = event.kind
            frame_count = 1
        elif type(event) is MoveEvent:
            kind = event.kind
            frame_count = _duration_frames(event.durations, microseconds_per_frame)
        elif type(event) is GateBatch:
            kind = "gate_batch"
            frame_count = 8
        elif type(event) is CrosstalkEvent:
            kind = "crosstalk"
            frame_count = _duration_frames(event.durations, microseconds_per_frame)
        else:
            raise TypeError(f"unsupported ZonedPlan event: {type(event).__name__}")
        spans.append(_FrameSpan(event_index, kind, frame, frame + frame_count))
        frame += frame_count

    return tuple(spans)


def _verify_fps(fps: int) -> None:
    if type(fps) is not int or fps <= 0:
        raise ValueError("fps must be a positive integer")


def _duration_frames(
    durations: tuple[float, ...], microseconds_per_frame: float
) -> int:
    return max(
        1,
        math.ceil(
            max((float(duration) for duration in durations), default=0.0)
            / microseconds_per_frame
        ),
    )


class _NAZonedAnimator:
    def __init__(
        self,
        plan: ZonedPlan,
        architecture: Mapping[str, object],
        fps: int,
    ) -> None:
        self._plan = plan
        self._fps = fps
        self._spans = _build_frame_spans(plan, fps)
        self._storage_sites = tuple(architecture_sites(architecture, "storage_zones"))
        self._entanglement_sites = tuple(
            architecture_sites(architecture, "entanglement_zones")
        )
        self._atom_index = {atom: index for index, atom in enumerate(plan.atoms)}
        self._initial_positions = tuple(
            dict(plan.initial_placement)[atom] for atom in plan.atoms
        )
        self.current_positions = list(self._initial_positions)
        self.gate_highlights: list[Circle] = []
        self.global_rydberg_overlay: Rectangle | None = None
        self._setup_canvas()

    def create_animation(self) -> FuncAnimation:
        return FuncAnimation(
            self.figure,
            self.render_frame,
            init_func=self.initialize,
            frames=self._spans[-1].end,
            interval=1000.0 / self._fps,
            blit=False,
            repeat=False,
        )

    def _setup_canvas(self) -> None:
        sites = self._storage_sites + self._entanglement_sites
        if not sites:
            raise ValueError("architecture must define at least one trap site")
        x_values = [position[0] for position in sites]
        y_values = [position[1] for position in sites]
        self._x_low = min(x_values)
        self._x_high = max(x_values)
        self._y_low = min(y_values)
        self._y_high = max(y_values)

        pixels_per_point = 1.0 / plt.rcParams["figure.dpi"]
        scale = pixels_per_point * _POINTS_PER_MICRON
        width = max(1.0, (self._x_high - self._x_low) * scale)
        height = max(1.0, (self._y_high - self._y_low) * scale)
        self.figure, self.ax = plt.subplots(figsize=(width, height))
        self.title = self.ax.set_title("")
        self.ax.set_xlim(self._x_low - _CANVAS_PADDING, self._x_high + _CANVAS_PADDING)
        self.ax.set_ylim(self._y_low - _CANVAS_PADDING, self._y_high + _CANVAS_PADDING)
        self.ax.set_aspect("equal", adjustable="box")

        for position in sites:
            self.ax.add_patch(
                Circle(
                    position,
                    _TRAP_RADIUS,
                    fill=False,
                    edgecolor="#515252",
                    linewidth=1,
                )
            )

        self.column_guides = [
            self.ax.axvline(position[0], color=(1, 0, 0), alpha=0.0, linestyle="--")
            for position in self._initial_positions
        ]
        self.row_guides = [
            self.ax.axhline(position[1], color=(1, 0, 0), alpha=0.0, linestyle="--")
            for position in self._initial_positions
        ]
        self.atom_scatter = self.ax.scatter(
            [position[0] for position in self._initial_positions],
            [position[1] for position in self._initial_positions],
            s=16,
            color="black",
            edgecolors=[(1, 0, 0, 0) for _ in self._plan.atoms],
        )
        self.atom_labels = [
            self.ax.text(x - 1, y + 1, str(index), fontsize=8)
            for index, (x, y) in enumerate(self._initial_positions)
        ]

    def initialize(self) -> tuple[Artist, ...]:
        self._clear_transient_artists()
        self.current_positions[:] = self._initial_positions
        self.title.set_text("Initial Placement")
        self._sync_positions(frozenset())
        return self._artists()

    def render_frame(self, frame: int) -> tuple[Artist, ...]:
        span = self._span_for_frame(frame)
        self._clear_transient_artists()
        positions, active_atoms = self._state_at(span, frame)
        self.current_positions[:] = positions
        self._sync_positions(active_atoms)

        if span.event_index is None:
            self.title.set_text("Initial Placement")
            return self._artists()

        event = self._plan.events[span.event_index]
        if type(event) is TransferEvent:
            self.title.set_text(event.kind.capitalize())
        elif type(event) is MoveEvent:
            self.title.set_text(event.kind.replace("_", " ").title())
        elif type(event) is GateBatch:
            self._draw_gate_batch(event)
        elif type(event) is CrosstalkEvent:
            self.title.set_text("Crosstalk")
        return self._artists()

    def _span_for_frame(self, frame: int) -> _FrameSpan:
        bounded_frame = min(max(int(frame), 0), self._spans[-1].end - 1)
        return next(
            span for span in self._spans if span.begin <= bounded_frame < span.end
        )

    def _state_at(
        self, span: _FrameSpan, frame: int
    ) -> tuple[list[Position], frozenset[int]]:
        positions = list(self._initial_positions)
        active_atoms: set[int] = set()

        if span.event_index is None:
            return positions, frozenset()

        for event_index, event in enumerate(self._plan.events):
            if event_index > span.event_index:
                break
            if type(event) is TransferEvent:
                indices = {self._atom_index[atom] for atom in event.atoms}
                if event.kind == "activate":
                    active_atoms.update(indices)
                else:
                    active_atoms.difference_update(indices)
            elif type(event) is MoveEvent:
                if event_index < span.event_index:
                    self._set_move_end_positions(positions, event)
                elif event_index == span.event_index:
                    self._set_interpolated_positions(positions, event, span, frame)

        return positions, frozenset(active_atoms)

    def _set_move_end_positions(
        self, positions: list[Position], event: MoveEvent
    ) -> None:
        for atom, end in zip(event.atoms, event.ends, strict=True):
            positions[self._atom_index[atom]] = end

    def _set_interpolated_positions(
        self,
        positions: list[Position],
        event: MoveEvent,
        span: _FrameSpan,
        frame: int,
    ) -> None:
        frame_count = span.end - span.begin
        if frame_count == 1:
            progress = 1.0
        else:
            progress = (frame - span.begin) / (frame_count - 1)
        for atom, start, end in zip(event.atoms, event.starts, event.ends, strict=True):
            positions[self._atom_index[atom]] = (
                _cubic_interpolate(progress, start[0], end[0]),
                _cubic_interpolate(progress, start[1], end[1]),
            )

    def _sync_positions(self, active_atoms: frozenset[int]) -> None:
        self.atom_scatter.set_offsets(self.current_positions)
        self.atom_scatter.set_edgecolors(
            [
                (1, 0, 0, 1) if index in active_atoms else (1, 0, 0, 0)
                for index in range(len(self._plan.atoms))
            ]
        )
        for index, (x, y) in enumerate(self.current_positions):
            self.atom_labels[index].set_position((x - 1, y + 1))
            self.column_guides[index].set_xdata((x, x))
            self.column_guides[index].set_alpha(0.5 if index in active_atoms else 0.0)
            self.row_guides[index].set_ydata((y, y))
            self.row_guides[index].set_alpha(0.5 if index in active_atoms else 0.0)

    def _draw_gate_batch(self, event: GateBatch) -> None:
        self.title.set_text(f"Stage {event.stage}\nGate Batch")
        for gate in event.gates:
            for position in gate.positions:
                highlight = Circle(
                    position,
                    _HIGHLIGHT_RADIUS,
                    color=(0, 0, 1, 0.2),
                )
                self.ax.add_patch(highlight)
                self.gate_highlights.append(highlight)

        if any(type(gate.operation) is CZGate for gate in event.gates):
            self.global_rydberg_overlay = self._new_rydberg_overlay()
            self.ax.add_patch(self.global_rydberg_overlay)

    def _new_rydberg_overlay(self) -> Rectangle:
        if not self._entanglement_sites:
            raise ValueError(
                "architecture must define entanglement sites for a CZ gate"
            )
        x_values = [position[0] for position in self._entanglement_sites]
        y_values = [position[1] for position in self._entanglement_sites]
        x_low, x_high = min(x_values), max(x_values)
        y_low, y_high = min(y_values), max(y_values)
        return Rectangle(
            (x_low - _RYDBERG_PADDING, y_low - _RYDBERG_PADDING),
            x_high - x_low + 2 * _RYDBERG_PADDING,
            y_high - y_low + 2 * _RYDBERG_PADDING,
            linewidth=2,
            facecolor=(0, 0, 1, 0.2),
        )

    def _clear_transient_artists(self) -> None:
        for highlight in self.gate_highlights:
            highlight.remove()
        self.gate_highlights.clear()
        if self.global_rydberg_overlay is not None:
            self.global_rydberg_overlay.remove()
            self.global_rydberg_overlay = None

    def _artists(self) -> tuple[Artist, ...]:
        transient: list[Artist] = list(self.gate_highlights)
        if self.global_rydberg_overlay is not None:
            transient.append(self.global_rydberg_overlay)
        return (
            self.atom_scatter,
            *self.column_guides,
            *self.row_guides,
            *self.atom_labels,
            *transient,
            self.title,
        )


def _cubic_interpolate(progress: float, begin: float, end: float) -> float:
    distance = end - begin
    return begin + 3 * distance * progress**2 - 2 * distance * progress**3
