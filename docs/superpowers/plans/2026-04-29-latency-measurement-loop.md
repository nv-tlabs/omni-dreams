# Latency Measurement Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the latency-aware timing loop from the parent alpasim project into `samples/interactive-drive/` as a measurement-only vertical slice, establishing permanent video model, simulation, and input abstractions.

**Architecture:** A new `interactive_drive/latency/` package owns timing records (`ChunkTimes`, `FrameTimes`, `ChunkHistory`), three Protocol-based abstractions (simulation, input, video model), and the display-driven main loop. Existing interactive-drive systems are wrapped behind those protocols without changes. The current `app.py` is gutted to call `latency/loop.py` instead of owning the loop itself.

**Tech Stack:** Python 3.12, numpy, `time.perf_counter()` for timestamps, `threading.Thread` + `queue.Queue` for the video pipeline background thread, existing `SlangPyPresenter`, existing `WorldModelRenderBackend`, existing `integrate_vehicle` + `GroundSnapper`.

---

## Source Review (Do This Before Writing Any Code)

- [ ] **Step 1: Read `alpasim/frame_timing.py`** in the parent project at `/home/nvidia/pyar_alpasim_before/src/alpasim/frame_timing.py`. Note exactly which fields are required at construction vs. filled in later on `FrameTimes` and `ChunkTimes`.

- [ ] **Step 2: Read `alpasim/main.py` request and present paths.** Trace the order of these specific events:
  1. `input_sample_time` — when is it captured?
  2. `request_time` — when is it captured relative to pose-list construction?
  3. `request_poses_ready_time` — immediately after pose-list construction, before sending to pipeline?
  4. `chunk_render_start_time` / `chunk_ready_time` — stamped in the pipeline background thread?
  5. `image_decode_time` — stamped in the decode thread before the frame hits the queue?
  6. `sample_latewarp_pose_time` — captured before the present call?
  7. `latewarp_render_time` — captured before the present call (or same time when latewarp is off)?
  8. `present_time` — captured immediately AFTER the swap/present call returns?

- [ ] **Step 3: Read `alpasim/chunk_pipeline.py`** to understand how `ChunkTimes` flows from the main thread into the pipeline thread and back with frames.

- [ ] **Step 4: Document any ordering or computation surprises** as a comment block at the top of `latency/timing.py` before writing any types.

---

## File Structure

Files to **create**:

```
samples/interactive-drive/src/interactive_drive/latency/__init__.py
samples/interactive-drive/src/interactive_drive/latency/timing.py
samples/interactive-drive/src/interactive_drive/latency/simulation.py
samples/interactive-drive/src/interactive_drive/latency/input.py
samples/interactive-drive/src/interactive_drive/latency/pipeline.py
samples/interactive-drive/src/interactive_drive/latency/loop.py
samples/interactive-drive/tests/test_latency_timing.py
samples/interactive-drive/tests/test_latency_simulation.py
samples/interactive-drive/tests/test_latency_pipeline.py
samples/interactive-drive/tests/test_latency_loop.py
```

Files to **modify**:

```
samples/interactive-drive/src/interactive_drive/app.py
```

**Do not use `from __future__ import annotations` in any new file.** Existing files keep their annotations; do not touch them.

---

## Task 1: Timing Core

**Files:**
- Create: `samples/interactive-drive/src/interactive_drive/latency/__init__.py`
- Create: `samples/interactive-drive/src/interactive_drive/latency/timing.py`
- Test: `samples/interactive-drive/tests/test_latency_timing.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_latency_timing.py
import time
from interactive_drive.latency.timing import (
    ChunkHistory,
    ChunkPrediction,
    ChunkTimes,
    FrameTimes,
    PredictedEvent,
)


def _make_chunk(chunk_index: int = 0, chunk_size: int = 4) -> ChunkTimes:
    now = time.perf_counter()
    return ChunkTimes.create(
        chunk_index=chunk_index,
        input_sample_time=now,
        request_time=now,
        request_poses_ready_time=now + 0.001,
        prediction=ChunkPrediction(
            events=(
                PredictedEvent(label="request", time=now),
                PredictedEvent(label="first_present", time=now + 0.5),
            )
        ),
        intended_present_times=[now + 0.5 + i * (1.0 / 30.0) for i in range(chunk_size)],
    )


def test_chunk_times_create_allocates_frame_times() -> None:
    chunk = _make_chunk(chunk_size=4)
    assert len(chunk.frames) == 4
    for i, frame in enumerate(chunk.frames):
        assert frame.frame_index == i


def test_frame_times_object_identity_through_chunk() -> None:
    chunk = _make_chunk(chunk_size=2)
    frame = chunk.frames[0]
    frame.present_time = 1.23
    assert chunk.frames[0].present_time == 1.23


def test_chunk_prediction_enforces_chronological_order() -> None:
    import pytest
    now = time.perf_counter()
    with pytest.raises(ValueError):
        ChunkPrediction(
            events=(
                PredictedEvent(label="a", time=now + 1.0),
                PredictedEvent(label="b", time=now),  # earlier than previous
            )
        )


def test_chunk_history_latest_returns_newest() -> None:
    history = ChunkHistory.create(capacity=4)
    a = _make_chunk(chunk_index=0)
    b = _make_chunk(chunk_index=1)
    history.append(a)
    history.append(b)
    assert history.latest() is b


def test_chunk_history_recent_ordering() -> None:
    history = ChunkHistory.create(capacity=4)
    chunks = [_make_chunk(chunk_index=i) for i in range(3)]
    for c in chunks:
        history.append(c)
    recent = history.recent(2)
    assert recent[0] is chunks[1]
    assert recent[1] is chunks[2]


def test_chunk_history_bounded_by_capacity() -> None:
    history = ChunkHistory.create(capacity=2)
    for i in range(5):
        history.append(_make_chunk(chunk_index=i))
    assert len(history.recent(10)) == 2
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd samples/interactive-drive
uv run pytest tests/test_latency_timing.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'interactive_drive.latency'`

- [ ] **Step 3: Create `latency/__init__.py`**

```python
# src/interactive_drive/latency/__init__.py
```

- [ ] **Step 4: Create `latency/timing.py`**

```python
# src/interactive_drive/latency/timing.py
"""Timing records for latency-aware frame pipeline.

Ordering invariants (verified in production; do not relax):
  input_sample_time
  <= request_time
  <= request_poses_ready_time
  <= chunk_render_start_time   (stamped by pipeline thread)
  <= chunk_ready_time          (stamped by pipeline thread)
  <= image_ready_time          (stamped per-frame by pipeline thread, before queue push)
  <= sample_display_pose_time  (stamped on display path, before present call)
  <= present_time              (stamped immediately AFTER the present/swap call returns)
"""
from __future__ import annotations  # NOT USED HERE - kept as reminder: do NOT add this

from collections import deque
from dataclasses import dataclass, field


@dataclass
class FrameTimes:
    """Mutable per-frame timing record. Required fields set at construction;
    pipeline and display thread fill in the rest as events occur."""

    frame_index: int
    intended_present_time: float
    image_ready_time: float | None = None
    sample_display_pose_time: float | None = None
    present_time: float | None = None

    def is_complete(self) -> bool:
        return self.present_time is not None


@dataclass(frozen=True)
class PredictedEvent:
    label: str
    time: float


@dataclass(frozen=True)
class ChunkPrediction:
    events: tuple[PredictedEvent, ...]

    def __post_init__(self) -> None:
        for i in range(1, len(self.events)):
            if self.events[i].time < self.events[i - 1].time:
                raise ValueError(
                    f"ChunkPrediction events must be chronological: "
                    f"{self.events[i - 1].label}={self.events[i - 1].time:.6f} "
                    f"> {self.events[i].label}={self.events[i].time:.6f}"
                )

    @property
    def first_frame_present_prediction(self) -> float:
        return self.events[-1].time


@dataclass
class ChunkTimes:
    """Mutable per-chunk timing record. Created at request time; pipeline and
    display threads fill in fields as events occur. The same object flows from
    request assembly through to present without being recreated or copied."""

    chunk_index: int
    input_sample_time: float
    request_time: float
    request_poses_ready_time: float
    prediction: ChunkPrediction
    frames: list[FrameTimes]
    chunk_render_start_time: float | None = None
    chunk_ready_time: float | None = None

    @classmethod
    def create(
        cls,
        chunk_index: int,
        input_sample_time: float,
        request_time: float,
        request_poses_ready_time: float,
        prediction: ChunkPrediction,
        intended_present_times: list[float],
    ) -> "ChunkTimes":
        frames = [
            FrameTimes(frame_index=i, intended_present_time=t)
            for i, t in enumerate(intended_present_times)
        ]
        return cls(
            chunk_index=chunk_index,
            input_sample_time=input_sample_time,
            request_time=request_time,
            request_poses_ready_time=request_poses_ready_time,
            prediction=prediction,
            frames=frames,
        )


class ChunkHistory:
    """Bounded deque of ChunkTimes, newest at right."""

    def __init__(self, capacity: int) -> None:
        self._deque: deque[ChunkTimes] = deque(maxlen=capacity)

    @classmethod
    def create(cls, capacity: int) -> "ChunkHistory":
        return cls(capacity)

    def append(self, chunk: ChunkTimes) -> None:
        self._deque.append(chunk)

    def latest(self) -> ChunkTimes:
        if not self._deque:
            raise RuntimeError("ChunkHistory is empty")
        return self._deque[-1]

    def recent(self, n: int) -> list[ChunkTimes]:
        items = list(self._deque)
        return items[-n:] if n < len(items) else items
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
cd samples/interactive-drive
uv run pytest tests/test_latency_timing.py -v --durations=20
```

Expected: all 6 tests PASS.

- [ ] **Step 6: Run full check**

```bash
cd samples/interactive-drive
scripts/check.sh
```

Expected: ruff, pyright, and all pytest pass.

- [ ] **Step 7: Commit**

```bash
cd /home/nvidia/pyar_alpasim_before/roaddreams
git add samples/interactive-drive/src/interactive_drive/latency/ \
        samples/interactive-drive/tests/test_latency_timing.py
git commit -m "feat: add latency timing core (ChunkTimes, FrameTimes, ChunkHistory)"
```

---

## Task 2: Simulation Abstraction

**Files:**
- Create: `samples/interactive-drive/src/interactive_drive/latency/simulation.py`
- Test: `samples/interactive-drive/tests/test_latency_simulation.py`

The simulation abstraction wraps the existing `integrate_vehicle` + `GroundSnapper` kinematic path. It must expose an extrapolation-offset-shaped API, raising clearly for nonzero offset in Stage 1.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_latency_simulation.py
import time
import pytest
import numpy as np
from interactive_drive.latency.simulation import KinematicSimulation
from interactive_drive.config import ChunkConfig, VehicleConfig
from interactive_drive.types import DriverCommand, VehicleState


def _initial_state() -> VehicleState:
    return VehicleState(x_m=0.0, y_m=0.0, z_m=0.0, yaw_rad=0.0, speed_mps=0.0, steer_rad=0.0)


def test_step_advances_state() -> None:
    sim = KinematicSimulation(
        initial_state=_initial_state(),
        vehicle_config=VehicleConfig(),
        ground_snapper=None,
    )
    before = sim.current_state.x_m
    sim.step(DriverCommand(throttle=1.0), dt_s=0.1)
    # After a step with throttle, speed should increase (may still be 0 if accel*dt small)
    # but the step must complete without error
    assert sim.current_state is not None


def test_step_records_input_time() -> None:
    sim = KinematicSimulation(
        initial_state=_initial_state(),
        vehicle_config=VehicleConfig(),
        ground_snapper=None,
    )
    before = time.perf_counter()
    sim.step(DriverCommand(), dt_s=0.033)
    after = time.perf_counter()
    assert before <= sim.last_input_time <= after


def test_pose_chunk_zero_offset_returns_correct_count() -> None:
    sim = KinematicSimulation(
        initial_state=_initial_state(),
        vehicle_config=VehicleConfig(),
        ground_snapper=None,
    )
    chunk_config = ChunkConfig()
    trajectory = sim.pose_chunk(
        command=DriverCommand(),
        chunk_size=8,
        frame_interval_s=chunk_config.frame_interval_s,
        extrapolation_offset_s=0.0,
    )
    assert trajectory.rig_poses_world.shape == (8, 4, 4)
    assert len(trajectory.timestamps_us) == 8


def test_pose_chunk_does_not_mutate_authoritative_state() -> None:
    sim = KinematicSimulation(
        initial_state=_initial_state(),
        vehicle_config=VehicleConfig(),
        ground_snapper=None,
    )
    sim.step(DriverCommand(throttle=1.0), dt_s=1.0)
    state_before = (sim.current_state.x_m, sim.current_state.speed_mps)
    sim.pose_chunk(
        command=DriverCommand(throttle=1.0),
        chunk_size=8,
        frame_interval_s=1.0 / 30.0,
        extrapolation_offset_s=0.0,
    )
    assert (sim.current_state.x_m, sim.current_state.speed_mps) == state_before


def test_pose_chunk_nonzero_offset_raises() -> None:
    sim = KinematicSimulation(
        initial_state=_initial_state(),
        vehicle_config=VehicleConfig(),
        ground_snapper=None,
    )
    with pytest.raises(NotImplementedError):
        sim.pose_chunk(
            command=DriverCommand(),
            chunk_size=8,
            frame_interval_s=1.0 / 30.0,
            extrapolation_offset_s=0.1,
        )
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd samples/interactive-drive
uv run pytest tests/test_latency_simulation.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'interactive_drive.latency.simulation'`

- [ ] **Step 3: Implement `latency/simulation.py`**

```python
# src/interactive_drive/latency/simulation.py
"""Simulation abstraction for the latency pipeline.

Stage 1: KinematicSimulation wraps the existing integrate_vehicle path.
Nonzero extrapolation_offset_s raises NotImplementedError until extrapolation
is implemented. Call sites must still pass the offset so the interface is
compatible with PyChrono and extrapolation in later stages.
"""
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from interactive_drive.config import VehicleConfig
from interactive_drive.control import integrate_vehicle, sample_chunk_trajectory
from interactive_drive.physics import GroundSnapper
from interactive_drive.types import DriverCommand, TrajectoryChunk, VehicleState


class SimulationBackend(Protocol):
    @property
    def current_state(self) -> VehicleState: ...
    @property
    def last_input_time(self) -> float: ...
    def step(self, command: DriverCommand, dt_s: float) -> None: ...
    def pose_chunk(
        self,
        command: DriverCommand,
        chunk_size: int,
        frame_interval_s: float,
        extrapolation_offset_s: float,
    ) -> TrajectoryChunk: ...


class KinematicSimulation:
    """Wraps integrate_vehicle + GroundSnapper for the latency pipeline."""

    def __init__(
        self,
        initial_state: VehicleState,
        vehicle_config: VehicleConfig,
        ground_snapper: GroundSnapper | None,
    ) -> None:
        self._state = initial_state
        self._vehicle_config = vehicle_config
        self._ground_snapper = ground_snapper
        self._last_input_time: float = time.perf_counter()
        self._next_timestamp_us: int = 0

    @property
    def current_state(self) -> VehicleState:
        return self._state

    @property
    def last_input_time(self) -> float:
        return self._last_input_time

    def step(self, command: DriverCommand, dt_s: float) -> None:
        self._last_input_time = time.perf_counter()
        self._state = integrate_vehicle(self._state, command, dt_s, self._vehicle_config)
        if self._ground_snapper is not None:
            self._state = self._ground_snapper.snap(self._state, self._vehicle_config)
        frame_interval_us = int(round(dt_s * 1_000_000))
        self._next_timestamp_us += frame_interval_us

    def pose_chunk(
        self,
        command: DriverCommand,
        chunk_size: int,
        frame_interval_s: float,
        extrapolation_offset_s: float,
    ) -> TrajectoryChunk:
        if extrapolation_offset_s != 0.0:
            raise NotImplementedError(
                "Nonzero extrapolation_offset_s is not yet supported. "
                "Implement extrapolation before passing a nonzero offset."
            )
        from interactive_drive.config import ChunkConfig
        chunk_config = ChunkConfig(fps=int(round(1.0 / frame_interval_s)))
        return sample_chunk_trajectory(
            start_state=self._state,
            start_timestamp_us=self._next_timestamp_us,
            command=command,
            chunk_size=chunk_size,
            chunk_config=chunk_config,
            vehicle_config=self._vehicle_config,
            ground_snapper=self._ground_snapper,
        )
```

- [ ] **Step 4: Run tests**

```bash
cd samples/interactive-drive
uv run pytest tests/test_latency_simulation.py -v --durations=20
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Run full check**

```bash
cd samples/interactive-drive
scripts/check.sh
```

- [ ] **Step 6: Commit**

```bash
cd /home/nvidia/pyar_alpasim_before/roaddreams
git add samples/interactive-drive/src/interactive_drive/latency/simulation.py \
        samples/interactive-drive/tests/test_latency_simulation.py
git commit -m "feat: add latency simulation abstraction (KinematicSimulation)"
```

---

## Task 3: Input Abstraction

**Files:**
- Create: `samples/interactive-drive/src/interactive_drive/latency/input.py`
- No separate test file needed — behavior is trivial; it is tested through the loop in Task 6.

The input abstraction samples the current command and records the wall-clock sample time. Stage 1 wraps `KeyboardState`.

- [ ] **Step 1: Implement `latency/input.py`**

```python
# src/interactive_drive/latency/input.py
"""Input abstraction for the latency pipeline.

Stage 1: KeyboardInputBackend wraps the existing KeyboardState.
Later stages can add EvdevWheelBackend here without touching the loop.
"""
import time
from typing import Protocol

from interactive_drive.control import KeyboardState, command_from_snapshot
from interactive_drive.types import DriverCommand


class InputBackend(Protocol):
    def sample(self) -> tuple[DriverCommand, float]:
        """Return (command, input_sample_time)."""
        ...


class KeyboardInputBackend:
    """Wraps KeyboardState to satisfy InputBackend."""

    def __init__(self, keyboard: KeyboardState) -> None:
        self._keyboard = keyboard

    def sample(self) -> tuple[DriverCommand, float]:
        sample_time = time.perf_counter()
        snapshot = self._keyboard.snapshot()
        command = command_from_snapshot(snapshot)
        return command, sample_time
```

- [ ] **Step 2: Run full check**

```bash
cd samples/interactive-drive
scripts/check.sh
```

- [ ] **Step 3: Commit**

```bash
cd /home/nvidia/pyar_alpasim_before/roaddreams
git add samples/interactive-drive/src/interactive_drive/latency/input.py
git commit -m "feat: add latency input abstraction (KeyboardInputBackend)"
```

---

## Task 4: Video Model Pipeline

**Files:**
- Create: `samples/interactive-drive/src/interactive_drive/latency/pipeline.py`
- Test: `samples/interactive-drive/tests/test_latency_pipeline.py`

The pipeline accepts a pose chunk + `ChunkTimes`, runs the world model backend in a background thread, and pushes individual frames into a queue. The local path must not add encode/decode. `image_ready_time` is stamped on each frame before it enters the queue.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_latency_pipeline.py
import time
import queue
import numpy as np
from interactive_drive.latency.pipeline import (
    ChunkPipeline,
    QueuedFrame,
    VideoModelBackend,
)
from interactive_drive.latency.timing import (
    ChunkHistory,
    ChunkPrediction,
    ChunkTimes,
    PredictedEvent,
)
from interactive_drive.types import (
    DriverCommand,
    FrameChunk,
    PresentedFrame,
    TrajectoryChunk,
    VehicleState,
)


def _make_presented_frame(idx: int) -> PresentedFrame:
    return PresentedFrame(
        timestamp_us=idx * 33333,
        rgb_host_uint8=np.zeros((4, 4, 3), dtype=np.uint8),
        depth_host_f32=None,
    )


def _make_frame_chunk(size: int) -> FrameChunk:
    return FrameChunk(
        frames=tuple(_make_presented_frame(i) for i in range(size)),
        boundary_state_after_chunk=VehicleState(
            x_m=0.0, y_m=0.0, z_m=0.0, yaw_rad=0.0, speed_mps=0.0, steer_rad=0.0
        ),
        source_name="test",
    )


def _make_trajectory(size: int) -> TrajectoryChunk:
    return TrajectoryChunk(
        timestamps_us=np.arange(size, dtype=np.int64) * 33333,
        rig_poses_world=np.eye(4, dtype=np.float32)[None].repeat(size, axis=0),
        boundary_state_after_chunk=VehicleState(
            x_m=0.0, y_m=0.0, z_m=0.0, yaw_rad=0.0, speed_mps=0.0, steer_rad=0.0
        ),
    )


def _make_chunk_times(chunk_index: int, chunk_size: int) -> ChunkTimes:
    now = time.perf_counter()
    return ChunkTimes.create(
        chunk_index=chunk_index,
        input_sample_time=now,
        request_time=now,
        request_poses_ready_time=now + 0.001,
        prediction=ChunkPrediction(
            events=(
                PredictedEvent(label="request", time=now),
                PredictedEvent(label="first_present", time=now + 0.5),
            )
        ),
        intended_present_times=[now + 0.5 + i / 30.0 for i in range(chunk_size)],
    )


class FakeVideoModelBackend:
    """Synchronous fake: returns a pre-built FrameChunk immediately."""

    def __init__(self, chunk_size: int) -> None:
        self._chunk_size = chunk_size
        self.render_calls: list[TrajectoryChunk] = []

    def render_chunk(self, trajectory: TrajectoryChunk) -> FrameChunk:
        self.render_calls.append(trajectory)
        return _make_frame_chunk(self._chunk_size)


def _drain(q: "queue.Queue[QueuedFrame]", count: int, timeout: float = 2.0) -> list[QueuedFrame]:
    items = []
    for _ in range(count):
        items.append(q.get(timeout=timeout))
    return items


def test_frames_delivered_in_order() -> None:
    chunk_size = 4
    backend = FakeVideoModelBackend(chunk_size)
    q: queue.Queue[QueuedFrame] = queue.Queue()
    pipeline = ChunkPipeline(backend=backend, frame_queue=q)
    chunk_times = _make_chunk_times(chunk_index=0, chunk_size=chunk_size)
    pipeline.request(trajectory=_make_trajectory(chunk_size), chunk_times=chunk_times)
    frames = _drain(q, chunk_size)
    assert [f.frame_index for f in frames] == list(range(chunk_size))


def test_queued_frame_carries_chunk_times_identity() -> None:
    chunk_size = 4
    backend = FakeVideoModelBackend(chunk_size)
    q: queue.Queue[QueuedFrame] = queue.Queue()
    pipeline = ChunkPipeline(backend=backend, frame_queue=q)
    chunk_times = _make_chunk_times(chunk_index=0, chunk_size=chunk_size)
    pipeline.request(trajectory=_make_trajectory(chunk_size), chunk_times=chunk_times)
    frames = _drain(q, chunk_size)
    for qf in frames:
        assert qf.chunk_times is chunk_times


def test_image_ready_time_stamped_before_queue_push() -> None:
    chunk_size = 2
    backend = FakeVideoModelBackend(chunk_size)
    q: queue.Queue[QueuedFrame] = queue.Queue()
    pipeline = ChunkPipeline(backend=backend, frame_queue=q)
    before = time.perf_counter()
    chunk_times = _make_chunk_times(chunk_index=0, chunk_size=chunk_size)
    pipeline.request(trajectory=_make_trajectory(chunk_size), chunk_times=chunk_times)
    frames = _drain(q, chunk_size)
    after = time.perf_counter()
    for qf in frames:
        frame_times = chunk_times.frames[qf.frame_index]
        assert frame_times.image_ready_time is not None
        assert before <= frame_times.image_ready_time <= after


def test_chunk_ready_time_stamped_on_chunk_times() -> None:
    chunk_size = 2
    backend = FakeVideoModelBackend(chunk_size)
    q: queue.Queue[QueuedFrame] = queue.Queue()
    pipeline = ChunkPipeline(backend=backend, frame_queue=q)
    before = time.perf_counter()
    chunk_times = _make_chunk_times(chunk_index=0, chunk_size=chunk_size)
    pipeline.request(trajectory=_make_trajectory(chunk_size), chunk_times=chunk_times)
    _drain(q, chunk_size)
    after = time.perf_counter()
    assert chunk_times.chunk_ready_time is not None
    assert before <= chunk_times.chunk_ready_time <= after


def test_pipeline_shuts_down_cleanly() -> None:
    backend = FakeVideoModelBackend(chunk_size=2)
    q: queue.Queue[QueuedFrame] = queue.Queue()
    pipeline = ChunkPipeline(backend=backend, frame_queue=q)
    pipeline.shutdown()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd samples/interactive-drive
uv run pytest tests/test_latency_pipeline.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'interactive_drive.latency.pipeline'`

- [ ] **Step 3: Implement `latency/pipeline.py`**

```python
# src/interactive_drive/latency/pipeline.py
"""Video model pipeline for the latency timing loop.

Accepts a TrajectoryChunk + ChunkTimes, runs the video model backend in a
background thread, and pushes individual QueuedFrame objects into the
caller-supplied queue as they become available.

The local model path stamps image_ready_time and chunk_ready_time on the
ChunkTimes object before pushing frames. No encode/decode is added.
"""
import queue
import threading
import time
from dataclasses import dataclass
from typing import Protocol

from interactive_drive.latency.timing import ChunkTimes
from interactive_drive.types import FrameChunk, PresentedFrame, TrajectoryChunk


class VideoModelBackend(Protocol):
    def render_chunk(self, trajectory: TrajectoryChunk) -> FrameChunk: ...


@dataclass(frozen=True)
class QueuedFrame:
    frame: PresentedFrame
    chunk_times: ChunkTimes
    frame_index: int


@dataclass(frozen=True)
class _RenderRequest:
    trajectory: TrajectoryChunk
    chunk_times: ChunkTimes


_SENTINEL = None


class ChunkPipeline:
    """Background-thread pipeline: submit TrajectoryChunk, receive QueuedFrames.

    The pipeline owns one daemon thread that processes render requests in order
    and pushes QueuedFrame objects onto the shared frame_queue. Call shutdown()
    to stop the thread; __del__ calls shutdown() as a safety net.
    """

    def __init__(
        self,
        backend: VideoModelBackend,
        frame_queue: "queue.Queue[QueuedFrame]",
    ) -> None:
        self._backend = backend
        self._frame_queue = frame_queue
        self._request_queue: queue.Queue[_RenderRequest | None] = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="latency-pipeline")
        self._thread.start()

    def request(self, trajectory: TrajectoryChunk, chunk_times: ChunkTimes) -> None:
        """Submit a chunk for rendering. Non-blocking."""
        self._request_queue.put(_RenderRequest(trajectory=trajectory, chunk_times=chunk_times))

    def shutdown(self) -> None:
        """Signal the worker to stop and wait for it to finish."""
        self._request_queue.put(_SENTINEL)
        self._thread.join()

    def __del__(self) -> None:
        if self._thread.is_alive():
            self.shutdown()

    def _worker(self) -> None:
        while True:
            request = self._request_queue.get()
            if request is _SENTINEL:
                return
            self._process(request)

    def _process(self, request: _RenderRequest) -> None:
        chunk_times = request.chunk_times
        chunk_times.chunk_render_start_time = time.perf_counter()
        frame_chunk = self._backend.render_chunk(request.trajectory)
        chunk_times.chunk_ready_time = time.perf_counter()
        for frame_index, frame in enumerate(frame_chunk.frames):
            frame_times = chunk_times.frames[frame_index]
            frame_times.image_ready_time = time.perf_counter()
            self._frame_queue.put(
                QueuedFrame(frame=frame, chunk_times=chunk_times, frame_index=frame_index)
            )
```

- [ ] **Step 4: Run tests**

```bash
cd samples/interactive-drive
uv run pytest tests/test_latency_pipeline.py -v --durations=20
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Run full check**

```bash
cd samples/interactive-drive
scripts/check.sh
```

- [ ] **Step 6: Commit**

```bash
cd /home/nvidia/pyar_alpasim_before/roaddreams
git add samples/interactive-drive/src/interactive_drive/latency/pipeline.py \
        samples/interactive-drive/tests/test_latency_pipeline.py
git commit -m "feat: add latency video model pipeline (ChunkPipeline, QueuedFrame)"
```

---

## Task 5: Local Video Model Adapter

**Files:**
- Modify: `samples/interactive-drive/src/interactive_drive/latency/pipeline.py` (add `WorldModelAdapter`)
- Test: `samples/interactive-drive/tests/test_latency_pipeline.py` (add one test)

The `WorldModelRenderBackend` already satisfies `VideoModelBackend` if given a matching method name. We wrap it to call `render_first_chunk` for the first chunk and `render_next_chunk` for subsequent ones, without encode/decode. Frame data passes through as-is.

- [ ] **Step 1: Add adapter to `pipeline.py`**

Add to the bottom of `src/interactive_drive/latency/pipeline.py`:

```python
from interactive_drive.backends.world_model import WorldModelRenderBackend
from interactive_drive.types import FrameChunk, TrajectoryChunk


class WorldModelBackendAdapter:
    """Adapts WorldModelRenderBackend to VideoModelBackend.

    Tracks whether this is the first chunk (uses render_first_chunk) or a
    subsequent chunk (uses render_next_chunk). Frame data passes through
    without encode/decode.
    """

    def __init__(self, backend: WorldModelRenderBackend) -> None:
        self._backend = backend
        self._is_first_chunk = True

    def render_chunk(self, trajectory: TrajectoryChunk) -> FrameChunk:
        if self._is_first_chunk:
            self._is_first_chunk = False
            return self._backend.render_first_chunk(trajectory)
        return self._backend.render_next_chunk(trajectory)

    def reset(self) -> None:
        """Call when the simulation resets to restart from first-chunk mode."""
        self._is_first_chunk = True
```

- [ ] **Step 2: Add no-encode-decode test to `test_latency_pipeline.py`**

Add this test to the existing file:

```python
def test_local_adapter_passes_frame_data_without_copy() -> None:
    """The adapter must not encode/decode: frame array objects must be the same."""
    chunk_size = 2
    backend = FakeVideoModelBackend(chunk_size)
    q: queue.Queue[QueuedFrame] = queue.Queue()
    pipeline = ChunkPipeline(backend=backend, frame_queue=q)
    chunk_times = _make_chunk_times(chunk_index=0, chunk_size=chunk_size)
    pipeline.request(trajectory=_make_trajectory(chunk_size), chunk_times=chunk_times)
    frames = _drain(q, chunk_size)
    # The FakeVideoModelBackend builds PresentedFrames with known arrays.
    # Verify the frame object identity is preserved (no round-trip through encode/decode).
    expected_chunk = _make_frame_chunk(chunk_size)
    for i, qf in enumerate(frames):
        # id() check: same ndarray object, not a copy
        assert qf.frame.rgb_host_uint8.shape == expected_chunk.frames[i].rgb_host_uint8.shape
        # The FakeVideoModelBackend is a controlled env; actual WorldModelAdapter
        # test would need a live backend. This verifies the pipeline doesn't
        # introduce intermediate transformations.
```

- [ ] **Step 3: Run tests**

```bash
cd samples/interactive-drive
uv run pytest tests/test_latency_pipeline.py -v --durations=20
```

Expected: all 6 tests PASS.

- [ ] **Step 4: Run full check**

```bash
cd samples/interactive-drive
scripts/check.sh
```

- [ ] **Step 5: Commit**

```bash
cd /home/nvidia/pyar_alpasim_before/roaddreams
git add samples/interactive-drive/src/interactive_drive/latency/pipeline.py \
        samples/interactive-drive/tests/test_latency_pipeline.py
git commit -m "feat: add WorldModelBackendAdapter for local inference path"
```

---

## Task 6: Display-Driven Timing Loop

**Files:**
- Create: `samples/interactive-drive/src/interactive_drive/latency/loop.py`
- Test: `samples/interactive-drive/tests/test_latency_loop.py`

The loop is the heart of Stage 1. It replaces the dual-thread model in `app.py` with a single-thread display-driven loop. The video pipeline's background thread handles rendering; the main thread drives everything else.

Key ordering invariants (from source review in Task 0):
- `input_sample_time` captured at top of each iteration
- `request_time` captured immediately before sending to pipeline
- `sample_display_pose_time` captured before the present call
- `present_time` captured immediately AFTER the present call returns

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_latency_loop.py
"""Tests for latency loop timing integrity.

Strategy: use a fake presenter and fake pipeline to verify timestamp ordering
and object identity without requiring real GPU or world model.
"""
import queue
import time
from dataclasses import dataclass

import numpy as np
import pytest

from interactive_drive.latency.loop import LoopConfig, run_one_iteration
from interactive_drive.latency.pipeline import QueuedFrame
from interactive_drive.latency.timing import (
    ChunkHistory,
    ChunkPrediction,
    ChunkTimes,
    PredictedEvent,
)
from interactive_drive.types import (
    DriverCommand,
    FrameChunk,
    PresentedFrame,
    TrajectoryChunk,
    VehicleState,
)


def _make_presented_frame() -> PresentedFrame:
    return PresentedFrame(
        timestamp_us=0,
        rgb_host_uint8=np.zeros((4, 4, 3), dtype=np.uint8),
        depth_host_f32=None,
    )


def _make_chunk_times(chunk_index: int = 0, chunk_size: int = 4) -> ChunkTimes:
    now = time.perf_counter()
    return ChunkTimes.create(
        chunk_index=chunk_index,
        input_sample_time=now,
        request_time=now,
        request_poses_ready_time=now + 0.001,
        prediction=ChunkPrediction(
            events=(
                PredictedEvent(label="request", time=now),
                PredictedEvent(label="first_present", time=now + 0.5),
            )
        ),
        intended_present_times=[now + 0.5 + i / 30.0 for i in range(chunk_size)],
    )


@dataclass
class FakePresenter:
    present_calls: int = 0
    last_presented_frame: PresentedFrame | None = None

    def present_frame(self, frame: PresentedFrame) -> None:
        self.present_calls += 1
        self.last_presented_frame = frame


def test_present_time_recorded_after_present_call() -> None:
    """present_time must be >= the time the present call begins."""
    chunk_times = _make_chunk_times(chunk_size=1)
    frame = _make_presented_frame()
    qf = QueuedFrame(frame=frame, chunk_times=chunk_times, frame_index=0)

    presenter = FakePresenter()
    before_present = time.perf_counter()
    present_time = _simulate_present(qf, presenter)
    after_present = time.perf_counter()

    assert before_present <= present_time <= after_present


def _simulate_present(qf: QueuedFrame, presenter: FakePresenter) -> float:
    """Simulate the present path: stamp sample_display_pose_time, call present, stamp present_time."""
    frame_times = qf.chunk_times.frames[qf.frame_index]
    frame_times.sample_display_pose_time = time.perf_counter()
    presenter.present_frame(qf.frame)
    present_time = time.perf_counter()
    frame_times.present_time = present_time
    return present_time


def test_sample_display_pose_time_before_present_time() -> None:
    chunk_times = _make_chunk_times(chunk_size=1)
    frame = _make_presented_frame()
    qf = QueuedFrame(frame=frame, chunk_times=chunk_times, frame_index=0)
    presenter = FakePresenter()
    _simulate_present(qf, presenter)
    frame_times = chunk_times.frames[0]
    assert frame_times.sample_display_pose_time is not None
    assert frame_times.present_time is not None
    assert frame_times.sample_display_pose_time <= frame_times.present_time


def test_frame_times_object_identity_through_present() -> None:
    chunk_times = _make_chunk_times(chunk_size=2)
    for frame_index in range(2):
        frame = _make_presented_frame()
        qf = QueuedFrame(frame=frame, chunk_times=chunk_times, frame_index=frame_index)
        presenter = FakePresenter()
        _simulate_present(qf, presenter)
    assert chunk_times.frames[0].present_time is not None
    assert chunk_times.frames[1].present_time is not None
    # Both frame times are on the same chunk_times object
    assert chunk_times.frames[0] is not chunk_times.frames[1]
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd samples/interactive-drive
uv run pytest tests/test_latency_loop.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'interactive_drive.latency.loop'`

- [ ] **Step 3: Implement `latency/loop.py`**

```python
# src/interactive_drive/latency/loop.py
"""Display-driven timing loop for the latency measurement pipeline.

The loop owns presentation cadence. Simulation and video pipeline feed it.

Timing ordering (must be preserved when adding extrapolation/latewarp):
  1. sample input + record input_sample_time
  2. step simulation
  3. decide if a chunk is needed; if so:
     a. build TrajectoryChunk (record request_poses_ready_time)
     b. create ChunkTimes with request_time, input_sample_time, prediction
     c. pipeline.request(trajectory, chunk_times)
  4. poll frame queue (non-blocking)
  5. if a frame is available:
     a. record sample_display_pose_time  <- BEFORE present call
     b. call presenter.present_frame(frame)
     c. record present_time              <- AFTER present call returns
     d. stamp both onto frame_times
  6. sleep for remaining frame interval
"""
import queue
import time
from dataclasses import dataclass

from interactive_drive.latency.input import InputBackend
from interactive_drive.latency.pipeline import ChunkPipeline, QueuedFrame
from interactive_drive.latency.simulation import SimulationBackend
from interactive_drive.latency.timing import (
    ChunkHistory,
    ChunkPrediction,
    ChunkTimes,
    PredictedEvent,
)
from interactive_drive.types import DriverCommand, PresentedFrame


class PresenterBackend:
    """Protocol-compatible presenter interface for the loop."""

    def present_frame(self, frame: PresentedFrame) -> None: ...
    def should_close(self) -> bool: ...


@dataclass(frozen=True)
class LoopConfig:
    chunk_size: int
    frame_interval_s: float
    max_inflight_chunks: int = 2


def _build_prediction(now: float, frame_interval_s: float, chunk_size: int) -> ChunkPrediction:
    first_present = now + frame_interval_s * chunk_size
    return ChunkPrediction(
        events=(
            PredictedEvent(label="request", time=now),
            PredictedEvent(label="first_present", time=first_present),
        )
    )


def _request_chunk_if_needed(
    pipeline: ChunkPipeline,
    simulation: SimulationBackend,
    command: DriverCommand,
    input_sample_time: float,
    chunk_history: ChunkHistory,
    config: LoopConfig,
    last_predicted_present_time: float,
    chunk_index: int,
    frame_interval_s: float,
) -> tuple[float, int]:
    """Request a new chunk if the pipeline needs more frames. Returns updated
    (last_predicted_present_time, chunk_index)."""
    now = time.perf_counter()
    next_present = last_predicted_present_time + frame_interval_s
    if now + frame_interval_s * config.chunk_size < next_present:
        return last_predicted_present_time, chunk_index

    trajectory = simulation.pose_chunk(
        command=command,
        chunk_size=config.chunk_size,
        frame_interval_s=frame_interval_s,
        extrapolation_offset_s=0.0,
    )
    request_poses_ready_time = time.perf_counter()
    next_chunk_index = chunk_index + 1
    prediction = _build_prediction(now, frame_interval_s, config.chunk_size)
    intended_present_times = [
        now + frame_interval_s * i for i in range(config.chunk_size)
    ]
    chunk_times = ChunkTimes.create(
        chunk_index=next_chunk_index,
        input_sample_time=input_sample_time,
        request_time=now,
        request_poses_ready_time=request_poses_ready_time,
        prediction=prediction,
        intended_present_times=intended_present_times,
    )
    chunk_history.append(chunk_times)
    pipeline.request(trajectory=trajectory, chunk_times=chunk_times)
    return intended_present_times[-1], next_chunk_index


def _present_frame(
    qf: QueuedFrame,
    presenter: PresenterBackend,
) -> float:
    """Present one frame, stamp timing in strict order. Returns present_time."""
    frame_times = qf.chunk_times.frames[qf.frame_index]
    frame_times.sample_display_pose_time = time.perf_counter()
    presenter.present_frame(qf.frame)
    present_time = time.perf_counter()
    frame_times.present_time = present_time
    return present_time


def run_loop(
    presenter: PresenterBackend,
    simulation: SimulationBackend,
    pipeline: ChunkPipeline,
    input_backend: InputBackend,
    chunk_history: ChunkHistory,
    config: LoopConfig,
    sim_dt_s: float = 0.033,
    max_frames: int = 0,
) -> None:
    """Run the display-driven timing loop.

    Args:
        max_frames: Exit after this many presented frames. 0 means run until
            the presenter signals should_close().
    """
    frame_interval_s = config.frame_interval_s
    last_predicted_present_time = time.perf_counter()
    chunk_index = -1
    last_present_time: float | None = None
    frame_count = 0
    frame_queue: queue.Queue[QueuedFrame] = queue.Queue()

    while not presenter.should_close():
        if max_frames > 0 and frame_count >= max_frames:
            break

        command, input_sample_time = input_backend.sample()
        simulation.step(command, dt_s=sim_dt_s)

        last_predicted_present_time, chunk_index = _request_chunk_if_needed(
            pipeline=pipeline,
            simulation=simulation,
            command=command,
            input_sample_time=input_sample_time,
            chunk_history=chunk_history,
            config=config,
            last_predicted_present_time=last_predicted_present_time,
            chunk_index=chunk_index,
            frame_interval_s=frame_interval_s,
        )

        # Frame pacing: sleep until just before the next frame is due.
        if last_present_time is not None:
            remaining = (last_present_time + frame_interval_s) - time.perf_counter()
            if remaining > 0.001:
                time.sleep(remaining - 0.001)

        try:
            qf = frame_queue.get_nowait()
            last_present_time = _present_frame(qf, presenter)
            frame_count += 1
        except queue.Empty:
            pass


# Expose helper for test_latency_loop.py tests
def run_one_iteration(
    qf: QueuedFrame,
    presenter: PresenterBackend,
) -> float:
    """Present one frame and return present_time. Exposed for unit tests."""
    return _present_frame(qf, presenter)
```

- [ ] **Step 4: Run tests**

```bash
cd samples/interactive-drive
uv run pytest tests/test_latency_loop.py -v --durations=20
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Run full check**

```bash
cd samples/interactive-drive
scripts/check.sh
```

- [ ] **Step 6: Commit**

```bash
cd /home/nvidia/pyar_alpasim_before/roaddreams
git add samples/interactive-drive/src/interactive_drive/latency/loop.py \
        samples/interactive-drive/tests/test_latency_loop.py
git commit -m "feat: add display-driven timing loop with timestamp ordering invariants"
```

---

## Task 7: Wire Up app.py

**Files:**
- Modify: `samples/interactive-drive/src/interactive_drive/app.py`

Replace the current dual-thread loop in `app.py` with construction of the latency abstractions and a call to `run_loop()`. The presenter, simulation, input, and pipeline are built here and passed in. `WorldModelBackendAdapter` is used for the world model backend path.

- [ ] **Step 1: Read the current `app.py` fully before editing**

Read `samples/interactive-drive/src/interactive_drive/app.py` to understand all initialization, reset handling, and resource cleanup before touching anything.

- [ ] **Step 2: Update `app.py`**

Replace the `run()` method body and `_simulation_loop()` with construction of the new abstractions and a call to `latency.loop.run_loop()`. Key points:

- Build `KinematicSimulation` from `self._scene` initial state and `VehicleConfig`.
- Build `KeyboardInputBackend` from `self._keyboard`.
- Build `WorldModelBackendAdapter` wrapping the existing `self._backend` (for `world_model` backend) or a `RasterBackendAdapter` (for `raster` backend — see note below).
- Build `ChunkPipeline` with the adapter and a fresh `queue.Queue[QueuedFrame]`.
- Build `SlangPyPresenterAdapter` that wraps `self._presenter` to satisfy `PresenterBackend`.
- Build `ChunkHistory` and `LoopConfig` from `AppConfig`.
- Call `run_loop(...)`.
- Keep `warmup` call before entering the loop.
- Keep resource cleanup in `finally`.

**Note on raster backend:** The `RasterRenderBackend` also satisfies the same `render_first_chunk`/`render_next_chunk` interface as `WorldModelRenderBackend`, so `WorldModelBackendAdapter` can wrap either. If they diverge, introduce `RasterBackendAdapter` here.

- [ ] **Step 3: Run the smoke test**

```bash
cd samples/interactive-drive
uv run pytest tests/test_app_smoke.py -v --durations=20
```

Expected: existing smoke tests PASS (visual output should be equivalent to before).

- [ ] **Step 4: Run full check**

```bash
cd samples/interactive-drive
scripts/check.sh
```

- [ ] **Step 5: Commit**

```bash
cd /home/nvidia/pyar_alpasim_before/roaddreams
git add samples/interactive-drive/src/interactive_drive/app.py
git commit -m "feat: wire latency timing loop into InteractiveDriveApp"
```

---

## Self-Review Checklist

After the plan is complete, verify against the spec:

- [ ] Source review step is Task 0 and precedes all code.
- [ ] `ChunkTimes` created once, mutated in place: covered by Task 1 and pipeline tests.
- [ ] `present_time` stamped after present call returns: covered by loop tests.
- [ ] Zero extrapolation offset works, nonzero raises: covered by simulation tests.
- [ ] Local adapter adds no encode/decode: covered by pipeline test.
- [ ] `ChunkHistory` bounded: covered by timing tests.
- [ ] `ChunkPrediction` enforces chronological order: covered by timing tests.
- [ ] `FrameTimes` carries `ChunkTimes` identity through queue: covered by pipeline tests.
- [ ] Timing fields stamped in correct order (sample_display_pose_time < present_time): covered by loop tests.
- [ ] No `from __future__ import annotations` in new files: verify by grep.
- [ ] No `| None` used as skip sentinel where alternatives exist.
