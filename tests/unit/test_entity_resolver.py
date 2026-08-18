"""Unit tests for deterministic EntityResolver behavior."""

from __future__ import annotations

import pytest

from qwen_stream_video.config import EntityRegistryConfig
from qwen_stream_video.domain import (
    EntityObservation,
    EntityResolutionStatus,
    EntityType,
    GlobalState,
    ObservationBatch,
    RelationObservation,
    SceneObservation,
    ViewType,
    WindowObservation,
)
from qwen_stream_video.state import EntityRegistry, EntityResolver, SceneTracker


def _observation(
    window_index: int,
    entities: list[EntityObservation],
    camera_change: bool = False,
    view_type: ViewType = ViewType.WIDE,
    relations: list[RelationObservation] | None = None,
) -> ObservationBatch:
    return ObservationBatch(
        window=WindowObservation(
            global_index=window_index,
            start_seconds=float(window_index * 3),
            commit_start_seconds=float(window_index * 3),
            end_seconds=float(window_index * 3 + 6),
        ),
        scene=SceneObservation(
            camera_change=camera_change,
            view_type=view_type,
            continuity_hint="camera_change" if camera_change else "continuous",
        ),
        entities=entities,
        relations=relations or [],
    )


def _person(local_id: str, **kwargs: object) -> EntityObservation:
    return EntityObservation(
        local_id=local_id,
        entity_type=EntityType.PERSON,
        name=kwargs.get("name", "worker"),
        confidence=kwargs.get("confidence", 0.9),
        spatial_region=kwargs.get("spatial_region", "center"),
        appearance=kwargs.get("appearance", {}),
        candidate_global_id=kwargs.get("candidate_global_id"),
        evidence_frames=kwargs.get("evidence_frames", []),
    )


def _device(local_id: str, **kwargs: object) -> EntityObservation:
    return EntityObservation(
        local_id=local_id,
        entity_type=EntityType.DEVICE,
        name=kwargs.get("name", "cabinet"),
        confidence=kwargs.get("confidence", 0.9),
        spatial_region=kwargs.get("spatial_region", "left"),
        appearance=kwargs.get("appearance", {}),
        candidate_global_id=kwargs.get("candidate_global_id"),
        evidence_frames=kwargs.get("evidence_frames", []),
    )


def _tool(local_id: str, **kwargs: object) -> EntityObservation:
    return EntityObservation(
        local_id=local_id,
        entity_type=EntityType.TOOL,
        name=kwargs.get("name", "wrench"),
        confidence=kwargs.get("confidence", 0.9),
        spatial_region=kwargs.get("spatial_region", "center"),
        appearance=kwargs.get("appearance", {}),
        candidate_global_id=kwargs.get("candidate_global_id"),
        evidence_frames=kwargs.get("evidence_frames", []),
    )


@pytest.fixture
def config() -> EntityRegistryConfig:
    return EntityRegistryConfig(
        confident_match_threshold=0.78,
        ambiguous_match_threshold=0.58,
        ambiguous_margin=0.08,
        max_missing_windows=10,
        temporary_entity_prefix="temp",
        candidate_hint_weight=0.05,
        allow_delayed_merge=True,
        delayed_merge_support_windows=2,
    )


@pytest.fixture
def registry(config: EntityRegistryConfig) -> EntityRegistry:
    return EntityRegistry(config)


@pytest.fixture
def resolver(registry: EntityRegistry) -> EntityResolver:
    return EntityResolver(registry)


@pytest.fixture
def scene_tracker() -> SceneTracker:
    return SceneTracker()


@pytest.fixture
def empty_state() -> GlobalState:
    return GlobalState(run_id="test")


def _seed_entity(
    registry: EntityRegistry,
    state: GlobalState,
    entity_type: EntityType,
    name: str,
    scene_id: str,
    appearance: dict[str, str] | None = None,
    spatial_region: str = "center",
    window_index: int = 0,
) -> str:
    """Create and seed a formal entity with an observation-like update."""
    entity = registry.create_entity(
        state, entity_type, name=name, window_index=window_index, scene_id=scene_id
    )
    registry.update_from_observation(
        state,
        entity.entity_id,
        EntityObservation(
            local_id="seed",
            entity_type=entity_type,
            name=name,
            confidence=0.95,
            appearance=appearance or {},
            spatial_region=spatial_region,
            evidence_frames=[],
        ),
        scene_id=scene_id,
        run_id=state.run_id,
        window_index=window_index,
    )
    return entity.entity_id


def test_entity_type_mismatch_is_excluded_by_find_candidates(
    resolver: EntityResolver, registry: EntityRegistry, scene_tracker: SceneTracker, empty_state: GlobalState
) -> None:
    """find_candidates filters by entity type, so a person cannot match a device."""
    scene_res = scene_tracker.update(empty_state, _observation(0, []))
    formal = registry.create_entity(empty_state, EntityType.DEVICE, window_index=0, scene_id=scene_res.scene_id)
    obs = _observation(1, [_person("P1", confidence=1.0)])
    scene_res = scene_tracker.update(empty_state, obs)
    batch = resolver.resolve(empty_state, registry, scene_res, obs, [])
    mapping = batch.mappings[0]
    assert mapping.status == EntityResolutionStatus.CREATED
    # The device was filtered out before scoring, so it cannot appear as a candidate.
    assert formal.entity_id not in mapping.candidate_scores


def test_two_visible_locals_cannot_map_same_global(
    resolver: EntityResolver, registry: EntityRegistry, scene_tracker: SceneTracker, empty_state: GlobalState
) -> None:
    scene_res = scene_tracker.update(empty_state, _observation(0, []))
    formal_id = _seed_entity(
        registry, empty_state, EntityType.PERSON, "worker", scene_res.scene_id, spatial_region="center"
    )
    obs = _observation(
        1,
        [
            _person("P1", name="worker", confidence=1.0, spatial_region="center"),
            _person("P2", name="worker", confidence=1.0, spatial_region="center"),
        ],
    )
    scene_res = scene_tracker.update(empty_state, obs)
    batch = resolver.resolve(empty_state, registry, scene_res, obs, [])
    ids = [m.global_entity_id for m in batch.mappings]
    # At most one local maps to the formal entity.
    assert sum(1 for i in ids if i == formal_id) <= 1


def test_candidate_hint_cannot_override_hard_constraint(
    resolver: EntityResolver, registry: EntityRegistry, scene_tracker: SceneTracker, empty_state: GlobalState
) -> None:
    scene_res = scene_tracker.update(empty_state, _observation(0, []))
    formal = registry.create_entity(empty_state, EntityType.DEVICE, window_index=0, scene_id=scene_res.scene_id)
    obs = _observation(1, [_person("P1", candidate_global_id=formal.entity_id, confidence=1.0)])
    scene_res = scene_tracker.update(empty_state, obs)
    batch = resolver.resolve(empty_state, registry, scene_res, obs, [])
    mapping = batch.mappings[0]
    assert mapping.status == EntityResolutionStatus.CREATED
    assert formal.entity_id in mapping.rejected_reasons[0] or any(formal.entity_id in r for r in mapping.rejected_reasons)


def test_candidate_hint_has_low_weight(
    resolver: EntityResolver, registry: EntityRegistry, scene_tracker: SceneTracker, empty_state: GlobalState
) -> None:
    scene_res = scene_tracker.update(empty_state, _observation(0, []))
    formal_id = _seed_entity(
        registry, empty_state, EntityType.PERSON, "worker", scene_res.scene_id, appearance={"hat": "red"}
    )
    # Very different appearance so that without hint it would not match confidently.
    obs = _observation(1, [_person("P1", name="other", candidate_global_id=formal_id, confidence=0.9)])
    scene_res = scene_tracker.update(empty_state, obs)
    batch = resolver.resolve(empty_state, registry, scene_res, obs, [])
    mapping = batch.mappings[0]
    assert mapping.status != EntityResolutionStatus.MATCHED


def test_confident_match_reuses_entity(
    resolver: EntityResolver, registry: EntityRegistry, scene_tracker: SceneTracker, empty_state: GlobalState
) -> None:
    scene_res = scene_tracker.update(empty_state, _observation(0, []))
    formal_id = _seed_entity(
        registry,
        empty_state,
        EntityType.PERSON,
        "worker",
        scene_res.scene_id,
        appearance={"hat": "red"},
        spatial_region="center",
    )
    obs = _observation(
        1,
        [_person("P1", name="worker", confidence=1.0, spatial_region="center", appearance={"hat": "red"})],
    )
    scene_res = scene_tracker.update(empty_state, obs)
    batch = resolver.resolve(empty_state, registry, scene_res, obs, [])
    mapping = batch.mappings[0]
    assert mapping.status == EntityResolutionStatus.MATCHED
    assert mapping.global_entity_id == formal_id
    assert any(e.event_type == "entity_matched" for e in batch.events)


def test_low_score_creates_new_entity(
    resolver: EntityResolver, registry: EntityRegistry, scene_tracker: SceneTracker, empty_state: GlobalState
) -> None:
    scene_res = scene_tracker.update(empty_state, _observation(0, []))
    _ = registry.create_entity(empty_state, EntityType.PERSON, name="worker", window_index=0, scene_id=scene_res.scene_id)
    # Different name and spatial region -> low score.
    obs = _observation(1, [_person("P1", name="stranger", spatial_region="far", confidence=0.9)])
    scene_res = scene_tracker.update(empty_state, obs)
    batch = resolver.resolve(empty_state, registry, scene_res, obs, [])
    mapping = batch.mappings[0]
    assert mapping.status == EntityResolutionStatus.CREATED
    assert mapping.global_entity_id.startswith("person_")
    assert any(e.event_type == "entity_created" for e in batch.events)


def test_ambiguous_threshold_creates_temporary(
    resolver: EntityResolver, registry: EntityRegistry, scene_tracker: SceneTracker, empty_state: GlobalState
) -> None:
    scene_res = scene_tracker.update(empty_state, _observation(0, []))
    registry.create_entity(empty_state, EntityType.PERSON, name="worker", window_index=0, scene_id=scene_res.scene_id)
    # Lower confidence produces lower score.
    obs = _observation(1, [_person("P1", name="worker", confidence=0.6, spatial_region="center")])
    scene_res = scene_tracker.update(empty_state, obs)
    batch = resolver.resolve(empty_state, registry, scene_res, obs, [])
    mapping = batch.mappings[0]
    assert mapping.status == EntityResolutionStatus.AMBIGUOUS
    assert mapping.global_entity_id.startswith("temp_")
    assert any(e.event_type == "entity_ambiguous" for e in batch.events)


def test_below_ambiguous_threshold_creates_formal(
    resolver: EntityResolver, registry: EntityRegistry, scene_tracker: SceneTracker, empty_state: GlobalState
) -> None:
    scene_res = scene_tracker.update(empty_state, _observation(0, []))
    _ = registry.create_entity(empty_state, EntityType.PERSON, name="worker", window_index=0, scene_id=scene_res.scene_id)
    obs = _observation(1, [_person("P1", name="stranger", confidence=0.3, spatial_region="far")])
    scene_res = scene_tracker.update(empty_state, obs)
    batch = resolver.resolve(empty_state, registry, scene_res, obs, [])
    mapping = batch.mappings[0]
    assert mapping.status == EntityResolutionStatus.CREATED
    assert not mapping.global_entity_id.startswith("temp_")


def test_margin_degrades_confident_to_ambiguous(
    resolver: EntityResolver, registry: EntityRegistry, scene_tracker: SceneTracker, empty_state: GlobalState
) -> None:
    scene_res = scene_tracker.update(empty_state, _observation(0, []))
    formal_a = registry.create_entity(
        empty_state, EntityType.PERSON, name="worker_a", window_index=0, scene_id=scene_res.scene_id
    )
    formal_b = registry.create_entity(
        empty_state, EntityType.PERSON, name="worker_b", window_index=0, scene_id=scene_res.scene_id
    )
    obs = _observation(
        1,
        [
            _person(
                "P1",
                name="worker",
                confidence=1.0,
                spatial_region="center",
                appearance={"hat": "red"},
            )
        ],
    )
    # Seed both formals with near-identical signatures.
    registry.update_from_observation(
        empty_state, formal_a.entity_id, _person("Px", name="worker_a", confidence=0.9), scene_id=scene_res.scene_id, run_id="test", window_index=0
    )
    registry.update_from_observation(
        empty_state, formal_b.entity_id, _person("Py", name="worker_b", confidence=0.9), scene_id=scene_res.scene_id, run_id="test", window_index=0
    )
    scene_res = scene_tracker.update(empty_state, obs)
    batch = resolver.resolve(empty_state, registry, scene_res, obs, [])
    mapping = batch.mappings[0]
    # Two very similar candidates should force ambiguous/temporary.
    assert mapping.status == EntityResolutionStatus.AMBIGUOUS


def test_rejected_hint_when_candidate_global_id_missing(
    resolver: EntityResolver, registry: EntityRegistry, scene_tracker: SceneTracker, empty_state: GlobalState
) -> None:
    scene_res = scene_tracker.update(empty_state, _observation(0, []))
    obs = _observation(1, [_person("P1", candidate_global_id="person_9999", confidence=0.9)])
    scene_res = scene_tracker.update(empty_state, obs)
    batch = resolver.resolve(empty_state, registry, scene_res, obs, [])
    mapping = batch.mappings[0]
    assert mapping.status == EntityResolutionStatus.REJECTED_HINT
    assert "person_9999" in mapping.rejected_reasons[0] or any("person_9999" in r for r in mapping.rejected_reasons)


def test_resolution_is_deterministic(
    resolver: EntityResolver, registry: EntityRegistry, scene_tracker: SceneTracker
) -> None:
    results: list[tuple[str, str]] = []
    for _ in range(3):
        empty_state = GlobalState(run_id="test")
        scene_res = scene_tracker.update(empty_state, _observation(0, []))
        _seed_entity(registry, empty_state, EntityType.PERSON, "worker", scene_res.scene_id)
        obs = _observation(1, [_person("P1", name="worker", confidence=1.0, appearance={"hat": "red"})])
        scene_res = scene_tracker.update(empty_state, obs)
        batch = resolver.resolve(empty_state, registry, scene_res, obs, [])
        results.append((batch.mappings[0].global_entity_id, batch.mappings[0].status.value))
    assert len({r[0] for r in results}) == 1
    assert len({r[1] for r in results}) == 1


def test_three_people_keep_ids_after_closeup_sequence(
    resolver: EntityResolver, registry: EntityRegistry, scene_tracker: SceneTracker
) -> None:
    state = GlobalState(run_id="test")

    # Window 0: wide view with three people in distinct regions.
    obs0 = _observation(
        0,
        [
            _person("A", spatial_region="left", appearance={"hat": "red"}),
            _person("B", spatial_region="center", appearance={"hat": "blue"}),
            _person("C", spatial_region="right", appearance={"hat": "green"}),
        ],
    )
    scene_res0 = scene_tracker.update(state, obs0)
    batch0 = resolver.resolve(state, registry, scene_res0, obs0, [])
    ids0 = {m.local_id: m.global_entity_id for m in batch0.mappings}
    assert len(set(ids0.values())) == 3

    # Window 1: closeup only hand from B.
    obs1 = _observation(1, [_person("B", spatial_region="center", appearance={"hat": "blue"})], view_type=ViewType.CLOSEUP)
    scene_res1 = scene_tracker.update(state, obs1)
    resolver.resolve(state, registry, scene_res1, obs1, [])

    # Window 2: wide again, A B C in same distinct regions.
    obs2 = _observation(
        2,
        [
            _person("A", spatial_region="left", appearance={"hat": "red"}),
            _person("B", spatial_region="center", appearance={"hat": "blue"}),
            _person("C", spatial_region="right", appearance={"hat": "green"}),
        ],
    )
    scene_res2 = scene_tracker.update(state, obs2)
    batch2 = resolver.resolve(state, registry, scene_res2, obs2, [])
    ids2 = {m.local_id: m.global_entity_id for m in batch2.mappings}

    # IDs should be stable: A, B, C map to themselves.
    assert ids2["A"] == ids0["A"]
    assert ids2["B"] == ids0["B"]
    assert ids2["C"] == ids0["C"]


def test_full_score_breakdown_saved(
    resolver: EntityResolver, registry: EntityRegistry, scene_tracker: SceneTracker, empty_state: GlobalState
) -> None:
    scene_res = scene_tracker.update(empty_state, _observation(0, []))
    formal_id = _seed_entity(
        registry, empty_state, EntityType.PERSON, "worker", scene_res.scene_id, appearance={"hat": "red"}
    )
    obs = _observation(1, [_person("P1", name="worker", confidence=1.0, appearance={"hat": "red"})])
    scene_res = scene_tracker.update(empty_state, obs)
    batch = resolver.resolve(empty_state, registry, scene_res, obs, [])
    mapping = batch.mappings[0]
    assert formal_id in mapping.candidate_scores
    breakdown = mapping.candidate_scores[formal_id]
    assert 0.0 <= breakdown.total_score <= 1.0
    assert breakdown.type_name_score is not None
    assert breakdown.appearance_score is not None
    assert breakdown.spatial_score is not None
    assert breakdown.relation_score is not None
    assert breakdown.recency_score is not None
    assert breakdown.candidate_hint_score is not None


def test_hard_constraint_rejected_reasons_saved(
    resolver: EntityResolver, registry: EntityRegistry, scene_tracker: SceneTracker, empty_state: GlobalState
) -> None:
    """High-confidence appearance conflict hard-rejects a candidate."""
    scene_res = scene_tracker.update(empty_state, _observation(0, []))
    formal_id = _seed_entity(
        registry,
        empty_state,
        EntityType.PERSON,
        "worker",
        scene_res.scene_id,
        appearance={"hat": "red"},
        spatial_region="center",
    )
    # Trigger a stable appearance conflict by first observing red, then blue at high confidence.
    registry.update_from_observation(
        empty_state,
        formal_id,
        _person("Px", name="worker", confidence=0.95, appearance={"hat": "blue"}, spatial_region="center"),
        scene_id=scene_res.scene_id,
        run_id="test",
        window_index=0,
    )
    obs = _observation(
        1,
        [_person("P1", name="worker", confidence=0.95, appearance={"hat": "red"}, spatial_region="center")],
    )
    scene_res = scene_tracker.update(empty_state, obs)
    batch = resolver.resolve(empty_state, registry, scene_res, obs, [])
    mapping = batch.mappings[0]
    assert formal_id in mapping.rejected_reasons[0] or any(formal_id in r for r in mapping.rejected_reasons)


def test_appearance_conflict_excludes_candidate(
    resolver: EntityResolver, registry: EntityRegistry, scene_tracker: SceneTracker, empty_state: GlobalState
) -> None:
    """A candidate with a recorded high-confidence appearance conflict is excluded."""
    scene_res = scene_tracker.update(empty_state, _observation(0, []))
    formal_id = _seed_entity(
        registry,
        empty_state,
        EntityType.PERSON,
        "worker",
        scene_res.scene_id,
        appearance={"uniform": "blue"},
        spatial_region="center",
    )
    # Record a conflicting high-confidence observation.
    registry.update_from_observation(
        empty_state,
        formal_id,
        _person("Px", name="worker", confidence=0.95, appearance={"uniform": "orange"}, spatial_region="center"),
        scene_id=scene_res.scene_id,
        run_id="test",
        window_index=0,
    )
    obs = _observation(
        1,
        [_person("P1", name="worker", confidence=0.95, appearance={"uniform": "blue"}, spatial_region="center")],
    )
    scene_res = scene_tracker.update(empty_state, obs)
    batch = resolver.resolve(empty_state, registry, scene_res, obs, [])
    mapping = batch.mappings[0]
    assert formal_id not in mapping.candidate_scores
    assert mapping.status == EntityResolutionStatus.CREATED


def test_impossible_spatial_jump_excludes_candidate(
    resolver: EntityResolver, registry: EntityRegistry, scene_tracker: SceneTracker, empty_state: GlobalState
) -> None:
    """In a continuous scene, a candidate cannot jump between regions in one window."""
    scene_res = scene_tracker.update(empty_state, _observation(0, []))
    formal_id = _seed_entity(
        registry,
        empty_state,
        EntityType.PERSON,
        "worker",
        scene_res.scene_id,
        spatial_region="left",
    )
    obs = _observation(
        1,
        [_person("P1", name="worker", confidence=0.95, spatial_region="right")],
    )
    scene_res = scene_tracker.update(empty_state, obs)
    batch = resolver.resolve(empty_state, registry, scene_res, obs, [])
    mapping = batch.mappings[0]
    assert formal_id not in mapping.candidate_scores
    assert mapping.status == EntityResolutionStatus.CREATED


def test_relation_consistency_boosts_relation_score(
    resolver: EntityResolver, registry: EntityRegistry, scene_tracker: SceneTracker, empty_state: GlobalState
) -> None:
    """A current relation matching the candidate's history increases relation_score."""
    scene_res = scene_tracker.update(empty_state, _observation(0, []))
    person_id = _seed_entity(registry, empty_state, EntityType.PERSON, "worker", scene_res.scene_id)
    tool_id = _seed_entity(registry, empty_state, EntityType.TOOL, "wrench", scene_res.scene_id)
    # Record historical relation between person and tool.
    registry.record_relations(
        empty_state,
        [RelationObservation(subject_local_id="P", relation_type="holding", object_local_id="T", confidence=0.9)],
        {"P": person_id, "T": tool_id},
        window_index=0,
    )
    obs = _observation(
        1,
        [
            _person("P1", name="worker", confidence=0.95),
            _tool("T1", name="wrench", confidence=0.95),
        ],
        relations=[RelationObservation(subject_local_id="P1", relation_type="holding", object_local_id="T1", confidence=0.9)],
    )
    scene_res = scene_tracker.update(empty_state, obs)
    batch = resolver.resolve(empty_state, registry, scene_res, obs, [])
    person_mapping = next(m for m in batch.mappings if m.local_id == "P1")
    tool_mapping = next(m for m in batch.mappings if m.local_id == "T1")
    assert person_mapping.candidate_scores[person_id].relation_score > 0.5
    assert tool_mapping.candidate_scores[tool_id].relation_score > 0.5


def test_relation_conflict_lowers_relation_score(
    resolver: EntityResolver, registry: EntityRegistry, scene_tracker: SceneTracker, empty_state: GlobalState
) -> None:
    """A current relation not present in the candidate's history yields a low relation_score."""
    scene_res = scene_tracker.update(empty_state, _observation(0, []))
    person_id = _seed_entity(registry, empty_state, EntityType.PERSON, "worker", scene_res.scene_id)
    tool_id = _seed_entity(registry, empty_state, EntityType.TOOL, "wrench", scene_res.scene_id)
    # No historical relation between person and tool.
    obs = _observation(
        1,
        [
            _person("P1", name="worker", confidence=0.95),
            _tool("T1", name="wrench", confidence=0.95),
        ],
        relations=[RelationObservation(subject_local_id="P1", relation_type="holding", object_local_id="T1", confidence=0.9)],
    )
    scene_res = scene_tracker.update(empty_state, obs)
    batch = resolver.resolve(empty_state, registry, scene_res, obs, [])
    person_mapping = next(m for m in batch.mappings if m.local_id == "P1")
    tool_mapping = next(m for m in batch.mappings if m.local_id == "T1")
    assert person_mapping.candidate_scores[person_id].relation_score == 0.0
    assert tool_mapping.candidate_scores[tool_id].relation_score == 0.0


def test_resolution_is_independent_of_dict_order(
    resolver: EntityResolver, registry: EntityRegistry, scene_tracker: SceneTracker
) -> None:
    """Resolution results must not depend on Python dict iteration order."""
    results: list[tuple[str, str]] = []
    for _ in range(3):
        empty_state = GlobalState(run_id="test")
        scene_res = scene_tracker.update(empty_state, _observation(0, []))
        _seed_entity(registry, empty_state, EntityType.PERSON, "worker", scene_res.scene_id, appearance={"hat": "red"})
        obs = _observation(
            1,
            [
                _person("A", name="worker", confidence=0.95, appearance={"hat": "red"}),
                _person("B", name="worker", confidence=0.95, appearance={"hat": "red"}),
            ],
        )
        scene_res = scene_tracker.update(empty_state, obs)
        batch = resolver.resolve(empty_state, registry, scene_res, obs, [])
        results.append(
            (
                tuple(sorted(m.global_entity_id for m in batch.mappings)),
                tuple(sorted(m.status.value for m in batch.mappings)),
            )
        )
    assert len(set(results)) == 1


def test_delayed_merge_support_windows(
    resolver: EntityResolver, registry: EntityRegistry, scene_tracker: SceneTracker, empty_state: GlobalState
) -> None:
    """A temporary entity is merged into the formal entity after enough consecutive support."""
    scene_res = scene_tracker.update(empty_state, _observation(0, []))
    formal_id = _seed_entity(
        registry,
        empty_state,
        EntityType.PERSON,
        "worker",
        scene_res.scene_id,
        appearance={"hat": "red"},
        spatial_region="center",
    )

    # Window 1: ambiguous match -> temporary entity created.
    obs1 = _observation(1, [_person("P1", name="worker", confidence=0.3)])
    scene_res = scene_tracker.update(empty_state, obs1)
    batch1 = resolver.resolve(empty_state, registry, scene_res, obs1, [])
    temp_id = batch1.mappings[0].global_entity_id
    assert batch1.mappings[0].status == EntityResolutionStatus.AMBIGUOUS

    # Window 2: confident match to formal accumulates delayed-merge support.
    obs2 = _observation(2, [_person("P1", name="worker", confidence=1.0, appearance={"hat": "red"})])
    scene_res = scene_tracker.update(empty_state, obs2)
    batch2 = resolver.resolve(empty_state, registry, scene_res, obs2, [])
    assert batch2.mappings[0].status == EntityResolutionStatus.MATCHED
    assert batch2.mappings[0].global_entity_id == formal_id
    assert not any(e.event_type == "entity_merged" for e in batch2.events)

    # Window 3: another confident match triggers delayed merge.
    obs3 = _observation(3, [_person("P1", name="worker", confidence=1.0, appearance={"hat": "red"})])
    scene_res = scene_tracker.update(empty_state, obs3)
    batch3 = resolver.resolve(empty_state, registry, scene_res, obs3, [])
    mapping3 = batch3.mappings[0]
    assert mapping3.global_entity_id == formal_id
    assert any(e.event_type == "entity_merged" for e in batch3.events)
    assert empty_state.entities[temp_id].merged_into == formal_id


def test_delayed_merge_is_disabled_by_config(
    resolver: EntityResolver, registry: EntityRegistry, scene_tracker: SceneTracker, empty_state: GlobalState
) -> None:
    """When allow_delayed_merge is false, temporary entities are never merged."""
    registry.config.allow_delayed_merge = False
    scene_res = scene_tracker.update(empty_state, _observation(0, []))
    formal_id = _seed_entity(
        registry,
        empty_state,
        EntityType.PERSON,
        "worker",
        scene_res.scene_id,
        appearance={"hat": "red"},
    )

    for window_index in (1, 2, 3):
        obs = _observation(
            window_index,
            [_person("P1", name="worker", confidence=0.3)],
        )
        scene_res = scene_tracker.update(empty_state, obs)
        batch = resolver.resolve(empty_state, registry, scene_res, obs, [])
        assert batch.mappings[0].global_entity_id != formal_id
        assert not any(e.event_type == "entity_merged" for e in batch.events)


def test_delayed_merge_support_is_reset_on_interruption(
    resolver: EntityResolver, registry: EntityRegistry, scene_tracker: SceneTracker, empty_state: GlobalState
) -> None:
    """A gap window resets the delayed-merge support counter."""
    scene_res = scene_tracker.update(empty_state, _observation(0, []))
    formal_id = _seed_entity(
        registry,
        empty_state,
        EntityType.PERSON,
        "worker",
        scene_res.scene_id,
        appearance={"hat": "red"},
    )

    # Window 1: ambiguous.
    obs1 = _observation(1, [_person("P1", name="worker", confidence=0.3)])
    scene_res = scene_tracker.update(empty_state, obs1)
    batch1 = resolver.resolve(empty_state, registry, scene_res, obs1, [])
    assert batch1.mappings[0].status == EntityResolutionStatus.AMBIGUOUS

    # Window 3 (skip window 2): support counter resets; no merge.
    obs3 = _observation(3, [_person("P1", name="worker", confidence=1.0, appearance={"hat": "red"})])
    scene_res = scene_tracker.update(empty_state, obs3)
    batch3 = resolver.resolve(empty_state, registry, scene_res, obs3, [])
    assert batch3.mappings[0].global_entity_id == formal_id
    assert not any(e.event_type == "entity_merged" for e in batch3.events)


def test_first_seen_time_comes_from_evidence(
    resolver: EntityResolver, registry: EntityRegistry, scene_tracker: SceneTracker, empty_state: GlobalState
) -> None:
    """A new entity created mid-run gets first_seen_time from its evidence timestamp."""
    scene_res = scene_tracker.update(empty_state, _observation(0, []))
    # Window 5 starts at 15s.
    obs = _observation(5, [_person("P1", name="worker", confidence=0.95, evidence_frames=[0])])
    # Fake sampled frames so evidence timestamp is 16s, not 0.
    from qwen_stream_video.video import SampledFrame

    sampled = [SampledFrame(run_index=5, global_index=5, sample_index=0, frame_index=0, timestamp_seconds=16.0)]
    scene_res = scene_tracker.update(empty_state, obs)
    batch = resolver.resolve(empty_state, registry, scene_res, obs, sampled)
    entity_id = batch.mappings[0].global_entity_id
    entity = empty_state.entities[entity_id]
    assert entity.first_seen_time == 16.0
    assert entity.last_seen_time == 16.0
