"""Phase 8A: the adaptive routing layer.

    Question
       -> Cheap Hybrid RRF baseline (frozen Phase 4.1 pipeline, unmodified)
       -> Router
            SIMPLE  -> keep Hybrid results
            MEDIUM  -> escalate to Hybrid + Reranker (frozen Phase 5 pipeline)
            COMPLEX -> escalate to bounded Agentic retrieval (frozen Phase 7,
                       4500-token budget from Phase 7.1)

This package builds and evaluates ONLY the routing decision. It does not
run answer generation, and it does not run the full Hybrid+Reranker or
Agentic pipelines for real (Phase 8A is a routing-level cost PROJECTION,
not the final end-to-end result — see `mhrag.routing.cost_projection`).

Two clearly separated worlds, enforced structurally (see each module's
docstring and tests/test_routing_no_gold_leakage.py):

  - EVALUATOR-ONLY (`mhrag.routing.oracle`, `mhrag.routing.tune_thresholds`,
    `mhrag.routing.metrics`, `mhrag.routing.split`): touches gold evidence,
    gold document ids, and oracle route labels — used only to calibrate
    thresholds and to score router performance, never imported by the
    runtime router.
  - RUNTIME (`mhrag.routing.features`, `mhrag.routing.heuristic`,
    `mhrag.routing.glm_router`, `mhrag.routing.router`): sees only the
    question text and retrieval results the frozen Hybrid pipeline
    actually returns for it — no function in this half of the package has
    a parameter through which a gold answer, evidence_list, question_type,
    or oracle label could reach it.
"""
