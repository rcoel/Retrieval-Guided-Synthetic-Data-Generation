# Implementation Plan - Project Differentiation Strategy

## Goal Description
Transform the current "Adaptive RAG" from a heuristic-based loop (temperature tuning) into a robust **Agentic Privacy Framework**. This aims to solve the "Utility-Privacy Tradeoff" more effectively than standard open-source tools by using active LLM critique and adversarial testing.

## User Review Required
> [!IMPORTANT]
> **Architectural Change**: This moves the generation logic from a simple loop to a multi-step Agentic workflow (Generator -> Critic -> Refiner). This will increase generation time per sample but significantly improve quality and privacy guarantees.

## Proposed Changes

### Core Logic Upgrade: Agentic Self-Correction
Replace the numeric feedback loop (Temperature +/-) with semantic feedback.
#### [MODIFY] [generation.py](file:///Users/veerasagar/Retrieval-Guided-Synthetic-Data-Generation/src/pipeline/generation.py)
- **Current**: `if fail: temp += 0.2`
- **New**: 
    1. **Generator**: Produce draft.
    2. **Critic**: If `fail`, generate strict instructions (e.g., "You leaked the date. Remove it.").
    3. **Refiner**: Re-generate using `Original Prompt + Critic Feedback`.

### New Capability: Adversarial "Red Team" Filter
Most RAG tools check *overlap*. We will add a check for *information leakage*.
#### [NEW] [red_team.py](file:///Users/veerasagar/Retrieval-Guided-Synthetic-Data-Generation/src/evaluation/red_team.py)
- Implement a `PrivacyAttacker` class.
- **Logic**: Prompt an LLM to try and guess the private entity or label from the synthetic text.
- Integrated into `generation.py` as a final gate.

### Enhancement: Noisy Retrieval
Add a layer of privacy at the retrieval step.
#### [MODIFY] [indexing.py](file:///Users/veerasagar/Retrieval-Guided-Synthetic-Data-Generation/src/pipeline/indexing.py)
- Add a `privacy_epsilon` parameter.
- Inject Gaussian noise into query embeddings before searching the FAISS index to prevent exact pattern matching of private data against the public corpus.

## Verification Plan

### Automated Tests
- **Privacy Attack Simulation**: Run `red_team.py` on known leaks to verify it catches them.
- **Feedback Loop Test**: Mock a failure case and ensure the `Critic` generates prompts that actually lead to a fixed output.

### Manual Verification
- **Before/After Comparison**: Generate 10 samples with the old logic vs. new Agentic logic.
- **Review Artifact**: Create a `comparison.md` showing how Agentic Refinement fixed specific privacy leaks that simple temperature tuning missed.
