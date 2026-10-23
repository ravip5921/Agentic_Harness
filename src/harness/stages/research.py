
"""Research stage: inspect context, gather evidence, and produce a spec."""

import json
import re
import os

from collections.abc import Callable
from pathlib import Path

from textwrap import dedent

from harness.models import (
    EvidenceSource,
    ResearchAssessment,
    ResearchResult,
    StageOutcome,
)
from harness.events import emit_event
from harness.state import HarnessState

from harness.tools.repository import (
    RepositoryContext,
    inspect_repository,
    RepositoryError,
)
from harness.tools.evidence import (
    retrieve_evidence,
    EvidenceError,
)
from harness.agents.research_agent import (
    GroqResearchAgent,
    ResearchGenerationError,
)


def render_spec(spec) -> str:
    """Render a validated engineering specification as Markdown."""

    def bullets(items):
        return "\n".join(f"- {item}" for item in items) or "- None"

    return dedent(f"""# {spec.title}

        ## Objective

        {spec.objective}

        ## Requirements

        {bullets(spec.requirements)}

        ## Repository Context

        {spec.repository_context or "No repository context provided."}

        ## Technical Approach

        {spec.technical_approach}

        ## Implementation Steps

        {bullets(spec.implementation_steps)}

        ## Acceptance Criteria

        {bullets(spec.acceptance_criteria)}

        ## Assumptions

        {bullets(spec.assumptions)}

        ## Constraints

        {bullets(spec.constraints)}

        ## Evidence References

        {bullets(spec.source_ids)}
    """)


def validate_evidence_references(
    assessment: ResearchAssessment,
    evidence: list[EvidenceSource],
) -> None:
    """Reject references to sources that were not retrieved."""

    if assessment.specification is None:
        return

    valid_ids = {source.source_id for source in evidence}
    referenced_ids = set(assessment.specification.source_ids)

    unknown = referenced_ids - valid_ids

    if unknown:
        raise ValueError(
            f"Specification references unknown sources: {sorted(unknown)}"
        )


def write_research_artifacts(
    run_id: str,
    assessment: ResearchAssessment,
    evidence: list[EvidenceSource],
    runs_dir: Path,
) -> list[str]:
    """Persist a validated specification and evidence."""

    if assessment.specification is None:
        raise ValueError("Cannot write a missing specification.")

    # Run identifiers must not be usable as filesystem paths.
    if not re.fullmatch(r"[a-f0-9]{32}", run_id):
        raise ValueError("Invalid run ID.")

    output_dir = runs_dir.resolve() / run_id / "research"
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = assessment.specification

    markdown_path = output_dir / "engineering_spec.md"
    json_path = output_dir / "engineering_spec.json"
    evidence_path = output_dir / "evidence.json"

    markdown_path.write_text(
        render_spec(spec),
        encoding="utf-8",
    )

    json_path.write_text(
        spec.model_dump_json(indent=2),
        encoding="utf-8",
    )

    evidence_path.write_text(
        json.dumps(
            [source.model_dump(mode="json") for source in evidence],
            indent=2,
        ),
        encoding="utf-8",
    )

    return [
        str(markdown_path),
        str(json_path),
        str(evidence_path),
    ]


def make_research_stage(
    generator: Callable,
    repository_reader: Callable = inspect_repository,
    evidence_reader: Callable = retrieve_evidence,
    runs_dir: Path = Path("runs"),
):
    """Create a Research node with injected dependencies.

    generator must accept (task, repository, evidence)
    and return ResearchAssessment.
    """

    def research(state: HarnessState) -> dict:
        task = state["task"]

        try:
            if task.repository_path is not None:
                emit_event(
                    "research", "repository_inspection_started",
                    "Inspecting repository context", level="start",
                )
            repository = repository_reader(
                task.repository_path
            )
            if task.repository_path is not None:
                if repository is None:
                    emit_event(
                        "research", "repository_inspection_completed",
                        "No repository context available",
                    )
                elif isinstance(repository, RepositoryContext):
                    emit_event(
                        "research", "repository_inspection_completed",
                        f"Repository inspected: {len(repository.files)} files",
                        data={"file_count": len(repository.files)},
                        level="success",
                    )
                else:
                    emit_event(
                        "research", "repository_inspection_completed",
                        "Repository inspection completed",
                    )

            # Documentation URLs are supplied as task data.
            urls = task.parameters.get(
                "documentation_urls", []
            )

            if not isinstance(urls, list) or not all(
                isinstance(url, str) for url in urls
            ):
                raise ValueError(
                    "documentation_urls must be a list of strings."
                )

            emit_event(
                "research", "evidence_retrieval_started",
                "Retrieving supplied documentation",
                data={"supplied_source_count": len(urls)}, level="start",
            )
            evidence = evidence_reader(
                urls,
            )
            if isinstance(evidence, list):
                emit_event(
                    "research", "evidence_retrieval_completed",
                    f"Documentation retrieved: {len(evidence)} sources",
                    data={"source_count": len(evidence)}, level="success",
                )
            else:
                emit_event(
                    "research", "evidence_retrieval_completed",
                    "Documentation retrieval completed",
                )

            emit_event(
                "research", "specification_generation_started",
                "Generating engineering specification", level="start",
            )
            assessment = generator(
                task,
                repository,
                evidence,
            )

            if assessment.status == "blocked":
                emit_event(
                    "research", "specification_generation_blocked",
                    "Essential information is unavailable for the specification",
                    level="warning",
                )
                result = ResearchResult(
                    outcome=StageOutcome.BLOCKED,
                    summary=assessment.reason,
                    errors=[assessment.reason],
                )

            else:
                validate_evidence_references(
                    assessment,
                    evidence,
                )

                artifacts = write_research_artifacts(
                    task.run_id,
                    assessment,
                    evidence,
                    runs_dir,
                )
                specification = assessment.specification
                emit_event(
                    "research", "specification_generated",
                    "Specification generated: "
                    f"{len(specification.requirements)} requirements, "
                    f"{len(specification.acceptance_criteria)} acceptance criteria",
                    data={
                        "requirement_count": len(specification.requirements),
                        "acceptance_criteria_count": len(
                            specification.acceptance_criteria
                        ),
                    },
                    level="success",
                )

                result = ResearchResult(
                    outcome=StageOutcome.COMPLETED,
                    summary="Engineering specification generated.",
                    artifacts=artifacts,
                )

        except (
            FileNotFoundError,
            RepositoryError,
            EvidenceError,
            ValueError,
        ) as exc:
            emit_event(
                "research", "research_blocked",
                "Research prerequisites could not be satisfied",
                level="warning",
            )
            result = ResearchResult(
                outcome=StageOutcome.BLOCKED,
                summary="Research prerequisites could not be satisfied.",
                errors=[str(exc)],
            )

        except (
            ResearchGenerationError,
            OSError,
        ) as exc:
            emit_event(
                "research", "research_failed",
                "Research generation or artifact persistence failed",
                level="failure",
            )
            result = ResearchResult(
                outcome=StageOutcome.FAILED,
                summary="Research execution failed.",
                errors=[str(exc)],
            )

        return {"research_result": result}

    return research


def research(state: HarnessState) -> dict:
    """Default LangGraph entry point for Research."""

    agent = GroqResearchAgent()

    node = make_research_stage(
        generator=agent.generate
    )

    return node(state)
