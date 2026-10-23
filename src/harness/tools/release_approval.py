"""Human release decisions. Missing input never authorizes publication."""

from pathlib import Path

from harness.models import ReleaseDecision, TaskRequest


def prompt_release_approval(
    *, task: TaskRequest, candidate: Path, report_path: Path,
) -> ReleaseDecision:
    """Ask for approval and then a destination, defaulting to local staging."""
    print(f"\nRelease review for run {task.run_id}", flush=True)
    print(f"Candidate: {candidate}\nReport: {report_path}", flush=True)
    try:
        if input("Approve this release candidate? [y/N]: ").strip().lower() not in {"y", "yes"}:
            return ReleaseDecision.REJECT
        if task.repository_path is None:
            print("No target repository supplied; keeping the approved candidate staged.")
            return ReleaseDecision.KEEP_STAGED
        while True:
            choice = input(
                "Keep staged or push a branch and create a draft GitHub PR? [staged/pr]: "
            ).strip().lower()
            if choice in {"", "staged"}:
                return ReleaseDecision.KEEP_STAGED
            if choice == "pr":
                return ReleaseDecision.CREATE_PR
            print("Enter 'staged' or 'pr'.")
    except (EOFError, KeyboardInterrupt, OSError):
        return ReleaseDecision.REJECT
