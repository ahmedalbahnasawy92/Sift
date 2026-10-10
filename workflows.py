"""
workflows.py: STEP 10, saved prompt templates ("workflows"), like Mike's.

A workflow is a folder with a SKILL.md file (the Agent Skills format that Agno
already parses and validates with agno.skills.LocalSkills):

    workflows/
      shared/                       <- for everyone (written by admins)
        gazette-appointments/SKILL.md
      users/<user_id>/              <- the user's own workflows
        my-weekly-report/SKILL.md

    SKILL.md:
      ---
      name: gazette-appointments          (lowercase, hyphens, = folder name)
      description: One line saying WHEN to use it (shown to the model)
      ---
      Step-by-step instructions the agent follows...

How the agent uses them
  - AVAILABLE WORKFLOWS (name + description) is added to the system prompt.
  - list_workflows / read_workflow tools load a workflow on demand, so long
    instructions only cost tokens when used (progressive disclosure).
  - The user can also run one explicitly: /wf <name> [extra text] in the CLI.

Why not Agent(skills=...) directly? Agno's Skills also exposes get_skill_script,
which can EXECUTE scripts, and it has no per-user folders. We reuse its loader
and validator, and expose only read-only tools.
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

from agno.run import RunContext
from agno.skills import LocalSkills, Skill

log = logging.getLogger("sift.workflows")

WORKFLOWS_DIR = Path(os.getenv("WORKFLOWS_DIR", "workflows"))
MAX_WORKFLOW_CHARS = 20_000


def _user_folder(user_id: Optional[str]) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", user_id or "anonymous").lstrip(".") or "anonymous"
    return WORKFLOWS_DIR / "users" / safe


def _load_folder(folder: Path) -> Dict[str, Skill]:
    if not folder.is_dir():
        return {}
    out = {}
    for item in sorted(folder.iterdir()):  # one bad workflow must not hide the others
        if item.is_dir() and (item / "SKILL.md").exists():
            try:
                for skill in LocalSkills(str(item), validate=True).load():
                    out[skill.name] = skill
            except Exception as e:
                log.warning("Skipping workflow %s: %s", item.name, e)
    return out


def load_workflows(user_id: Optional[str]) -> Dict[str, Tuple[Skill, str]]:
    """name -> (skill, scope). The user's own workflows override shared ones
    with the same name (only for that user)."""
    found = {n: (s, "shared") for n, s in _load_folder(WORKFLOWS_DIR / "shared").items()}
    found.update({n: (s, "mine") for n, s in _load_folder(_user_folder(user_id)).items()})
    return found


def workflows_block(user_id: Optional[str]) -> str:
    """AVAILABLE WORKFLOWS for the system prompt: names + descriptions only."""
    wfs = load_workflows(user_id)
    if not wfs:
        return ""
    lines = ["AVAILABLE WORKFLOWS (saved step-by-step procedures):"]
    for name, (skill, scope) in sorted(wfs.items()):
        lines.append(f"- {name} ({scope}): {' '.join(skill.description.split())[:300]}")
    lines.append("When the user's request matches a workflow, call read_workflow(name) "
                 "FIRST and follow its steps. Don't follow a workflow the user didn't ask for "
                 "if it doesn't fit the request.")
    return "\n".join(lines)


def get_workflow(name: str, user_id: Optional[str]) -> Optional[Tuple[Skill, str]]:
    return load_workflows(user_id).get(name.strip().lower())


def format_workflow(skill: Skill, scope: str) -> str:
    body = skill.instructions.strip()[:MAX_WORKFLOW_CHARS]
    return (f"Workflow '{skill.name}' ({scope}): {skill.description}\n\n"
            f"--- instructions ---\n{body}\n--- end of workflow ---\n"
            "Follow these steps using your tools. All earlier rules still apply "
            "(grounding, citations, approvals, reply language).")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
def list_workflows(run_context: RunContext) -> str:
    """List the saved workflows (step-by-step procedures) available to the
    user, with a description of when to use each one.
    """
    wfs = load_workflows(run_context.user_id)
    if not wfs:
        return "No workflows are available."
    return "\n".join(f"- {n} ({scope}): {s.description}" for n, (s, scope) in sorted(wfs.items()))


def read_workflow(name: str, run_context: RunContext) -> str:
    """Load the full instructions of a saved workflow, then follow them.
    Call this when the user's request matches a workflow in AVAILABLE WORKFLOWS,
    or when the user asks to run a workflow by name.

    Args:
        name: The workflow name, e.g. "gazette-appointments".
    """
    found = get_workflow(name, run_context.user_id)
    if not found:
        names = ", ".join(sorted(load_workflows(run_context.user_id))) or "none"
        return f"No workflow named '{name}'. Available: {names}."
    return format_workflow(*found)


WORKFLOW_TOOLS = [list_workflows, read_workflow]
