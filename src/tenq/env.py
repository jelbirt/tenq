"""Which of the run's secrets each child process may see.

The Actions job holds every secret in its environment. Child processes get
only what they need: the model CLI keeps its own token and nothing else;
xerj needs none of them.
"""

from __future__ import annotations

import os

# the run's secrets other than the model's own credentials
PRIVATE_ENV = ("SMTP_", "LANGFUSE_", "RECIPIENT_EMAILS", "EDGAR_IDENTITY", "OPENROUTER_")
MODEL_ENV = ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY")


def scrubbed(drop: tuple[str, ...]) -> dict[str, str]:
    """os.environ without any variable whose name starts with one of `drop`."""
    return {k: v for k, v in os.environ.items() if not k.startswith(drop)}
