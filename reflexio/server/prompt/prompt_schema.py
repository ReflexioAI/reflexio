from pydantic import BaseModel


class Prompt(BaseModel):
    """Self-contained prompt loaded from a .prompt.md file with YAML frontmatter."""

    active: bool = False
    description: str | None = None
    changelog: str | None = None
    variables: list[str]
    content: str
