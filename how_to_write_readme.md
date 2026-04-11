# How to Write and Update README.md

## Purpose
README.md should be just code maps help LLMs (and developers) quickly understand codebase structure, navigate to the right files, and follow established patterns. They should be **concise, high-level, and focused on navigation**.

## Structure

**Two-Tier Approach:**
1. **Main Code Map** (`README.md`) - High-level overview of all components
2. **Component Code Maps** (e.g., `src/server/README.md`) - Detailed documentation

**Template:**
```markdown
## /path/to/component
Description: One-sentence summary

### Main Entry Points
- **Name**: `file.py` or `directory/` - Brief description

### Purpose
1. **First responsibility** - What it does
2. **Second responsibility** - What it does

### Architecture Pattern
Brief description of design patterns, data flow, or key decisions.
```

## When to Update

Update when:
1. **New component added** - Add section to main README.md
2. **Component structure changes** - Update Main Entry Points
3. **Architecture changes** - Update Architecture Pattern
4. **Component relationships change** - Update Purpose

## Guidelines

**Be Concise:**
- ✓ "FastAPI backend that processes user interactions"
- ✗ "This is a FastAPI backend server that receives user interactions from various clients..."

**Focus on Navigation:**
- Include file paths and directory names
- Match headings to directory structure
- Reference detailed documentation when it exists

**Show Relationships:**
```
ComponentA
   -> ComponentB
      -> ComponentC
   -> ComponentD
```

**Highlight Rules:**
- **NEVER import storage implementations directly** - Always use BaseStorage
- **ALWAYS use LiteLLMClient** - Not OpenAIClient or ClaudeClient

## Detailed Component Maps

For complex components, create `README.md` in that directory with:
- Main entry points and when to modify them
- Sub-components with key files
- Architecture patterns and data flow

## Update Process

1. **Check git changes** - Run `git diff` or `git status` to see what changed since last commit
2. **Identify changes** - New files? Interface changes? Pattern changes?
3. **Read existing map** - Understand current structure
4. **Update sections** - Start with component map, then main map
5. **Verify** - Check file paths and relationships
6. **Test navigation** - Can an LLM find the right file to modify?

## Repository Patterns

**Abstract Base Classes:**
```markdown
**Service Pattern**: Load configs → Create actors → Run in parallel → Save results
```

**Never Import Implementations:**
```markdown
**NEVER import storage implementations directly** - Always use `request_context.storage`
```

**Context Passing:**
```markdown
- **`api_endpoints/`**: `RequestContext` (bundles storage/config/prompts)
```

## Best Practices

- Update README as part of feature development
- README is code map for navigation, not API docs
- Use **bold** for important rules/patterns
- Test: Can an LLM find the right file for common tasks?

## Checklist

- [ ] One-sentence component description
- [ ] File paths for main entry points
- [ ] Purpose explains component relationships
- [ ] Architecture Pattern explains key decisions
- [ ] Follow consistent syntax and highlight in the file
- [ ] No unnecessary details or duplication

## Key Questions

1. **Can an LLM find the right file?** - Clear file paths?
2. **Can an LLM understand the pattern?** - Design patterns documented?
3. **Can an LLM avoid mistakes?** - Anti-patterns highlighted (NEVER/ALWAYS)?
4. **Is it concise?** - No unnecessary elaboration?
5. **Is it complete?** - All major components and relationships covered?

## Example: Adding New File

**Scenario:** New `evolvement_playbook_extractor.py` added to `src/server/services/feedback/`

**Steps:**
1. Check changes: `git status` shows new file in `services/feedback/`
2. Review changes: `git diff --cached` or `git diff HEAD` to see new code
3. Identify scope: Component-specific (update `src/server/README.md`)
4. Read current docs: `cat src/server/README.md | grep -A 20 "Playbook"`
5. Add to Playbook Extraction section with other extractors
6. Check main map: No update needed (already covered at high level)
7. Verify: Correct path, follows pattern, no duplication

## Good vs Bad

**Good README (code map):**
- Concise, high-level overview
- Help find the right file quickly
- Explain how to make changes correctly
- Hierarchical (main + detailed component maps)

**Bad README (code map):**
- Too verbose with unnecessary details
- Inconsistent syntax or formatting
- Missing file paths
- Duplicating information
- Out of sync with actual code
