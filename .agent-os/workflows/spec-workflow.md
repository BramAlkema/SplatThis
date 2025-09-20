# Spec Development Workflow

This document outlines the standard workflow for developing and implementing specifications in the SplatThis project.

## Workflow Stages

### 1. Spec Creation
- Create new spec folder: `.agent-os/specs/YYYY-MM-DD-spec-name/`
- Draft `spec.md` with comprehensive requirements
- Create `spec-lite.md` for quick reference
- Break down into tasks in `tasks.md`

### 2. Technical Planning
- Create `sub-specs/technical-spec.md` for implementation details
- Add `sub-specs/api-spec.md` if API changes required
- Add `sub-specs/database-schema.md` if database changes required
- Add `sub-specs/tests.md` for testing requirements

### 3. Implementation
- Work through tasks in priority order
- Update task status as work progresses
- Reference spec documents for clarity and requirements
- Create implementation branches following naming convention

### 4. Review & Documentation
- Ensure all spec requirements are met
- Update technical documentation as needed
- Mark spec as completed
- Archive or update for future iterations

## File Templates

All spec files should follow the standard Agent OS templates:
- Use proper metadata headers
- Include creation dates and version numbers
- Cross-reference related documents using @ notation
- Maintain consistent formatting and structure

## Naming Conventions

- Spec folders: `YYYY-MM-DD-descriptive-name`
- Branch names: `spec/YYYY-MM-DD-descriptive-name`
- Task references: Use spec folder name for tracking

## Integration Points

- Link specs to product roadmap phases
- Reference product decisions that influence specs
- Update tech stack documentation for new dependencies
- Coordinate with overall product mission and goals