# Agent OS Configuration

This directory contains the Agent OS configuration and workflow files for the SplatThis project.

## Directory Structure

- `/product/` - Product documentation and strategic planning
  - `mission.md` - Core product mission and vision
  - `mission-lite.md` - Condensed mission statement
  - `tech-stack.md` - Technical architecture decisions
  - `roadmap.md` - Product development roadmap
  - `decisions.md` - Product decision log

- `/specs/` - Feature specifications and requirements
  - Each spec gets its own dated folder (e.g., `2025-01-20-feature-name/`)
  - Standard spec structure includes:
    - `spec.md` - Main specification document
    - `spec-lite.md` - Summary version
    - `tasks.md` - Implementation tasks
    - `sub-specs/` - Technical specifications, API specs, database schemas

- `/workflows/` - Development workflow configurations
  - Custom Agent OS workflow definitions
  - CI/CD pipeline configurations
  - Development process templates

## Usage

### Creating a New Spec

1. Create a new dated folder in `/specs/`: `2025-01-20-feature-name/`
2. Use the standard spec templates provided by Agent OS
3. Break down implementation into actionable tasks
4. Reference related technical specifications in sub-specs

### Product Planning

1. Start with `/product/mission.md` to define core product vision
2. Use `/product/roadmap.md` for release planning
3. Document major decisions in `/product/decisions.md`
4. Keep tech stack current in `/product/tech-stack.md`

## Agent OS Integration

This configuration follows Agent OS conventions for:
- Structured specification development
- Task-based implementation tracking
- Product decision documentation
- Technical architecture planning