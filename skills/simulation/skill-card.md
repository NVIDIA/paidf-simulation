## Description: <br>
Build Docker-backed PAIDF Simulation commands for PCBA synthetic-data renders (good/defect/missing/lighting/ChangeNet pairs) and ROI crops with optional MI registration. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers generating synthetic image data for PCB automated optical inspection (AOI) training, including good/defect/missing/lighting renders, paired ChangeNet datasets, and per-component ROI crops with optional mutual-information registration. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [SKILL.md](SKILL.md) <br>
- [Single-Flow Stages](references/single-flow/stages.md) <br>
- [Single-Flow Routing](references/single-flow/routing.md) <br>
- [Single-Flow Overrides](references/single-flow/overrides.md) <br>
- [Single-Flow Local Mode](references/single-flow/local-mode.md) <br>
- [Single-Flow Troubleshooting](references/single-flow/troubleshooting.md) <br>
- [ROI Stages](references/roi/stages.md) <br>
- [ROI Day-0](references/roi/day0.md) <br>
- [ROI Day-1](references/roi/day1.md) <br>
- [ROI Semantic Rules](references/roi/semantic-rules.md) <br>
- [ROI Troubleshooting](references/roi/troubleshooting.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- claude-code <br>
- codex <br>



## Evaluation Tasks: <br>
Evaluated against 15 evaluation tasks (15 positive skill-activation cases). 2 attempts per task, 50% pass threshold. NVSkills-Eval profile: external. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access. <br>
- Correctness: Checks whether the agent follows the expected workflow and produces the correct final output. <br>
- Discoverability: Checks whether the agent loads the skill when relevant and avoids using it when irrelevant. <br>
- Effectiveness: Checks whether the agent performs measurably better with the skill than without it. <br>
- Efficiency: Checks whether the agent uses fewer tokens and avoids redundant work. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Verifies that the agent loaded the expected skill and workflow. <br>
- `skill_efficiency`: Checks routing quality, decoy avoidance, and redundant tool usage. <br>
- `accuracy`: Grades final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Checks whether the overall user task completed successfully. <br>
- `behavior_check`: Verifies expected behavior steps, including safety expectations. <br>
- `token_efficiency`: Compares token usage with and without the skill. <br>



## Evaluation Results: <br>
| Dimension | Num | `claude-code` | `codex` |
|---|---:|---:|---:|
| Security | 8 | 100% (+3%) | 97% (+0%) |
| Correctness | 8 | 83% (-0%) | 69% (+3%) |
| Discoverability | 8 | 92% (+6%) | 73% (-2%) |
| Effectiveness | 8 | 57% (-3%) | 50% (+7%) |
| Efficiency | 8 | 80% (+10%) | 62% (-0%) |

## Skill Version(s): <br>
1.0.0 (source: frontmatter, pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
