# Skill: Paper Reviewer

A top-tier AI conference reviewer that critically evaluates papers and provides detailed feedback.

## When to use
Use this skill when you need to:
- Get a critical review of a paper for conference submission
- Identify weaknesses and areas for improvement
- Prepare a paper for submission to NeurIPS, ICML, ICLR, ACL, AAAI, EMNLP
- Compare a paper against top-tier conference standards

## How to invoke
Invoke the `ai-conference-reviewer` agent with a task to review the paper in:
- Paper chapters: `E:\Projects\ImperfectStudent\paper_chapters/`
- Figures: `E:\Projects\ImperfectStudent\figures/`

## Expected output
A detailed review report covering:
1. Summary
2. Strengths
3. Weaknesses and concerns (with severity ratings)
4. Detailed comments by section
5. Questions for authors
6. Recommendation
7. Scores (novelty, technical quality, clarity, significance, reproducibility)

## Invocation

```
Use the Task tool with subagent_type="general" and prompt:
"Run the paper reviewer agent to review the controllable skill forgetting paper.
Read all files in E:\Projects\ImperfectStudent\paper_chapters/ and
examine figures in E:\Projects\ImperfectStudent\figures/.
Provide a detailed critical review as specified in the agent instructions."
```
