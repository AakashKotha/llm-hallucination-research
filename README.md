# Measuring Hallucination Rate in LLMs Across Domains and Model Versions

A systematic study of hallucination patterns in large language models across different knowledge domains and model versions, providing evidence-based insights for responsible AI deployment.

## Research Overview

This study analyzes 1,200 responses from three state-of-the-art models (GPT-3.5, GPT-4, Claude 3.5 Sonnet) across 400 fact-based questions spanning five knowledge domains to understand when, where, and why hallucinations occur in LLM outputs.

## Key Findings

**Domain-Specific Variation**: Hallucination rates vary dramatically by knowledge domain, ranging from 2.5% in Science to 30.4% in Pop Culture. Abstract and culturally-specific domains demonstrate significantly higher error rates than concrete, verifiable domains.

**Model Evolution**: Newer models show substantial improvements in factual accuracy, with GPT-4 achieving a 37% reduction in hallucination rate compared to GPT-3.5. These improvements are consistent across different measurement approaches but vary by question type.

**Question Complexity**: Contrary to expectations, question length shows no significant correlation with hallucination likelihood, indicating that content domain rather than surface complexity drives error patterns.

## Methodology

The research employs a comprehensive evaluation framework including manual annotation of responses for factual correctness, statistical analysis across multiple measurement specifications, and validation through cross-model comparison. Questions were sourced from established datasets including TriviaQA, SciQ, Natural Questions, HotpotQA, and MedMCQA.

## Repository Contents

The repository includes the complete dataset with questions, model responses, and annotations, along with analysis code and detailed methodology documentation. The full research paper provides comprehensive statistical analysis and practical implementation guidance.

## Practical Applications

Organizations can use these findings to implement domain-specific risk assessment frameworks, make informed model selection decisions, and develop content-based validation strategies for LLM deployment. The research provides evidence-based guidance for balancing accuracy requirements with operational efficiency.

## Citation

```
Emberi, N. B., & Kotha, V. V. S. A. (2025). Measuring Hallucination Rate in LLMs Across Domains and Model Versions.
```
