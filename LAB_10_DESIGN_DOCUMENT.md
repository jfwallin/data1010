# Lab 10: AI Self-Assessment and the Hallucination Boundary
## Complete Design Document

**Course:** DATA 1010 – Artificial Intelligence in Action
**Lab Number:** 10
**Total Time:** 60-75 minutes
**Format:** Hybrid (Jupyter notebooks + web-based LLM testing)

---

## Core Pedagogical Message

**AI models are overconfident and make mistakes. Users need caution, testing, and domain knowledge to identify when AI systems fail.**

Supporting lessons:
- AI cannot reliably predict its own failures
- Confidence in tone ≠ accuracy
- Systematic testing is essential
- Human oversight matters

---

## Lab Structure Overview

### Module 0: Setup and Understanding AI Self-Assessment (5-8 min, Prelab)
- Enter group code → generates 8 deterministic prompts
- Read conceptual explanation of AI self-assessment
- Make initial predictions about AI's self-awareness
- **Questions:** Q1-Q3

### Module 1: Collecting AI Self-Predictions (15-20 min, In-class)
- Test each of 8 prompts on real AI (ChatGPT, Claude, Gemini, etc.)
- Record AI's confidence level via dropdown
- Record student's prediction of accuracy
- Add optional notes/quotes
- **Questions:** Q4-Q7

### Module 2: Evaluating AI Responses (15-20 min, In-class)
- Verify AI's actual accuracy (Google, Wikipedia, domain knowledge)
- Record actual performance via dropdown
- Identify error types via multi-select
- Document specific errors
- **Questions:** Q8-Q12

### Module 3: Analysis and Visualization (15-20 min, In-class)
- Merge all collected data
- Generate 3 key visualizations (heatmap, bar chart, table)
- Calculate overconfidence rate
- Compare across groups
- **Questions:** Q13-Q18

### Module 4: Synthesis and Implications (10-12 min, In-class)
- Review key findings
- Read 4 real-world case studies
- Group discussion on implications
- Review best practices framework
- **Questions:** Q19-Q23

---

## Prompt Generation Strategy

### 8 Categories (1 prompt per category, selected deterministically)

Each group code generates 8 prompts by selecting 1 from each category pool (10 prompts per pool):

1. **Factual Recall** - Verifiable facts (population, geography, historical dates)
2. **Reasoning Chain** - Multi-step logic problems
3. **Citation Request** - Ask for sources (high hallucination risk)
4. **Ambiguous Query** - Underspecified questions
5. **Recent Events** - Training cutoff boundary tests
6. **Mathematical** - Verifiable calculations
7. **Commonsense** - Baseline reliability check
8. **Edge Case** - Trick questions

### Balanced Difficulty Distribution
- **Easy (baseline):** Commonsense, Mathematical (25%)
- **Medium:** Factual, Recent Events, Edge Case (37.5%)
- **Hard:** Reasoning Chain, Citation Request, Ambiguous (37.5%)

### Model-Agnostic Design
- All prompts work with any LLM
- No model-specific references
- No API calls required
- Students use free web interfaces

---

## Data Collection Method

### Module 1: Prediction Phase (~2 min per prompt)

For each of 8 prompts:

**Dropdown 1 - AI Confidence Level:**
1. No caveats - answered confidently
2. Mild caveats (e.g., "This might...", "Generally...")
3. Strong caveats (e.g., "I may be wrong", "Please verify")
4. Refused or heavily qualified the answer

**Dropdown 2 - Student's Prediction:**
1. Will be accurate
2. Might have minor errors
3. Likely to have major errors
4. Completely failed/refused

**Text Area:** Optional notes/quotes

### Module 2: Evaluation Phase (~2 min per prompt)

For each of 8 prompts:

**Dropdown - Actual Accuracy:**
1. Fully accurate - no errors found
2. Mostly accurate - minor errors or imprecision
3. Partially accurate - significant errors or omissions
4. Inaccurate - major errors or hallucinations
5. Refused or unable to answer

**Multi-Select - Error Types:**
1. No errors
2. Factual error (wrong information)
3. Hallucinated citation/source
4. Logic error in reasoning
5. Outdated information
6. Overgeneralization
7. Incomplete answer
8. Misunderstood question

**Text Area:** Error details

### Data Flow
```
Module 0 → prompts.csv (8 rows: prompt_id, category, prompt_text)
Module 1 → predictions.csv (8 rows: prompt_id, ai_confidence, student_prediction, notes)
Module 2 → evaluations.csv (8 rows: prompt_id, actual_accuracy, error_types, error_details)
Module 3 → complete_analysis.csv (merged dataset)
```

---

## Key Visualizations (Module 3)

### 1. Confusion Matrix Heatmap
- **Rows:** AI confidence level (Confident / Cautious / Refused)
- **Columns:** Actual accuracy (Accurate / Inaccurate / Refused)
- **Cells:** Count of prompts in each combination
- **Key Metric:** Overconfidence Rate = (Confident & Inaccurate) / (Total Confident) × 100%

### 2. Error Rate Bar Chart
- **X-axis:** 8 prompt categories
- **Y-axis:** Error rate (0-100%)
- **Sorted:** By error rate (descending)
- **Purpose:** Show systematic weaknesses

### 3. Overconfidence Examples Table
- **Filter:** Only show "Confident + Inaccurate" cases
- **Columns:** Prompt ID, Category, Prompt Text, AI Confidence, Actual Accuracy, Errors Found
- **Purpose:** Concrete evidence of overconfidence

---

## Complete Question List (23 total)

### Module 0: Setup and Understanding (3 questions)
- **Q1:** Can AI models accurately predict when they will make mistakes? Why or why not?
- **Q2:** What's the difference between an AI being confident and an AI being correct?
- **Q3:** If an AI says "I might be wrong about this," does that mean it's more likely to be wrong? Make a prediction.

### Module 1: Collecting Predictions (4 questions)
- **Q4:** Looking at Prompt #1, did the AI express any uncertainty or caveats? Quote specific phrases.
- **Q5:** For which prompt(s) did the AI refuse to answer or express strong uncertainty?
- **Q6:** Did the AI use similar language for all prompts, or did confidence levels vary? Give examples.
- **Q7:** Predict: For which prompts do you think the AI's self-assessment will be accurate? Which ones might be overconfident?

### Module 2: Evaluating Responses (5 questions)
- **Q8:** For Prompt #1, was the AI's response actually accurate? How did you verify this?
- **Q9:** Identify one prompt where the AI was confident but made errors. What went wrong?
- **Q10:** Identify one prompt where the AI expressed uncertainty. Was it actually accurate or inaccurate?
- **Q11:** Did you find any "hallucinated" information (specific false details presented as fact)? Give an example.
- **Q12:** Which category of prompts (factual, reasoning, citation, etc.) led to the most errors?

### Module 3: Analysis and Visualization (6 questions)
- **Q13:** Looking at the confusion matrix, how often did the AI's self-assessment match its actual performance?
- **Q14:** Did the AI show overconfidence (confident tone but inaccurate) or underconfidence (cautious but accurate)? Which was more common?
- **Q15:** Which prompt category had the highest error rate? Why do you think this category is difficult for AI?
- **Q16:** Looking at your group's most "overconfident" prompt, what made the AI fail despite sounding sure?
- **Q17:** Did expressing uncertainty (caveats) correlate with lower accuracy? Use your data to support your answer.
- **Q18:** Compare your results with another group. Did different groups find similar patterns of overconfidence?

### Module 4: Synthesis and Implications (5 questions)
- **Q19:** Based on your findings, complete this statement: "AI models are most likely to fail when..."
- **Q20:** What strategies should you use to verify AI-generated information before trusting it? List at least 3.
- **Q21:** Should AI systems always express uncertainty when they might be wrong? What are the trade-offs?
- **Q22:** How does understanding AI's hallucination boundary change how you will use ChatGPT, Claude, or similar tools in the future?
- **Q23:** (Synthesis) Explain to a friend who hasn't taken this class: Why can't AI reliably predict its own mistakes? Use concepts from this lab.

---

## Technical Implementation Details

### Required Python Libraries
```python
import numpy as np               # Prompt generation, random seeding
import pandas as pd              # Data collection and analysis
import matplotlib.pyplot as plt  # Visualizations
import seaborn as sns           # Heatmap styling
import ipywidgets as widgets    # Interactive dropdowns, buttons
from IPython.display import display, clear_output, HTML, Markdown
```

### Key Functions

**Prompt Generation:**
```python
def generate_group_prompts(group_code, num_prompts=8):
    """Generate deterministic prompts for a group."""
    np.random.seed(group_code)

    # 8 categories, 10 prompts each
    pools = {
        'factual_recall': [...],      # 10 factual questions
        'reasoning_chain': [...],      # 10 logic problems
        'citation_request': [...],     # 10 source requests
        'ambiguous_query': [...],      # 10 ambiguous questions
        'recent_events': [...],        # 10 temporal boundary tests
        'mathematical': [...],         # 10 calculations
        'commonsense': [...],          # 10 baseline questions
        'edge_case': [...]            # 10 trick questions
    }

    prompts = []
    for i, (category, pool) in enumerate(pools.items()):
        idx = np.random.randint(0, len(pool))
        prompts.append({
            'prompt_id': i + 1,
            'category': category,
            'prompt_text': pool[idx]
        })

    return prompts
```

### Widget Patterns

**Dropdown Widget:**
```python
confidence_dropdown = widgets.Dropdown(
    options=['No caveats - answered confidently',
             'Mild caveats',
             'Strong caveats',
             'Refused or heavily qualified'],
    description='AI Confidence:',
    style={'description_width': 'initial'},
    layout={'width': '600px'}
)
```

**Multi-Select Widget:**
```python
error_types = widgets.SelectMultiple(
    options=['No errors', 'Factual error', 'Hallucinated citation',
             'Logic error', 'Outdated information', 'Overgeneralization',
             'Incomplete answer', 'Misunderstood question'],
    description='Error Types:',
    layout={'width': '650px', 'height': '150px'}
)
```

### Data Persistence
- Module 0: Save prompts → `lab10_group_{code}_prompts.csv`
- Module 1: Save predictions → `lab10_group_{code}_predictions.csv`
- Module 2: Save evaluations → `lab10_group_{code}_evaluations.csv`
- Module 3: Merge and save → `lab10_group_{code}_complete.csv`

---

## Files to Create

### Directory Structure
```
lab_10_modules/
├── lab_10_module_0_setup.ipynb
├── lab_10_module_1_collect_predictions.ipynb
├── lab_10_module_2_evaluate_responses.ipynb
├── lab_10_module_3_analysis.ipynb
├── lab_10_module_4_synthesis.ipynb
├── Lab_10_Student_Handout.md
└── Lab_10_Answer_Sheet.md
```

### File Specifications

**Jupyter Notebooks (5 files):**
1. Module 0: Setup (5-8 min) - Prompt generation, intro concepts
2. Module 1: Collect predictions (15-20 min) - Widget-based data entry
3. Module 2: Evaluate responses (15-20 min) - Accuracy assessment
4. Module 3: Analysis (15-20 min) - Visualizations and patterns
5. Module 4: Synthesis (10-12 min) - Case studies and best practices

**Documentation (2 files):**
1. Student Handout (~20-25 KB) - Complete pedagogical guide with all 23 questions
2. Answer Sheet (~7-9 KB) - Question scaffolding for submissions

---

## Connection to Previous Labs

**Lab 1-2:** Error measurement and optimization
- **Lab 10:** Explores unmeasurable error (AI's blind spots)

**Lab 4:** Neural network training and evaluation
- **Lab 10:** Evaluating a system that cannot evaluate itself

**Lab 5:** Embeddings and semantic meaning
- **Lab 10:** Limits of semantic understanding

**Lab 6:** Saliency and explainability
- **Lab 10:** When explanations are unreliable

**Lab 8:** Generative models
- **Lab 10:** Generative models generating unreliable self-assessments

---

## Case Studies for Module 4

### 1. Legal Brief Hallucinations (2023)
- Lawyer sanctioned for submitting brief with fabricated citations
- ChatGPT invented case names, citations, and quotes
- Cost: Professional sanctions, client harm, public embarrassment

### 2. Medical Misinformation
- AI confidently providing wrong drug dosages
- Hallucinated contraindications and side effects
- Risk: Patient safety, medical malpractice

### 3. Academic Paper Errors
- Students submitting papers with non-existent references
- AI fabricating study results and statistics
- Impact: Academic integrity violations

### 4. News Article Fabrication
- False quotes attributed to real people
- Invented statistics presented as fact
- Consequence: Misinformation spread, credibility damage

---

## Best Practices Framework (Module 4)

### When AI is Generally Reliable
- Brainstorming ideas
- Drafting text (with editing)
- Explaining concepts
- Summarizing provided documents
- Code suggestions (with testing)

### When AI Requires Extra Caution
- Specific facts and statistics
- Citations and sources
- Recent events (beyond training cutoff)
- High-stakes decisions (legal, medical, financial)
- Numerical calculations
- Obscure or specialized knowledge

### Verification Strategies
1. **Cross-reference multiple sources** (Google, Wikipedia, primary sources)
2. **Check citations directly** (never trust AI-provided sources without verification)
3. **Domain expert review** (for specialized topics)
4. **Test numerical results** (independent calculation)
5. **Look for red flags** (overly specific details, unfamiliar citations)
6. **Understand AI's training cutoff** (knowledge boundary)

---

## Implementation Priority Order

1. **Module 0** - Prompt generation (foundation for everything)
2. **Student Handout** - Instructional guide (defines pedagogy)
3. **Module 1** - Data collection UI (core student activity)
4. **Module 2** - Evaluation UI (verification phase)
5. **Module 3** - Visualizations (pedagogical centerpiece)
6. **Module 4** - Synthesis (ties concepts together)
7. **Answer Sheet** - Assessment document (last)

---

## Critical Success Factors

1. **Simplicity** - No code execution by students, dropdown-based interface
2. **Time efficiency** - 2 min per prompt, fits in 60-75 min total
3. **Model-agnostic** - Works with any LLM, future-proof as models evolve
4. **Evidence-based** - Data visualization shows patterns clearly
5. **Actionable** - Clear takeaways for responsible AI use
6. **Engaging** - Students test real AI systems, see real failures

---

## Expected Student Outcomes

By the end of Lab 10, students will:

1. **Understand** that AI confidence ≠ accuracy
2. **Recognize** patterns of AI overconfidence in their own data
3. **Identify** systematic weaknesses (e.g., citation hallucinations)
4. **Apply** verification strategies before trusting AI
5. **Synthesize** findings into principles for responsible AI use
6. **Evaluate** trade-offs in AI design (helpfulness vs. honesty)

---

## Pedagogical Strengths

✓ **Predict → Experiment → Explain** pattern maintained throughout
✓ **Evidence-based learning** through data collection and visualization
✓ **Cross-group discussion** enabled by varied prompts
✓ **Real-world relevance** through case studies
✓ **Actionable takeaways** for responsible AI use
✓ **Future-proof design** as models evolve
✓ **Accessible** - no API keys, free tools only

---

## Testing and Validation

**Before finalizing the lab:**
1. Pilot test all 80 prompts (10 per category) on current frontier models
2. Verify that prompts produce varied behaviors (confident, cautious, accurate, overconfident)
3. Confirm that error rates vary across categories as expected
4. Test that timing fits within 60-75 minutes
5. Validate that visualizations display correctly with sample data

**If models are too well-calibrated:**
- Adjust prompt difficulty
- Add more challenging edge cases
- Include prompts designed to test specific failure modes
- Reframe as "boundary exploration" rather than "failure detection"

---

## AI Use Policy for This Lab

**Appropriate AI Use:**
- Verify facts using AI + other sources
- Compare responses across different AI models
- Use AI to explain concepts in Module 0

**Inappropriate AI Use:**
- Asking AI to complete lab questions
- Using AI to fabricate experimental results
- Having AI write answers without actual testing

**The Irony:**
This lab teaches students NOT to blindly trust AI by having them systematically test AI reliability. Using AI to complete the lab would undermine the entire learning objective.

---

## Document Status

**Version:** 1.0
**Last Updated:** 2025-12-11
**Status:** Complete - Ready for Implementation
**Next Step:** Create lab_10_modules/ directory and begin building Module 0

---

END OF DESIGN DOCUMENT
