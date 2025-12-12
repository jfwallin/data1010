# Lab 10: AI Self-Assessment and the Hallucination Boundary

**Course:** DATA 1010 – Artificial Intelligence in Action

---

## Overview

### What You'll Learn Today

In today's lab, you will explore whether AI models can predict their own failures—a critical question for responsible AI use. You'll learn to:

- Understand the difference between AI confidence and actual accuracy
- Test whether AI can assess its own reliability before answering questions
- Identify patterns of AI overconfidence through systematic experimentation
- Recognize when AI hallucin

ates (fabricates information confidently)
- Develop verification strategies for responsible AI use
- Connect findings to real-world consequences of AI failures

### Lab Structure

This lab consists of **5 modules** that you'll complete in order:

| Module | Title | Time | Type |
|--------|-------|------|------|
| 0 | Setup and Understanding AI Self-Assessment | ~5-8 min | Prelab |
| 1 | Collecting AI Self-Predictions | ~15-20 min | In-class |
| 2 | Evaluating AI Responses | ~15-20 min | In-class |
| 3 | Analysis and Visualization | ~15-20 min | In-class |
| 4 | Synthesis and Implications | ~10-12 min | In-class |

**Total Time:** ~60-75 minutes

### Working in Groups

- Work in **small groups** (2-4 people)
- One group member runs Colab and shares their screen
- Everyone participates in discussion and predictions
- All group members can use their own notebooks

### Key Concepts

> **HALLUCINATION**: When an AI generates false information presented confidently as fact. Examples include inventing citations, fabricating statistics, or creating plausible-sounding but incorrect facts.
>
> **OVERCONFIDENCE**: When an AI expresses high confidence (no caveats, definitive tone) but provides inaccurate information. This is the gap between how sure the AI sounds and how correct it actually is.
>
> **CALIBRATION**: How well an AI's expressed confidence matches its actual accuracy. A well-calibrated AI expresses uncertainty when likely to be wrong and confidence when likely to be correct.
>
> **SELF-ASSESSMENT**: An AI's ability to predict its own performance on a task before attempting it. This lab investigates whether AI systems have reliable self-assessment capabilities.
>
> **VERIFICATION**: The process of checking AI-generated information against independent sources (search engines, primary sources, expert knowledge) before trusting it.

### Connection to Previous Labs

**Lab 1:** You learned about models, error measurement, and optimization basics
- **Lab 10:** You'll explore errors that models cannot reliably predict—their own blind spots

**Lab 2:** You learned about gradient descent for automatic parameter updates
- **Lab 10:** Training improves accuracy, but does it improve self-awareness?

**Lab 4:** You learned that neural networks can be evaluated on test data
- **Lab 10:** But can neural networks evaluate themselves?

**Lab 5:** You learned that embeddings encode meaning as geometry
- **Lab 10:** What are the limits of semantic understanding?

**Lab 6:** Saliency maps showed WHERE a model looks
- **Lab 10:** But can models tell WHEN they're looking at the wrong things?

**Lab 8:** You learned how diffusion models generate images
- **Lab 10:** Generative models can generate confident-sounding but false text too

---

## Module 0: Setup and Understanding AI Self-Assessment

**Time:** ~5-8 minutes
**Type:** Prelab

### Learning Objectives
- Understand what "self-assessment" means for AI models
- Learn the difference between confidence (tone) and accuracy (correctness)
- Generate group-specific prompts for testing
- Establish baseline expectations about AI reliability

### What You'll Do

1. Enter your group code to generate 8 unique prompts
2. Review the 8 prompt categories (Factual, Reasoning, Citation, etc.)
3. Read conceptual explanation of AI overconfidence
4. Make initial predictions about AI's self-awareness
5. Save prompts for use in later modules

### Key Insight

> **AI models sound confident even when wrong.** The tone of a response (confident, cautious, uncertain) does not reliably predict accuracy. This lab will help you see this pattern through data you collect yourself.

### Questions

**Q1.** Before testing, do you think AI models can accurately predict when they will make mistakes? Why or why not?

**Answer:**
<br><br><br>

**Q2.** What's the difference between an AI being confident (how it sounds) and being correct (actually right)?

**Answer:**
<br><br><br>

**Q3.** If an AI says "I might be wrong about this," does that mean it's more likely to be wrong? Make a prediction you can test later.

**Answer:**
<br><br><br>

---

## Module 1: Collecting AI Self-Predictions

**Time:** ~15-20 minutes
**Type:** In-class

### Learning Objectives
- Test prompts on real AI model(s) (ChatGPT, Claude, Gemini, etc.)
- Systematically record AI's confidence level for each response
- Observe how AI expresses uncertainty (or doesn't)
- Make predictions about which responses will be accurate

### What You'll Do

1. Load your 8 group-specific prompts from Module 0
2. For each prompt:
   - **Copy** the prompt to an AI web interface (ChatGPT, Claude, Gemini, etc.)
   - **Open a fresh/private window** to avoid context contamination
   - **Read** the AI's complete response carefully
   - **Record** the AI's confidence level using a dropdown menu
   - **Predict** whether you think the response is accurate
   - **Add notes** about specific phrases the AI used
3. Save all predictions for analysis in Module 2

### Important Testing Guidelines

**Use a fresh context for each prompt:**
- Open a new private/incognito window for each of the 8 prompts
- This ensures the AI doesn't use previous responses as context
- Close the window after recording the response

**What to look for in responses:**
- Does the AI use caveats like "I might be wrong" or "Please verify"?
- Does the AI refuse to answer or express strong uncertainty?
- Does the AI sound definitive and confident?
- Are there phrases like "definitely," "certainly," "I'm confident that..."?

**Time budget:** ~2 minutes per prompt (16 minutes total for 8 prompts)

### Key Insight

> **Pay close attention to HOW the AI expresses uncertainty.** Some phrases signal genuine awareness of limits ("I don't have access to real-time data"), while others are hedges that don't match the confident tone of the rest of the response ("This might vary slightly").

### Questions

**Q4.** Looking at Prompt #1 (whichever prompt your group got), did the AI express any uncertainty or caveats? Quote specific phrases from the response.

**Prompt #1 topic:** ____________________________

**Answer:**
<br><br><br>

**Q5.** For which prompt(s) did the AI refuse to answer or express strong uncertainty? List the prompt ID numbers and categories.

**Answer:**
<br><br><br>

**Q6.** Did the AI use similar language for all prompts, or did confidence levels vary across different categories? Give specific examples.

**Answer:**
<br><br><br>

**Q7.** PREDICTION: Looking at the 8 responses you collected, for which prompts do you think the AI's self-assessment will be accurate? Which ones do you suspect might show overconfidence (confident tone but actually wrong)?

**Will match (confidence = accuracy):**
<br><br>

**Might be overconfident:**
<br><br><br>

---

## Module 2: Evaluating AI Responses

**Time:** ~15-20 minutes
**Type:** In-class

### Learning Objectives
- Verify AI's actual accuracy through independent research
- Identify specific errors, hallucinations, and logical flaws
- Classify error types systematically
- Compare actual performance to AI's expressed confidence

### What You'll Do

1. Load predictions from Module 1
2. For each of the 8 prompts:
   - **Verify** the AI's response using Google, Wikipedia, or domain knowledge
   - **Record** the actual accuracy level using a dropdown
   - **Identify** error types using a multi-select menu (if errors exist)
   - **Document** specific errors or hallucinations you find
   - **Note** whether the AI's confidence matched its accuracy
3. Save evaluations for visualization in Module 3

### Verification Strategies

**For factual claims:**
- Search Google for authoritative sources
- Check Wikipedia for basic facts (but verify with other sources)
- Look for government or academic sources

**For citations:**
- Search for the exact paper/book title
- Check if authors exist and work in that field
- Verify journal names and publication dates
- **Red flag:** If you can't find it, it might be fabricated

**For calculations:**
- Use a calculator or spreadsheet
- Double-check mathematical reasoning

**For logic problems:**
- Work through the problem yourself
- Check if the AI's reasoning makes sense step-by-step

**Time budget:** ~2 minutes per prompt (16 minutes total for 8 prompts)

### Key Insight

> **Hallucinations are often highly specific.** An AI might invent a complete, plausible-sounding citation with real author names, a real journal, but a paper that doesn't exist. The specificity makes it seem credible—until you verify it.

### Questions

**Q8.** For Prompt #1, was the AI's response actually accurate? How did you verify this? (What sources did you use?)

**Actual accuracy:** ______________________

**Verification method:**
<br><br><br>

**Q9.** Identify ONE prompt where the AI was confident but made errors. What prompt was it? What went wrong?

**Prompt ID:** _____ **Category:** ____________________

**What the AI said (summary):**
<br><br>

**What was actually wrong:**
<br><br><br>

**Q10.** Identify ONE prompt where the AI expressed uncertainty (caveats or refusal). Was the AI actually accurate or inaccurate on that prompt?

**Prompt ID:** _____ **Category:** ____________________

**AI's uncertainty level:** ____________________

**Actual accuracy:** ____________________

**Analysis:**
<br><br><br>

**Q11.** Did you find any "hallucinated" information—specific false details the AI presented as fact? Give a concrete example.

**Answer:**
<br><br><br>

**Q12.** Which category of prompts (Factual Recall, Reasoning Chain, Citation Request, Ambiguous Query, Recent Events, Mathematical, Commonsense, Edge Case) led to the most errors in your group's data?

**Category with most errors:** ____________________

**Why do you think this category is difficult for AI?**
<br><br><br>

---

## Module 3: Analysis and Visualization

**Time:** ~15-20 minutes
**Type:** In-class

### Learning Objectives
- Merge all collected data into a complete dataset
- Visualize the relationship between AI confidence and actual accuracy
- Identify patterns of overconfidence across prompt categories
- Calculate key metrics (overconfidence rate, accuracy by category)
- Compare findings across different groups

### What You'll Do

1. Load and merge data from Modules 0, 1, and 2
2. Generate three key visualizations:
   - **Confusion Matrix Heatmap:** AI confidence vs. actual accuracy
   - **Error Rate Bar Chart:** Error percentage by prompt category
   - **Overconfidence Examples Table:** Specific cases where AI was confident but wrong
3. Calculate summary statistics (overconfidence rate, total accuracy, etc.)
4. Participate in instructor-led cross-group discussion
5. Answer analysis questions based on visualizations

### The Three Visualizations Explained

**1. Confusion Matrix Heatmap**

This heatmap shows how often the AI's expressed confidence matched its actual performance.

- **Rows:** AI's confidence level (Confident / Cautious / Refused)
- **Columns:** Actual accuracy (Accurate / Inaccurate / Refused)
- **Cell values:** Number of prompts in each combination
- **Key metric:** Overconfidence Rate = (Confident & Inaccurate) / (Total Confident) × 100%

**What to look for:**
- Are most prompts on the diagonal (confidence matches accuracy)?
- Are there many in the "Confident but Inaccurate" cell? (overconfidence)
- Are there any in the "Cautious but Accurate" cell? (underconfidence)

**2. Error Rate Bar Chart**

This shows which prompt categories had the highest error rates.

- **X-axis:** The 8 prompt categories
- **Y-axis:** Percentage of prompts in that category with errors
- **Sorted:** From highest error rate to lowest

**What to look for:**
- Which categories are hardest for AI? (tallest bars)
- Which are easiest? (shortest bars)
- Does this match your intuition from Module 1?

**3. Overconfidence Examples Table**

This table shows specific prompts where the AI was confident but wrong.

- **Columns:** Prompt ID, Category, Prompt Text, AI Confidence, Actual Accuracy, Errors Found
- **Filter:** Only shows cases where AI was confident AND inaccurate

**What to look for:**
- What kinds of prompts trigger overconfident failures?
- Are the errors subtle or obvious?
- Did you notice the overconfidence while testing, or only after verification?

### Key Insight

> **Visualizing data reveals patterns you might miss in individual cases.** Looking at all 8 prompts together shows whether overconfidence is rare or systematic, and which types of questions consistently cause problems.

### Questions

**Q13.** Looking at the confusion matrix heatmap, how often did the AI's self-assessment (expressed confidence) match its actual performance?

**Matches (on diagonal):** _____ out of _____ prompts = _____%

**Interpretation:**
<br><br><br>

**Q14.** Did the AI show **overconfidence** (confident tone but inaccurate) or **underconfidence** (cautious but accurate)? Which pattern was more common in your data?

**Overconfident prompts:** _____

**Underconfident prompts:** _____

**More common pattern:** ____________________

**Analysis:**
<br><br><br>

**Q15.** Which prompt category had the highest error rate in the bar chart? Why do you think this category is difficult for AI models?

**Category with highest error rate:** ____________________

**Error rate:** _____%

**Why this is difficult:**
<br><br><br>

**Q16.** Looking at the overconfidence examples table, pick your group's most "overconfident" prompt (confident but very wrong). What made the AI fail despite sounding sure of itself?

**Prompt ID:** _____ **Category:** ____________________

**Why it failed:**
<br><br><br>

**Q17.** Did expressing uncertainty (caveats like "I might be wrong") correlate with lower accuracy? Use your data to support your answer.

**Data analysis:**
<br><br>

**Conclusion:**
<br><br><br>

**Q18.** (Cross-group comparison) Compare your confusion matrix or overconfidence rate with another group's results. Did different groups find similar patterns of overconfidence?

**Your overconfidence rate:** _____%

**Other group's rate:** _____%

**Similar patterns?**
<br><br><br>

---

## Module 4: Synthesis and Implications

**Time:** ~10-12 minutes
**Type:** In-class

### Learning Objectives
- Synthesize findings into actionable principles for AI use
- Connect lab findings to real-world consequences of AI failures
- Understand when human oversight is essential
- Develop verification strategies for responsible AI interaction
- Evaluate trade-offs in AI system design

### What You'll Do

1. Review key findings from your group's Module 3 analysis
2. Read 4 case studies of real-world AI failures
3. Participate in group discussion about implications
4. Review best practices framework for AI use
5. Complete synthesis questions that connect concepts to practice

### Real-World Case Studies

#### Case Study 1: Legal Brief Hallucinations (2023)

**What happened:**
- A lawyer used ChatGPT to research case law for a federal court filing
- ChatGPT provided several relevant-seeming citations with case names, dates, and quotes
- The lawyer included these citations in the brief without verification
- Opposing counsel noticed the cases didn't exist
- Investigation revealed ChatGPT had invented all the cases, complete with plausible judicial opinions

**Consequences:**
- Lawyer sanctioned by federal judge
- $5,000 fine
- Reputation damage
- Client harm (weakened case)
- National news coverage highlighting AI risks

**Lesson:** Even when AI provides highly specific, detailed information (case names, quotes, citations), it can be completely fabricated. **Always verify citations independently.**

#### Case Study 2: Medical Misinformation

**What happened:**
- Healthcare workers and patients increasingly use AI chatbots for medical information
- Studies found AI models confidently providing incorrect drug dosages
- AI hallucinated contraindications (warnings about drug interactions) that don't exist
- AI also failed to mention real contraindications, creating safety risks

**Consequences:**
- Potential patient harm if incorrect dosages followed
- Erosion of trust in AI-assisted healthcare
- Regulatory scrutiny of AI in medical settings
- Calls for mandatory human expert review

**Lesson:** In high-stakes domains (medical, legal, financial), AI overconfidence can have life-or-death consequences. **Domain expertise and verification are non-negotiable.**

#### Case Study 3: Academic Paper Fabrication

**What happened:**
- Students submitted papers with AI-generated citations
- AI invented author names, paper titles, journals, and publication years
- Some fabricated sources were highly specific and seemed credible
- Professors detected inconsistencies when trying to access the cited papers
- Cross-referencing revealed systematic hallucination

**Consequences:**
- Academic integrity violations
- Failed assignments or course failures
- Institutional policies restricting AI use
- Damage to student credibility

**Lesson:** AI can fabricate academic sources with convincing detail. **Check every citation directly—search for the exact title, verify authors, confirm the journal exists.**

#### Case Study 4: News and Social Media Misinformation

**What happened:**
- AI systems used to generate news summaries or social media content
- False quotes attributed to real people
- Invented statistics presented as fact
- Real-sounding but fictional events described
- Misinformation spread rapidly before corrections

**Consequences:**
- Public confusion about facts
- Damage to people falsely quoted
- Erosion of information trust
- Platform policies restricting AI-generated content

**Lesson:** The ease of generating plausible-sounding false content at scale makes verification critical. **If a claim seems surprising or too specific, verify it independently.**

### Best Practices Framework

#### When AI is Generally Reliable
AI tends to perform well on:
- Brainstorming and ideation
- Drafting text that you'll edit
- Explaining well-established concepts
- Summarizing documents you provide
- Code suggestions (with testing!)
- Language translation (for common languages)
- General knowledge questions with well-known answers

**Still verify claims, but error rate is lower.**

#### When AI Requires Extra Caution
AI is more likely to fail on:
- Specific facts, statistics, and dates
- Citations and academic sources
- Recent events (beyond training cutoff)
- Numerical calculations
- Obscure or specialized knowledge
- High-stakes decisions (medical, legal, financial)
- Questions requiring access to real-time data
- Requests that combine multiple constraints

**Always verify before trusting.**

#### Verification Strategies

1. **Cross-reference multiple independent sources**
   - Don't trust a single source (including AI)
   - Look for government, academic, or journalistic sources
   - Check if multiple reliable sources agree

2. **Verify citations directly**
   - Search for exact paper/book titles
   - Check that authors exist and work in that field
   - Confirm journal or publisher is real
   - Access the source if possible

3. **Seek domain expert review**
   - For specialized topics, consult someone with expertise
   - Don't rely on AI for critical medical, legal, or technical advice
   - Understand the limits of your own knowledge

4. **Test numerical results independently**
   - Re-calculate mathematical claims
   - Use a calculator or spreadsheet
   - Check units and reasonableness

5. **Look for red flags**
   - Overly specific details that seem too convenient
   - Citations you can't find with simple searches
   - Claims that contradict what you know
   - Responses that ignore ambiguity in your question

6. **Understand knowledge boundaries**
   - Check the model's training cutoff date
   - Know that AI doesn't have real-time access to data
   - Recognize that AI can't verify its own claims

### Design Trade-offs

**Should AI always express uncertainty when it might be wrong?**

**Arguments for always expressing uncertainty:**
- Prevents overconfidence
- Encourages user verification
- Makes limitations transparent
- Reduces harm from errors

**Arguments against always expressing uncertainty:**
- Reduces usefulness (constant caveats frustrate users)
- Users might ignore warnings ("cry wolf" effect)
- Doesn't solve the underlying problem (errors still happen)
- Shifts responsibility to users who may lack expertise

**The tension:**
AI systems face a trade-off between being helpful (direct answers) and being honest (acknowledging limitations). Current systems tend toward helpfulness, which can lead to overconfidence.

### Key Insight

> **The core problem isn't that AI makes mistakes—it's that AI cannot reliably predict when it will make mistakes.** This makes blind trust dangerous. Responsible AI use requires systematic verification, especially for high-stakes applications.

### Questions

**Q19.** Based on your findings from this lab, complete this statement: "AI models are most likely to fail when..."

**Answer (list at least 3 scenarios):**

1. <br><br>
2. <br><br>
3. <br><br>

**Q20.** What strategies should you use to verify AI-generated information before trusting it? List at least 3 specific, actionable strategies.

**Strategy 1:**
<br><br>

**Strategy 2:**
<br><br>

**Strategy 3:**
<br><br>

**Q21.** Should AI systems always express uncertainty when they might be wrong? What are the trade-offs? Consider both user experience and safety.

**Answer:**
<br><br><br><br>

**Q22.** How does understanding AI's hallucination boundary change how you will use ChatGPT, Claude, or similar tools in the future? Be specific about what you'll do differently.

**Answer:**
<br><br><br><br>

**Q23.** (Synthesis) Explain to a friend who hasn't taken this class: Why can't AI reliably predict its own mistakes? Use concepts from this lab in your explanation.

**Answer:**
<br><br><br><br><br>

---

## Before You Submit

Make sure you have completed:

- [ ] Module 0: Generated prompts and answered Q1-Q3
- [ ] Module 1: Tested all 8 prompts on AI and answered Q4-Q7
- [ ] Module 2: Verified accuracy of all 8 responses and answered Q8-Q12
- [ ] Module 3: Reviewed visualizations and answered Q13-Q18
- [ ] Module 4: Read case studies and answered Q19-Q23
- [ ] All 23 questions answered on answer sheet
- [ ] Group code recorded for future reference
- [ ] Data files saved (prompts.csv, predictions.csv, evaluations.csv)

---

## Key Takeaways

By completing this lab, you should understand:

1. **AI confidence ≠ accuracy.** The tone of a response (confident, cautious, uncertain) does not reliably predict correctness.

2. **AI cannot predict its own failures.** Models lack reliable self-assessment capabilities—they can't tell you which questions will cause them to hallucinate.

3. **Overconfidence is systematic, not random.** Certain categories of questions (citations, specific facts, recent events) consistently trigger overconfident errors.

4. **Hallucinations are often highly specific.** Fabricated information includes detailed citations, statistics, and facts that seem credible until verified.

5. **Verification is essential for responsible AI use.** Cross-reference claims, check citations directly, and seek expert review for high-stakes decisions.

6. **The human-AI partnership requires human judgment.** Users must know when to verify, which sources to trust, and when to seek expert guidance.

7. **Design trade-offs matter.** AI systems balance helpfulness (providing answers) against honesty (acknowledging limits). Current systems lean toward helpfulness, creating overconfidence risk.

8. **Context matters.** Brainstorming and drafting tolerate more errors than legal briefs, medical advice, or academic citations.

---

## Connecting to AI and Machine Learning

| Lab Activity | ML Equivalent | Why It Matters |
|--------------|---------------|----------------|
| Testing AI confidence vs. accuracy | Model calibration in production systems | Poorly calibrated models make confident wrong predictions, causing real-world harm |
| Identifying prompt categories that cause errors | Failure mode analysis and dataset biases | Understanding systematic weaknesses guides when to use AI and when human oversight is needed |
| Verification strategies | Human-in-the-loop ML and active learning | Production AI systems need validation mechanisms, not just accuracy metrics |
| Comparing predicted vs. actual behavior | Out-of-distribution detection | Models must recognize when they're operating outside their training domain |
| Case studies of AI failures | Responsible AI and AI safety research | Real-world consequences drive policy, regulation, and technical improvements |

**The broader lesson:**

Machine learning models optimize for accuracy on training data, but they don't learn to assess their own reliability. This lab demonstrates why **trustworthy AI requires more than high accuracy**—it requires calibration, transparency, and human oversight.

Research on AI safety, alignment, and interpretability aims to build systems that:
- Know what they don't know (epistemic uncertainty)
- Communicate limitations clearly (uncertainty quantification)
- Refuse tasks beyond their capabilities (scope awareness)
- Enable verification (explainability and citations)

Your findings today reflect challenges that researchers and practitioners face when deploying AI at scale.

---

## Bridge to Future Topics

**From this lab:**
- AI models show overconfidence on certain prompt categories
- Verification requires independent sources and expert judgment
- Systematic testing reveals patterns invisible in individual cases

**To future courses:**
- **Machine Learning:** Calibration metrics, uncertainty estimation, out-of-distribution detection
- **AI Ethics:** Responsible deployment, transparency requirements, accountability frameworks
- **Natural Language Processing:** Citation generation, fact-checking systems, hallucination mitigation
- **Human-Computer Interaction:** Designing AI interfaces that communicate uncertainty effectively
- **Critical Thinking:** Evaluating information sources, recognizing cognitive biases, media literacy

**In your own use of AI:**
- Apply verification strategies to any AI-generated content you use
- Understand the limits of different AI models for different tasks
- Recognize overconfidence patterns and adjust your trust accordingly
- Advocate for transparency and accountability in AI systems you encounter

---

## Additional Resources

### Academic Papers on AI Hallucination
- "On the Dangers of Stochastic Parrots" (Bender et al., 2021)
- "TruthfulQA: Measuring How Models Mimic Human Falsehoods" (Lin et al., 2021)
- "Language Models (Mostly) Know What They Know" (Kadavath et al., 2022)

### Case Study Collections
- Partnership on AI: Case Studies Library
- AI Incident Database (incidentdatabase.ai)
- Stanford HAI: AI Audit

ing Cases

### Fact-Checking Resources
- Google Scholar (for academic citations)
- Snopes, FactCheck.org (for claims and statistics)
- Primary sources (government data, original publications)

### Further Reading
- "The Alignment Problem" by Brian Christian
- "Atlas of AI" by Kate Crawford
- OpenAI blog on model capabilities and limitations

---

**Questions or feedback?** Reach out to your instructor or TA!

**Remember:** The goal isn't to avoid AI entirely—it's to use AI responsibly by understanding its limits and verifying its outputs.
