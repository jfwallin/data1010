# Lab 3: When Lines Aren't Enough

## Activation Functions & Nonlinearity — Student Handout

**Course:** DATA 1010 – Artificial Intelligence in Action

---

## Overview

### What You'll Learn Today

Some patterns can't be separated by a straight line — no matter how hard you try. This lab walks you through the solution that powers every modern neural network:

1. **Straight lines fail** on certain dot patterns (Module 0)
2. **Activation functions bend space** so straight rules become curved boundaries (Module 1)
3. **Different activations have tradeoffs** — saturation, smoothness, simplicity (Module 2)
4. **A perceptron** combines a weighted sum with an activation — the basic building block (Module 3)
5. **One perceptron isn't enough** for the hardest patterns, motivating multi-layer networks (Module 4)

### Lab Structure

| Module | Title | Time |
|--------|-------|------|
| 0 | When Straight Lines Fail | ~5 min |
| 1 | Activation Functions — Bending Space | ~5 min |
| 2 | Activation Functions in Detail | ~5 min |
| 3 | Building a Perceptron | ~5 min |
| 4 | Testing the Perceptron's Limits | ~5 min |

**Total Time:** ~25 minutes of notebook work + discussion and answer-sheet writing

### Working in Groups

- Work in groups of **2–4 people**
- One person shares their screen running the notebooks
- Everyone participates in discussion and writes their own answers
- Talk through what you see before writing — explaining out loud helps everyone understand

---

## Module 0: When Straight Lines Fail

**What you'll do:** Try to draw a straight line that separates blue dots from red dots across three different patterns.

**What you'll discover:** One pattern is easy. The other two are impossible — no matter how you adjust the line.

### Questions (Q1–Q3)

**Q1.** Which dataset could you separate perfectly with a straight line? Describe what happened when you tried the slider on the other two — what kept going wrong no matter how you adjusted it?

**Q2.** For the ring dataset, sketch or describe what a boundary would have to look like. Why is a straight line doomed?

**Q3.** What is one thing a smarter boundary would need to be able to do that a straight line cannot?

---

## Module 1: Activation Functions — Bending Space

> **KEY IDEA:** Activation functions rearrange the dots first, so that a straight line can work afterward. Think of it like untangling a knot before you measure it.

**What you'll do:** Watch a square grid of points get warped by Sigmoid and ReLU. Then see how a straight rule in the warped space creates a curved boundary in the original space.

### Questions (Q4–Q6)

**Q4.** What did the activation function do to the grid of points? Use the before/after comparison.

**Q5.** After warping, a straight-line rule creates a curved boundary in the original space. How does this solve the problem from Q3?

**Q6.** Compare Sigmoid and ReLU warping. Which is more dramatic? What might be the tradeoff?

---

## Module 2: Activation Functions in Detail

**What you'll do:** Feed the same number into Sigmoid, ReLU, and Step and compare outputs — especially at extreme values.

### Key Vocabulary

- **Saturation:** When a function's output barely changes no matter how much bigger the input gets (like squeezing a sponge that's already dry).
- **Smooth:** Small input changes produce small output changes — important for learning.

### Questions (Q7–Q9)

**Q7.** Test a very large positive input (like 100) on Sigmoid and ReLU. What does each output? Which keeps changing?

**Q8.** Why is saturation a problem for a model that's trying to learn?

**Q9.** The Step function is the simplest — just on or off. Why isn't it the obvious choice for a learning system?

---

## Module 3: Building a Perceptron

> **KEY IDEA:** A perceptron is the single building block of every neural network. It does two steps: (1) weighted sum, (2) activation. Millions of them, connected in layers, power systems like ChatGPT.

**What you'll do:** Adjust weights and bias to move a decision boundary and classify dots. Try changing weights alone vs. bias alone to see what each controls.

### The Two Steps

```
Step 1: z = w₁·x₁ + w₂·x₂ + b     (weighted sum — defines a line)
Step 2: output = activation(z)       (bends the line into a curve)
```

### Questions (Q10–Q12)

**Q10.** Describe each step in plain language — what goes in, what happens, what comes out?

**Q11.** What changes when you adjust only weights? Only bias? What's the difference?

**Q12.** Where in the perceptron does the "bending" happen — Step 1 or Step 2? Why does that matter?

---

## Module 4: Testing the Perceptron's Limits

**What you'll do:** Take your perceptron back to the XOR and circle patterns from Module 0 and try to classify them. You'll discover the ceiling.

### Questions (Q13–Q15)

**Q13.** What was your best accuracy on XOR and circles? What ceiling did you keep hitting?

**Q14.** How many lines would you need to separate XOR's four corners? Describe or sketch where you'd place them.

**Q15.** Look back at the whole lab arc: straight lines failed → activation functions bent space → a single perceptron still hit a wall. What's the logical next move?

---

## Key Takeaways

| Concept | What you learned |
|---------|-----------------|
| **Linear limits** | Some patterns (XOR, rings) can never be separated by a straight line. |
| **Activation functions** | They warp space so that a simple straight rule becomes a curved boundary. |
| **Saturation** | When an activation flattens out, the model loses its learning signal. |
| **Perceptron** | Two steps (weighted sum → activation) = the basic building block. |
| **Single-perceptron limit** | One perceptron = one boundary. XOR and rings need more. |
| **Next step** | Multiple perceptrons in layers = a neural network that can handle anything. |

---

## Connection to Previous Labs

**Lab 1:** You learned about models, parameters, and optimization.

**Lab 2:** You learned how gradient descent automatically finds the best parameters.

**Lab 3 (this lab):** You learned why nonlinearity is essential — and built the basic unit that makes neural networks work.

---

## Before You Submit

Make sure you have:

- [ ] Completed all 5 modules (Module 0–4)
- [ ] Answered all 15 questions (Q1–Q15) on the answer sheet
- [ ] Discussed your answers with your group members
- [ ] Written thoughtful, complete answers in your own words

---

**Questions or issues?** Check the LMS discussion board or ask your instructor/TA.
