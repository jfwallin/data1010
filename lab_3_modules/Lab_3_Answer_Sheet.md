# Lab 3: When Lines Aren't Enough
## Activation Functions & Nonlinearity  ·  Answer Sheet

**Course:** DATA 1010 – Artificial Intelligence in Action

**Group Members (2–4 names)**

1. _______________________________ 2. _______________________________

3. _______________________________ 4. _______________________________

---

## Module 0: When Straight Lines Fail  (~5 min)

### Q1. Which dataset could you separate perfectly with a straight line? Describe what happened when you tried the slider on the other two — what kept going wrong no matter how you adjusted it?

**Answer:**

<br><br><br><br>

### Q2. For the ring dataset, sketch or describe what a boundary would have to look like to correctly put the inner circle on one side and the outer ring on the other. Why is a straight line doomed before you even start?

**Answer:**

<br><br><br><br>

### Q3. Based on what you saw, what is one thing a smarter boundary would need to be able to do that a straight line simply cannot?

**Answer:**

<br><br><br>

---

## Module 1: Activation Functions — Bending Space  (~5 min)

> **KEY IDEA:** In Module 0, some dot patterns couldn't be separated by any straight line. One solution would be to invent a more complicated boundary — but that gets messy fast.
> Activation functions take a completely different approach: they rearrange the dots first, so that a straight line can work afterward. Think of it like untangling a knot before you measure it.
> Keep that idea in mind as you answer these questions.

### Q4. In your own words, what did the activation function do to the grid of points? Use the before/after comparison in your answer.

**Answer:**

<br><br><br><br>

### Q5. After the warping, you drew what looked like a straight-line rule — but it created a curved boundary in the original space. How does this solve the problem you identified in Q3?

**Answer:**

<br><br><br><br>

### Q6. Compare how Sigmoid and ReLU each warped the grid. Which changed the space more dramatically? What might be a tradeoff between a dramatic warp and a gentler one?

**Answer:**

<br><br><br><br>

---

## Module 2: Activation Functions in Detail  (~5 min)

### Q7. Test a very large positive input (like 100) on Sigmoid and then on ReLU. What does each one output? Which one keeps changing, and which one flattens out?

**Answer:**

<br><br><br>

### Q8. When a function "saturates," its output barely changes even as the input keeps growing — like squeezing a sponge that's already dry. Why would that be a problem for a model that's trying to learn and adjust itself?

**Answer:**

<br><br><br><br>

### Q9. The Step function is the simplest of all — just on or off, like a light switch. If simple is usually good, why isn't Step the obvious choice for a learning system? What does it lose by being so rigid?

**Answer:**

<br><br><br><br>

---

## Module 3: Building a Perceptron  (~5 min)

> **KEY IDEA:** A perceptron is the single building block of every neural network. It is tiny and simple on its own — but millions of them, connected in layers, power systems like ChatGPT and image recognition.
> Before answering, look closely at the two-step diagram in the notebook. Make sure you can trace what happens to a number as it moves through the perceptron from input to output.

### Q10. Look at the two-step diagram in the notebook. Without using any math, describe each step in plain language — what goes in, what happens, and what comes out?

**Answer:**

<br><br><br><br>

### Q11. Try adjusting only the weights while keeping the bias fixed. What changes about the decision boundary? Now try adjusting only the bias. What changes? Describe the difference between what each one controls.

**Answer:**

<br><br><br><br>

### Q12. You've now seen activation functions bend space (Module 1) and a perceptron combine weights, bias, and an activation function (this module). Where exactly in the perceptron does the "bending" happen — Step 1 or Step 2? Why does that matter for what kinds of patterns the perceptron can separate?

**Answer:**

<br><br><br><br>

---

## Module 4: Testing the Perceptron's Limits  (~5 min)

### Q13. What was your best accuracy on XOR? On the circles? Describe what kept happening each time you tried a new setting — what ceiling did you keep hitting, and why couldn't you push past it?

**Answer:**

<br><br><br><br>

### Q14. A single perceptron can only draw one straight line. How many lines would you actually need to correctly separate XOR's four corners? Describe or sketch where you would place them.

**Answer:**

<br><br><br><br>

### Q15. Look back at the whole arc of this lab: straight lines failed → activation functions bent space → a single perceptron still hit a wall. What is the logical next move? What would you add to the system to finally break through?

**Answer:**

<br><br><br><br>

---

## Before You Submit

Make sure you have:

- [ ] Completed all 5 modules (Module 0–4) using the notebooks
- [ ] Answered all 15 questions (Q1–Q15)
- [ ] Tried the slider on all three datasets in Module 0
- [ ] Compared Sigmoid and ReLU grid warping in Module 1
- [ ] Tested large inputs in Module 2
- [ ] Adjusted weights and bias separately in Module 3
- [ ] Attempted to classify XOR and circles in Module 4
- [ ] Written thoughtful, complete answers
- [ ] Discussed your answers with your group members

---

**Submission Instructions:**

Submit this completed answer sheet according to your instructor's guidelines (PDF upload, hardcopy, etc.).
