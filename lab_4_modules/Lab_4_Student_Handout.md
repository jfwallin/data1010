# Lab 4: Building & Training Neural Networks

## Student Handout

**Course:** DATA 1010 – Artificial Intelligence in Action

---

## Overview

### What You'll Learn Today

In Lab 3 you discovered that a single perceptron hits a wall — it can only draw one boundary. This lab picks up where that left off. You'll see WHY multiple neurons help, HOW a network learns on its own, and what happens when you point these ideas at real data.

The story goes like this:

1. **Adding dimensions solves impossible problems** — XOR can't be separated in 2D, but it can in 3D (Module 0)
2. **A tiny neural network is just 9 numbers** — and those numbers control everything (Module 1)
3. **Gradient descent finds good numbers automatically** — no more manual slider tuning (Module 2)
4. **Real data works the same way** — classify penguin species from body measurements (Module 3)
5. **High-stakes predictions need careful thinking** — cancer diagnosis, where errors have consequences (Module 4)

### Lab Structure

| Module | Title | What Happens | Time |
|--------|-------|--------------|------|
| 0 | Lifting Dimensions | See XOR become separable in 3D | ~12 min |
| 1 | Anatomy of a Tiny Neural Network | Manually tune 9 weights to solve XOR | ~15 min |
| 2 | Training a Neural Network | Watch gradient descent learn automatically | ~15 min |
| 3 | Penguin Species Classification | Apply neural networks to real animal data | ~20 min |
| 4 | Breast Cancer Diagnosis | Apply to medical data, analyze errors | ~25 min |

**Total Time:** ~90 minutes

### Working in Groups

- Work in groups of **2–4 people**
- One person shares their screen running the notebooks
- Everyone participates in discussion and writes their own answers
- Talk through what you see before writing — explaining out loud helps everyone understand

---

## Module 0: Lifting Dimensions

> **THE BIG IDEA:** XOR is impossible in 2D — no straight line works. But if you add the right third dimension, the classes pop apart and a flat plane separates them perfectly. This is exactly what hidden layers do inside a neural network.

**What you'll do:** Revisit the XOR pattern from Lab 3. Try (and fail) to separate it with a line. Then add a third coordinate — x₃ = x₁ × x₂ — and watch the two classes separate in 3D space.

**What to notice:**
- Only the **product** x₁ × x₂ works as a third dimension — sum, squares, and other combinations don't help
- A flat plane in 3D creates a **curved** boundary when projected back to 2D
- This is the same "warping" idea from Lab 3's activation functions

### Questions (Q1–Q3)

**Q1.** In 2D, can you draw a straight line that separates XOR? After you added x₃ = x₁ × x₂ and looked at the 3D plot, what changed?

**Q2.** When you look straight down at the 3D separating plane, what shape does the boundary have in 2D? Why isn't it a straight line?

**Q3.** How is adding a third dimension similar to what activation functions did in Lab 3?

---

## Module 1: Anatomy of a Tiny Neural Network

> **THE BIG IDEA:** A neural network is just a collection of numbers — weights and biases. A 2-2-1 network has exactly 9 parameters. The two hidden neurons each create a new dimension, just like the x₃ you built by hand in Module 0 — except the network learns the best transformation automatically.

**What you'll do:** Examine a 2-2-1 network (2 inputs, 2 hidden neurons, 1 output). Use 9 sliders to adjust every weight and bias. Watch how the decision boundary changes as you tune them.

**Key vocabulary:**
- **Weights** — multiply the inputs; control the angle/tilt of boundaries
- **Biases** — shift boundaries left/right without changing their angle
- **Hidden neurons** — create new dimensions from the inputs (like x₃ in Module 0)

**What to notice:**
- Changing weights **rotates** the boundary
- Changing biases **slides** the boundary
- You need **two** hidden neurons for XOR — one isn't enough (just like one line wasn't enough in Lab 3)

### Questions (Q4–Q6)

**Q4.** How many total parameters does the 2-2-1 network have? What do the hidden neurons represent, and how do they connect to Module 0's dimension-lifting?

**Q5.** What changes when you adjust weights vs. bias?

**Q6.** Why do we need two hidden neurons instead of one?

---

## Module 2: Training a Neural Network

> **THE BIG IDEA:** In Module 1 you tuned 9 weights by hand — slow and frustrating. Gradient descent does the same thing automatically: it measures how wrong the predictions are (the loss), figures out which direction to nudge each weight, and repeats. Momentum makes this faster. Different random starting points lead to different results.

**What you'll do:** Watch gradient descent train the same 2-2-1 network to solve XOR. Experiment with learning rates, momentum, and different starting points.

**Key vocabulary:**
- **Loss function** — a single number measuring how wrong the network is (lower = better)
- **Learning rate** — how big each weight-update step is
- **Momentum** — remembers previous steps so the network doesn't get stuck
- **Multi-start** — trying several random starting points and picking the best

**What to notice:**
- Too-small learning rate: creeps along, may never finish
- Too-large learning rate: bounces wildly, may get worse
- Momentum smooths things out and speeds convergence
- Different starting points can reach different results — training is **stochastic**

### Questions (Q7–Q9)

**Q7.** What is the loss function, and why do we want to minimize it?

**Q8.** What happens when the learning rate is too small vs. too large? How does momentum help?

**Q9.** When you reset and retrain from different starting points, did every run reach the same result? What does this tell you?

---

## Module 3: Penguin Species Classification

> **THE BIG IDEA:** Everything you just learned on XOR applies to real data. The Palmer Penguins dataset has 4 body measurements and 3 species — same structure, real animals. Keras automates the gradient descent from Module 2 so you can focus on results. You'll also discover that running the same experiment twice gives different numbers, so professionals always report statistics.

**What you'll do:**
1. Load the Palmer Penguins dataset (bill length, bill depth, flipper length, body mass)
2. Train a **linear model** (no hidden layers) and check accuracy
3. Train a **hidden layer model** and compare
4. Experiment with different numbers of hidden units (2, 4, 8, 16, 32, 64)
5. Run each model **5 times** and see how much results vary

**The Palmer Penguins dataset:**
- 333 penguins from islands near Antarctica
- 3 species: **Adelie**, **Chinstrap**, **Gentoo**
- 4 measurements per penguin — things you can picture and understand
- Task: given the measurements, predict the species

**What to notice:**
- A simple linear model may already get high accuracy — not every problem needs a deep network
- Adding hidden units helps, but with **diminishing returns**
- Results **change between runs** because of random weight initialization
- Professional ML practice: report "94.2% ± 1.5%" not just "95%"

### Questions (Q10–Q12)

**Q10.** Did the linear model achieve high accuracy? How much did adding a hidden layer improve it?

**Q11.** Record your results for different numbers of hidden units. Where do diminishing returns start?

**Q12.** After running each model 5 times, what was the mean ± std? Did the box plots overlap?

---

## Module 4: Breast Cancer Classification

> **THE BIG IDEA:** Same principles, higher stakes. The Wisconsin Breast Cancer dataset has 30 measurements from cell images and a binary label: benign or malignant. A missed cancer (false negative) is far more dangerous than a false alarm (false positive). Accuracy alone doesn't tell the full story — you need to look at the **confusion matrix** to see where errors land.

**What you'll do:**
1. Load the breast cancer dataset (30 features, 569 samples)
2. Train a **baseline linear model** — see that it works surprisingly well
3. Experiment with architectures (different layers and units)
4. Read **confusion matrices** — count false positives and false negatives
5. Run multiple experiments and compare models statistically

**The breast cancer dataset:**
- 569 tumor samples from the University of Wisconsin
- 30 numeric features computed from cell nucleus images (radius, texture, area, etc.)
- Binary: **Benign** (not cancer) or **Malignant** (cancer)
- Task: given cell measurements, predict whether the tumor is cancerous

**Key vocabulary:**
- **False Positive** — predicting cancer when there isn't any (stress, extra tests, but not life-threatening)
- **False Negative** — missing real cancer (delayed treatment, potentially fatal)
- **Confusion Matrix** — a 2×2 table showing exactly how many of each error type the model made

**What to notice:**
- With 30 features, even a linear model can reach ~95% — high-dimensional data is often easier to separate
- Adding complexity may not help much — simpler can be better
- The **type** of error matters as much as the rate
- If false negatives vary from 2 to 4 depending on random initialization, that's a real problem for medical deployment

### Questions (Q13–Q15)

**Q13.** How did the baseline linear model perform? Record your architecture experiments — which was best?

**Q14.** Which error is worse in medical diagnosis — false positive or false negative? Did more complex models reduce missed cancers?

**Q15.** Look back at the full arc: lifting XOR to 3D → tuning 9 weights → gradient descent → penguins → cancer diagnosis. What's the connection between Module 0's dimension-lifting and hidden layers in Keras? What's the same between 2-feature XOR and 30-feature cancer diagnosis?

---

## Key Takeaways

| Concept | What you learned |
|---------|-----------------|
| **Dimension lifting** | Adding the right new dimension makes impossible problems solvable. Hidden layers do this automatically. |
| **Network parameters** | A 2-2-1 network has 9 numbers. Weights rotate boundaries, biases slide them. |
| **Gradient descent** | Automatically finds good weights by rolling downhill on the loss landscape. |
| **Learning rate & momentum** | Step size matters. Momentum helps escape flat spots and speeds things up. |
| **Stochastic training** | Different starting points → different results. Always run multiple times. |
| **Real-world ML** | Same principles work on penguins (4 features) and cancer (30 features). |
| **Confusion matrices** | Accuracy isn't everything — the type of error matters, especially in medicine. |
| **Statistical reporting** | Report mean ± std, not a single number. Compare distributions, not single runs. |

---

## Connection to Previous Labs

**Lab 1:** You learned about models, parameters, and optimization basics.

**Lab 2:** You learned how gradient descent automatically finds the best parameters.

**Lab 3:** You learned why nonlinearity is essential — activation functions bend space, and a single perceptron still hits a wall on XOR.

**Lab 4 (this lab):** You learned why multiple neurons solve that wall (dimension lifting), how networks learn on their own (gradient descent), and what happens when you apply these ideas to real data with real consequences.

---

## Before You Submit

Make sure you have:

- [ ] Completed all 5 modules (Module 0–4)
- [ ] Answered all 15 questions (Q1–Q15) on the answer sheet
- [ ] Experimented with the 3D visualization in Module 0
- [ ] Manually adjusted weights in Module 1
- [ ] Trained with different learning rates and momentum in Module 2
- [ ] Filled in both experiment tables (Q11, Q13)
- [ ] Recorded mean ± std from multiple runs (Q12)
- [ ] Discussed medical ethics (Q14)
- [ ] Written thoughtful, complete answers in your own words

---

**Questions or issues?** Check the LMS discussion board or ask your instructor/TA.
