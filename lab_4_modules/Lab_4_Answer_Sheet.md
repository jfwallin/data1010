# Lab 4: Building & Training Neural Networks
## Answer Sheet

**Course:** DATA 1010 – Artificial Intelligence in Action

**Group Members (2–4 names)**

1. _______________________________ 2. _______________________________

3. _______________________________ 4. _______________________________

---

## Module 0: Lifting Dimensions  (~12 min)

### Q1. In 2D, can you draw a straight line that separates the XOR pattern? After you added x₃ = x₁ × x₂ and looked at the 3D plot, what changed? Describe how the classes became separable.

**Answer:**

<br><br><br><br>

### Q2. When you look straight down at the 3D separating plane (bird's-eye view), what shape does the decision boundary have in the original 2D space? Why isn't it a straight line?

**Answer:**

<br><br><br>

### Q3. How is adding a third dimension similar to what activation functions did in Lab 3? (Hint: both transform the data so a simple rule can work.)

**Answer:**

<br><br><br>

---

## Module 1: Anatomy of a Tiny Neural Network  (~15 min)

### Q4. How many total parameters (weights + biases) does the 2-2-1 network have? What do the two hidden neurons represent, and how do they relate to the dimension-lifting you did in Module 0?

**Answer:**

<br><br><br><br>

### Q5. When you adjust the weights connecting inputs to the hidden layer, what changes about the decision boundary? What changes when you adjust only the bias?

**Answer:**

<br><br><br><br>

### Q6. Why do we need TWO hidden neurons instead of just one? What happens when you try to solve XOR with only one hidden neuron?

**Answer:**

<br><br><br>

---

## Module 2: Training a Neural Network  (~15 min)

### Q7. What is the "loss function" and why do we want to minimize it? Describe it as if you were explaining to a friend who has never taken this class.

**Answer:**

<br><br><br><br>

### Q8. Describe what happens when the learning rate is too small vs. too large. How does adding momentum help?

**Answer:**

**Too small:**

<br><br>

**Too large:**

<br><br>

**Momentum helps by:**

<br><br>

### Q9. When you reset the network and retrain from different random starting points, did every run reach the same final result? What does this tell you about training neural networks?

**Answer:**

<br><br><br>

---

## Module 3: Penguin Species Classification  (~20 min)

> The Palmer Penguins dataset has 4 measurements (bill length, bill depth, flipper length, body mass) and 3 species (Adelie, Chinstrap, Gentoo). This is the same kind of problem as XOR — but with real data, more inputs, and more classes.

### Q10. Did the linear model (no hidden layer) achieve high accuracy on Penguins? How much did adding a hidden layer improve it? Why do you think the improvement was large or small?

**Answer:**

Linear model accuracy: ________%

Hidden layer model accuracy: ________%

Explanation:

<br><br><br>

### Q11. Experiment with different numbers of hidden units. Record your results in the table. At what point do you see diminishing returns?

| Hidden Units | Test Accuracy |
|--------------|---------------|
| 2            | ________%     |
| 4            | ________%     |
| 8            | ________%     |
| 16           | ________%     |
| 32           | ________%     |
| 64           | ________%     |

Diminishing returns appear around: _______ units

<br>

### Q12. After running each model 5 times, what was the mean accuracy ± standard deviation for each? Did the box plots overlap — and what does that tell you about whether the hidden layer truly helps?

**Answer:**

Linear model (5 runs): ________% ± ________%

Hidden layer model (5 runs): ________% ± ________%

Do the box plots overlap?  **Yes** / **No**

What this means:

<br><br><br>

---

## Module 4: Breast Cancer Classification  (~25 min)

> The Wisconsin Breast Cancer dataset has 30 measurements from cell images and a binary label: benign or malignant. This is real medical data — the same principles from XOR apply, but errors have real consequences.

### Q13. How did the baseline linear model (no hidden layers) perform? Were you surprised that a simple model works well with 30 features? Experiment with architectures and record your results.

**Answer:**

Baseline accuracy: ________%

| Hidden Layers | Units per Layer | Test Accuracy | False Negatives |
|--------------|-----------------|---------------|-----------------|
| 0            | N/A             | ________%     | ________        |
| 1            | 16              | ________%     | ________        |
| 1            | 32              | ________%     | ________        |
| 2            | 16              | ________%     | ________        |

Best architecture: _______ layers, _______ units

Were you surprised?

<br><br><br>

### Q14. In medical diagnosis, which error is worse — a false positive (predicting cancer when there isn't any) or a false negative (missing real cancer)? Look at your confusion matrices: did more complex models reduce the number of missed cancers?

**Answer:**

More concerning error (circle one):  **False Positive** / **False Negative**

Reasoning:

<br><br><br>

Baseline false negatives: ________

Best model false negatives: ________

Did complexity help reduce missed cancers?

<br><br>

### Q15. Look back at the full arc of this lab — from lifting XOR into 3D, to manually tuning 9 weights, to watching gradient descent learn, to classifying penguins, to diagnosing cancer. What is the connection between adding x₃ = x₁ × x₂ in Module 0 and hidden layers in Keras? What's the SAME between 2-feature XOR and 30-feature cancer diagnosis?

**Answer:**

<br><br><br><br><br>

---

## Before You Submit

Make sure you have:

- [ ] Completed all 5 modules (Module 0–4)
- [ ] Answered all 15 questions (Q1–Q15)
- [ ] Experimented with the 3D visualization in Module 0
- [ ] Manually adjusted weights in Module 1
- [ ] Trained with different learning rates and momentum in Module 2
- [ ] Filled in both experiment tables (Q11, Q13)
- [ ] Recorded mean ± std from multiple runs (Q12)
- [ ] Discussed medical ethics (Q14)
- [ ] Written thoughtful, complete answers in your own words

---

**Submission Instructions:**

Submit this completed answer sheet according to your instructor's guidelines (PDF upload, hardcopy, etc.).
