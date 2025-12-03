# Lab 8: Diffusion Models — From Noise to Images
## Answer Sheet

**Course:** DATA 1010 – Artificial Intelligence in Action

**Student Name:** _________________________________

**Date:** _________________________________

---

## Module 0: What Is a Diffusion Model?

### Q1. In your own words, what is forward diffusion? What is reverse diffusion?

*(Hint: Think about the two-phase process—destruction and reconstruction)*

**Answer:**

**Forward diffusion:**

<br><br>

**Reverse diffusion:**

<br><br><br>

### Q2. Why is it useful to train a model to reverse the noise process? How does this help with image generation?

**Answer:**

<br><br><br>

### Q3. How is diffusion different from the CNNs you learned about in Lab 7? Look at the comparison table above.

**Answer:**

<br><br><br>

### Q4. Predict: When you start from pure random noise at t=200, what determines what image gets generated?

*(Make your prediction BEFORE continuing to Module 1!)*

**Answer:**

<br><br><br>

---

## Module 1: Forward Diffusion Demo

### Q5. Looking at the progression visualization, at approximately what timestep does the digit become unrecognizable to you?

**Answer:**

**Timestep:** _______________

**Why at this timestep?**

<br><br>

### Q6. Compare the noisy images at t=50 vs t=150. Describe the visual differences.

**Answer:**

**At t=50:**

<br><br>

**At t=150:**

<br><br>

**Key difference:**

<br><br>

### Q7. Looking at the SNR plot, at what approximate timestep does noise start to dominate the signal (SNR < 1)?

*(Hint: Look for where the SNR curve crosses the horizontal line at 1.0)*

**Answer:**

**Timestep:** _______________

<br><br>

### Q8. In the "different noise realizations" experiment, why do all 5 noisy images look different even though they started from the same original?

*(Hint: Think about what's random in the forward diffusion process)*

**Answer:**

<br><br><br>

### Q9. Why is the forward diffusion process easy to implement, but reversing it requires training a neural network?

*(Hint: Compare adding noise vs. removing noise. Which direction has a clear formula?)*

**Answer:**

<br><br><br>

---

## Module 2: Training a Toy Denoiser

### Q10. Before training, predict: How long will training take with your current NUM_CLASSES setting?

*(Make your prediction BEFORE running the training cell!)*

**My prediction:** ________________

**NUM_CLASSES used:** _______________

**Actual training time:** ________________

**Was your prediction close?**

<br><br>

### Q11. What final loss did you achieve? Is it close to the target of ~0.05?

**Final loss:** ________________

**Is it close to 0.05?**

<br><br>

### Q12. How long did training take? Was it faster or slower than your prediction from Q10?

**Training time:** ________________

**Comparison to prediction:**

<br><br>

### Q13. At t=100, does the denoiser successfully remove noise? How does the denoised image compare to the original?

**Answer:**

<br><br><br>

### Q14. Compare denoising performance at t=25 vs t=150. At which timestep does the model perform better? Why?

*(Hint: Think about how much noise is present at each timestep)*

**Answer:**

**Better at t=_____**

**Why:**

<br><br><br>

### Q15. How is this similar to Lab 7, Module 4 where you trained a CNN on MNIST? How is it different?

**Similar:**

<br><br>

**Different:**

<br><br><br>

---

## Module 3: Multi-Step Denoising (Reverse Diffusion)

### Q16. Predict: At approximately what timestep (t value) does the digit start to become recognizable?

*(Make your prediction BEFORE viewing the trajectory!)*

**My prediction:** t = _______________

**Actual observation:** t = _______________

**Why this makes sense:**

<br><br>

### Q17. Looking at the full trajectory, describe how structure emerges. Does it appear suddenly or gradually?

**Answer:**

<br><br><br>

### Q18. Looking at the 12 generated samples, what do you notice about variety and quality? Are all digits equally clear?

**Variety:**

<br><br>

**Quality:**

<br><br>

**Are all equally clear?**

<br><br>

### Q19. Compare the 50-step vs 200-step generations. Is the quality difference large or small? Is 200 steps necessary?

**Quality difference:**

<br><br>

**Is 200 steps necessary?**

<br><br>

**Trade-offs to consider:**

<br><br>

### Q20. Why do you think the generated digits aren't perfect? What limitations does this toy model have?

*(Hint: Think about model size, training data, resolution)*

**Answer:**

**Main limitations:**
1.

2.

3.

<br><br>

---

## Module 4: Professional Pre-Trained Model

### Q21. Look at the 16 generated images. How many distinct object categories can you identify?

*(Hint: CIFAR-10 has 10 classes: airplane, car, bird, cat, deer, dog, frog, horse, ship, truck)*

**Answer:**

**Number of categories:** _______________

**Categories I identified:**

<br><br><br>

### Q22. Compare the images generated with 25, 50, and 100 steps. Describe the quality differences you observe. Is the improvement from 50→100 steps as noticeable as 25→50 steps?

**Answer:**

**25 steps:**

<br><br>

**50 steps:**

<br><br>

**100 steps:**

<br><br>

**Diminishing returns?**

<br><br>

### Q23. Why might DALL-E and Stable Diffusion use 20-50 steps instead of 100+ steps? What's the trade-off?

*(Hint: Consider both user experience and computational cost)*

**Answer:**

**Why 20-50 steps:**

<br><br>

**Trade-off:**

<br><br><br>

### Q24. Looking at the comparison table, what specific factors contribute most to the quality difference between your toy model and the professional CIFAR-10 model?

**Answer:**

**Most important factors:**
1.

2.

3.

**Why these matter most:**

<br><br><br>

### Q25. In your own words, explain the complete diffusion pipeline: How does a model like DALL-E go from a text prompt ('a cat in a spacesuit') to a final image? Reference concepts from Modules 0-4.

*(This is the synthesis question—tie everything together!)*

**Answer:**

**Step-by-step explanation:**

1.

<br>

2.

<br>

3.

<br>

4.

<br>

5.

<br><br>

**Key concepts used:**

<br><br>

---

## Reflection (Optional)

### What was the most surprising thing you learned about diffusion models?

**Answer:**

<br><br><br>

### How does understanding diffusion models change your perspective on AI-generated art and content?

**Answer:**

<br><br><br>

### What connections do you see between Lab 8 (diffusion) and Lab 7 (CNNs)?

**Answer:**

<br><br><br>

---

## Before You Submit

Make sure you have:

- [ ] Completed all 5 modules (0, 1, 2, 3, 4)
- [ ] Answered all 25 questions (Q1-Q25)
- [ ] Made predictions BEFORE viewing results (Q4, Q10, Q16)
- [ ] Trained the toy denoiser in Module 2
- [ ] Generated images from noise in Module 3
- [ ] Used the professional CIFAR-10 model in Module 4
- [ ] Recorded specific observations where requested
- [ ] Compared toy model to professional model (Q24)
- [ ] Explained the complete pipeline (Q25)
- [ ] Run all code cells and viewed visualizations
- [ ] Written thoughtful, complete answers
- [ ] (Optional) Completed reflection questions
- [ ] Connected concepts to Lab 7 where relevant

---

## Submission Instructions

Submit this completed answer sheet according to your instructor's guidelines (Canvas upload, hardcopy, PDF, etc.).

**Congratulations on completing Lab 8!** You now understand how **diffusion models generate images from noise**—the revolutionary technology powering DALL-E, Midjourney, and Stable Diffusion!

---

## What You've Accomplished

By completing this lab, you've:

✅ Understood the two-phase diffusion process (forward and reverse)
✅ Implemented forward diffusion (noise addition)
✅ Trained a neural network denoiser
✅ Generated images from pure random noise
✅ Compared toy models to professional models
✅ Connected the algorithm to DALL-E and real-world AI

**Most importantly:** You implemented the SAME algorithm used by DALL-E, Midjourney, and Stable Diffusion. The only difference is scale—you have the knowledge to build cutting-edge generative AI systems!

---

## Next Steps in Your AI Journey

### Continue Learning:
- Explore Stable Diffusion and generate your own images
- Study advanced diffusion techniques (latent diffusion, classifier-free guidance)
- Learn about other generative models (GANs, VAEs)
- Build creative AI applications

### Apply Your Knowledge:
- Use diffusion models for creative projects
- Understand the capabilities and limitations of AI art generators
- Think critically about AI-generated content
- Explore career opportunities in AI and machine learning

**The future of AI is generative—and you now understand how it works!** 🎨🚀
