# Lab 8: Diffusion Models — From Noise to Images

**Course:** DATA 1010 – Artificial Intelligence in Action

---

## Overview

### What You'll Learn Today

In today's lab, you will explore **diffusion models**—the revolutionary technology behind DALL-E, Midjourney, and Stable Diffusion. You'll learn to:

- Understand what diffusion models are and how they generate images from noise
- Implement forward diffusion (progressive noise addition)
- Train a denoiser network to predict and remove noise
- Generate images from pure random noise using reverse diffusion
- Use a professional pre-trained model and see dramatic quality improvements
- Connect the toy model you build to real-world text-to-image systems

### Lab Structure

This lab consists of **5 modules** that you'll complete in order:

| Module | Title | Time | Type |
|--------|-------|------|------|
| 0 | What Is a Diffusion Model? | ~5-8 min | Conceptual |
| 1 | Forward Diffusion Demo | ~8-12 min | Colab |
| 2 | Training a Toy Denoiser | ~18-22 min | Colab |
| 3 | Multi-Step Denoising (Generation) | ~12-15 min | Colab |
| 4 | Professional Pre-Trained Model | ~12-15 min | Colab |

**Total Time:** ~60-75 minutes

### Working in Groups

- Work in **small groups** (2-4 people)
- One group member runs Colab and shares their screen
- Everyone participates in discussion and predictions
- All group members can use their own notebooks

### Key Concepts

> **DIFFUSION MODEL**: A generative model that learns to create images by reversing a noise-adding process. It gradually transforms random noise into coherent images.
>
> **FORWARD DIFFUSION**: The process of progressively adding noise to an image over many timesteps until it becomes pure random noise.
>
> **REVERSE DIFFUSION**: The process of progressively removing noise from a random noise image to create a coherent image.
>
> **DENOISER**: A neural network (typically U-Net) that predicts the noise in an image at a given timestep, enabling reverse diffusion.
>
> **U-NET**: An encoder-decoder architecture with skip connections, ideal for image-to-image tasks like denoising.
>
> **DDPM (Denoising Diffusion Probabilistic Model)**: The specific algorithm for reverse diffusion that you'll implement.
>
> **UNCONDITIONAL GENERATION**: Generating random images from noise without specifying what to create.
>
> **CONDITIONAL GENERATION**: Generating images from noise guided by a condition (e.g., text prompt).

### Connection to Previous Labs

**Lab 2:** You learned about gradient descent for training models
- **Lab 8:** Same training process, just with different architecture

**Lab 4:** You trained a classifier with gradient descent
- **Lab 8:** Training a denoiser uses the same optimization approach

**Lab 6:** Saliency maps showed **WHERE** a model looks
- **Lab 8:** Feature visualization shows **WHAT** a model creates

**Lab 7:** You learned how CNNs analyze images (image → classification)
- **Lab 8:** Diffusion models synthesize images (noise → image)

**The critical connection to Lab 7:**
```
Lab 7 (CNNs):      Image → Features → Classification
                   (Analysis: "What is this?")

Lab 8 (Diffusion): Noise → Features → Image
                   (Synthesis: "Create this!")
```

**Both use convolutional architectures—one for understanding, one for creating!**

---

## Module 0: What Is a Diffusion Model?

**Time:** ~5-8 minutes
**Type:** Conceptual (markdown only)

### Learning Objectives
- Understand diffusion as a two-phase process (forward and reverse)
- Grasp the intuition through multiple analogies
- Compare diffusion models to CNNs from Lab 7
- Predict how noise and randomness affect generation

### What You'll Do

1. Learn the core metaphor: "Sculpting in reverse"
2. Understand forward diffusion (adding noise progressively)
3. Understand reverse diffusion (removing noise step-by-step)
4. Explore analogies: scrambled eggs, fog, detective work
5. Compare to Lab 7: CNNs analyze, diffusion models create
6. See real-world examples: DALL-E, Midjourney, Stable Diffusion

### Key Insight

> **Diffusion models learn to reverse a destruction process. By training a model to undo noise corruption, we can start from pure noise and gradually sculpt it into meaningful images—the same process behind DALL-E and Stable Diffusion!**

### Questions

**Q1.** In your own words, what is forward diffusion? What is reverse diffusion?

**Q2.** Why is it useful to train a model to reverse the noise process? How does this help with image generation?

**Q3.** How is diffusion different from the CNNs you learned about in Lab 7? Look at the comparison table above.

**Q4.** Predict: When you start from pure random noise at t=200, what determines what image gets generated?

---

## Module 1: Forward Diffusion Demo

**Time:** ~8-12 minutes
**Type:** Colab notebook

### Learning Objectives
- Implement forward diffusion on MNIST digits
- Understand noise schedules (beta, alpha, alpha_bar)
- Visualize progressive information destruction
- Analyze Signal-to-Noise Ratio (SNR) over time
- Understand stochasticity in the diffusion process

### What You'll Do

1. Load a single MNIST digit
2. Implement the forward diffusion formula
3. Visualize noise addition at 9 timesteps (t=0 to t=199)
4. Plot noise schedule curves (alpha_bar vs timestep)
5. Analyze SNR in both linear and dB scales
6. Generate 5 different noise realizations at t=100

### The Mathematics (Three-Tier Presentation)

**Symbols:**
```
x_t = √(ᾱ_t) · x_0 + √(1 - ᾱ_t) · ε
```

**Plain English:**
> "Noisy image at time t = (signal weight × original) + (noise weight × random noise)"

**Pseudocode:**
```python
noisy_image = sqrt(alpha_bar_t) * original + sqrt(1 - alpha_bar_t) * noise
```

### Key Insight

> **Forward diffusion gradually destroys information in a controlled, predictable way. The noise schedule (alpha_bar) determines how quickly information is lost. This gradual destruction is what makes reverse diffusion learnable!**

### Questions

**Q5.** Looking at the progression visualization, at approximately what timestep does the digit become unrecognizable to you?

**Q6.** Compare the noisy images at t=50 vs t=150. Describe the visual differences.

**Q7.** Looking at the SNR plot, at what approximate timestep does noise start to dominate the signal (SNR < 1)?

**Q8.** In the "different noise realizations" experiment, why do all 5 noisy images look different even though they started from the same original?

**Q9.** Why is the forward diffusion process easy to implement, but reversing it requires training a neural network?

---

## Module 2: Training a Toy Denoiser

**Time:** ~18-22 minutes
**Type:** Colab notebook with training
**Training Time:** 2-3 minutes on T4 GPU

### Learning Objectives
- Build a simplified U-Net architecture
- Understand timestep embeddings
- Train a denoiser to predict noise
- Visualize training progress and loss curves
- Test denoising performance at multiple timesteps
- Compare to CNN training from Lab 7

### What You'll Do

1. Configure NUM_CLASSES (2, 4, 6, or 10 digits)
2. Load and preprocess MNIST data (downsampled to 16×16)
3. Build a simplified U-Net with timestep embeddings
4. Train for 8 epochs using on-the-fly noise addition
5. Visualize training loss curve
6. Test denoising at t=25, 50, 100, 150
7. Visualize learned filters from first conv layer

### Architecture Details

**Simplified U-Net:**
- **Input:** 16×16 noisy image + timestep embedding (32-dim sinusoidal)
- **Encoder:** Conv(32)→Pool→Conv(64)→Pool
- **Bottleneck:** Conv(128)
- **Decoder:** Upsample→Concat→Conv(64)→Upsample→Concat→Conv(32)
- **Output:** Predicted noise (16×16)
- **Total parameters:** ~100,000

**Training Configuration:**
- Loss: MSE (mean squared error)
- Optimizer: Adam (lr=1e-3)
- Epochs: 8
- Batch size: 128
- Target loss: ~0.05

### Connection to Lab 7

| Aspect | Lab 7 (CNN Classifier) | Lab 8 (Denoiser) |
|--------|------------------------|------------------|
| Task | Classify digit (0-9) | Predict noise |
| Loss | Categorical cross-entropy | MSE (mean squared error) |
| Architecture | Encoder only | U-Net (encoder + decoder) |
| Input | Clean image | Noisy image + timestep |
| Output | Class probabilities (10) | Noise prediction (16×16) |
| Training | Gradient descent | Same! |

### Key Insight

> **Training a denoiser uses the same gradient descent process as Lab 7's CNN! The difference is the task: instead of classifying digits, we're predicting noise. The U-Net architecture is perfect for this because it preserves spatial information through skip connections.**

### Questions

**Q10.** Before training, predict: How long will training take with your current NUM_CLASSES setting?

**Q11.** What final loss did you achieve? Is it close to the target of ~0.05?

**Q12.** How long did training take? Was it faster or slower than your prediction from Q10?

**Q13.** At t=100, does the denoiser successfully remove noise? How does the denoised image compare to the original?

**Q14.** Compare denoising performance at t=25 vs t=150. At which timestep does the model perform better? Why?

**Q15.** How is this similar to Lab 7, Module 4 where you trained a CNN on MNIST? How is it different?

---

## Module 3: Multi-Step Denoising (Reverse Diffusion)

**Time:** ~12-15 minutes
**Type:** Colab notebook with generation
**Generation Time:** ~20-30 seconds per image (200 steps)

### Learning Objectives
- Implement the DDPM reverse diffusion algorithm
- Generate images from pure random noise
- Visualize the generation trajectory
- Understand stochasticity in generation
- Compare different step counts (50 vs 100 vs 200)
- Recognize toy model limitations

### What You'll Do

1. Check if model exists from Module 2 (or run quick training)
2. Implement the reverse diffusion function
3. Generate your first image from noise
4. Visualize generation trajectory (10 snapshots)
5. Generate 12 different samples with different seeds
6. Compare quality at 50, 100, and 200 steps
7. Analyze limitations and bridge to Module 4

### The Reverse Diffusion Algorithm

**Symbols:**
```
x_{t-1} = (1/√α_t) · (x_t - ((1-α_t)/√(1-ᾱ_t)) · ε̂) + σ_t · z
```

**Plain English:**
> "Next step = (current - scaled predicted noise) / signal scale + tiny random noise"

**Pseudocode:**
```python
for t in reversed(range(T)):
    predicted_noise = model(x_t, t)
    x_t = (x_t - noise_removal_amount) / signal_scale
    if t > 0:
        x_t = x_t + small_random_noise  # Adds variety
```

### Why Add Noise Back?

- Prevents model from "collapsing" to one image
- Adds stochasticity → variety in generated images
- Amount of noise decreases over time
- Essential for diverse generations!

### Key Insight

> **Reverse diffusion is an iterative refinement process. Structure emerges gradually, not suddenly. More steps generally improve quality, but with diminishing returns. The stochasticity (random noise additions) creates variety in generations—essential for a generative model!**

### Questions

**Q16.** Predict: At approximately what timestep (t value) does the digit start to become recognizable?

**Q17.** Looking at the full trajectory, describe how structure emerges. Does it appear suddenly or gradually?

**Q18.** Looking at the 12 generated samples, what do you notice about variety and quality? Are all digits equally clear?

**Q19.** Compare the 50-step vs 200-step generations. Is the quality difference large or small? Is 200 steps necessary?

**Q20.** Why do you think the generated digits aren't perfect? What limitations does this toy model have?

---

## Module 4: Professional Pre-Trained Model

**Time:** ~12-15 minutes
**Type:** Colab notebook with PyTorch

### Learning Objectives
- Use a professional pre-trained diffusion model
- Understand the framework switch (TensorFlow → PyTorch)
- Compare quality to toy model from Module 3
- Analyze the speed vs quality trade-off (step counts)
- Understand text-to-image conditioning conceptually
- Connect to DALL-E, Stable Diffusion, Midjourney

### What You'll Do

1. Learn about PyTorch (alternative to TensorFlow)
2. Install Hugging Face diffusers library
3. Load pre-trained CIFAR-10 model (30M parameters)
4. Generate 16 images from noise (32×32 RGB)
5. Compare step counts: 25 vs 50 vs 100
6. Compare toy model to professional model
7. Learn how text conditioning works
8. Explore real-world applications

### Model Details

**CIFAR-10 Diffusion Model:**
- **Training data:** 60,000 images, 10 object classes
- **Classes:** airplane, car, bird, cat, deer, dog, frog, horse, ship, truck
- **Resolution:** 32×32 RGB (vs 16×16 grayscale toy model)
- **Parameters:** ~30 million (vs ~100,000 toy model)
- **Training time:** Days/weeks on multiple GPUs (vs 2-3 minutes)
- **Algorithm:** SAME DDPM you implemented!

### Comparison: Toy vs Professional vs DALL-E

| Aspect | Module 3 Toy | Module 4 CIFAR-10 | DALL-E 3 |
|--------|--------------|-------------------|----------|
| Parameters | 100K | 30M | ~10B+ |
| Training Data | 24K digits | 60K objects | Billions of images |
| Resolution | 16×16 gray | 32×32 RGB | 1024×1024+ RGB |
| Training Time | 2-3 min | Days/weeks | Weeks on supercomputers |
| Quality | Blurry digits | Clear objects | Photorealistic |
| Control | Random | Random | Text prompts |

**Same algorithm, increasing scale!**

### How Text Conditioning Works

**Unconditional (CIFAR-10):**
- Input: Random noise
- Output: Random object
- No control

**Conditional (DALL-E):**
- Input: Random noise + text embedding ("a cat in a spacesuit")
- Process: Text guides denoising at each step
- Output: Image matching text description

**Three-Tier Explanation:**

**Plain English:** The model asks "What object am I making?" at each step, using the text as a guide.

**Symbols:** x_{t-1} = Denoise(x_t, t, text_embedding)

**Pseudocode:**
```python
text_embed = encode_text("a red sports car")
for t in reversed(range(T)):
    noise_pred = model(x_t, t, text_embed)  # Text guides prediction
    x_t = denoise_step(x_t, noise_pred)
```

### Key Insight

> **Professional models use the EXACT SAME algorithm you implemented—just at much larger scale. The core diffusion process is identical. Text conditioning is simply adding the text description as an additional input to guide the denoising process. You now understand the technology behind DALL-E!**

### Questions

**Q21.** Look at the 16 generated images. How many distinct object categories can you identify?

**Q22.** Compare the images generated with 25, 50, and 100 steps. Describe the quality differences you observe. Is the improvement from 50→100 steps as noticeable as 25→50 steps?

**Q23.** Why might DALL-E and Stable Diffusion use 20-50 steps instead of 100+ steps? What's the trade-off?

**Q24.** Looking at the comparison table, what specific factors contribute most to the quality difference between your toy model and the professional CIFAR-10 model?

**Q25.** In your own words, explain the complete diffusion pipeline: How does a model like DALL-E go from a text prompt ('a cat in a spacesuit') to a final image? Reference concepts from Modules 0-4.

---

## Key Takeaways

By the end of this lab, you should understand:

- **Diffusion models** generate images by reversing a noise-adding process
- **Forward diffusion** progressively destroys information by adding noise
- **Reverse diffusion** progressively creates images by removing noise
- **Denoisers** are U-Net networks that predict noise at each timestep
- **Training** uses gradient descent and MSE loss (same process as Lab 7!)
- **Noise schedules** control the rate of information destruction
- **Stochasticity** (random noise additions) creates variety in generations
- **More inference steps** generally improve quality (with diminishing returns)
- **Scale matters:** Billions of parameters + massive data = DALL-E quality
- **Text conditioning** guides the denoising process toward specific concepts
- **Real-world applications** include DALL-E, Midjourney, Stable Diffusion, medical imaging, drug discovery

---

## Connection to Real-World AI

### From MNIST Digits to DALL-E: The Scaling Path

Your learning journey mirrors the evolution of diffusion models:

1. **Your Toy Model (Module 3):**
   - 100,000 parameters
   - 24,000 MNIST digits
   - 16×16 grayscale
   - Blurry but recognizable digits

2. **CIFAR-10 Model (Module 4):**
   - 30 million parameters (300× more!)
   - 60,000 diverse objects
   - 32×32 RGB color
   - Clear, recognizable objects

3. **Stable Diffusion:**
   - 890 million parameters (30× more!)
   - 2+ billion text-image pairs
   - 512×512 high resolution
   - Photorealistic images from text

4. **DALL-E 3:**
   - ~10 billion+ parameters
   - Billions of curated images
   - 1024×1024+ resolution
   - Photorealistic, prompt-accurate images

**The progression is clear:** Same algorithm, increasing scale!

### Applications Where Diffusion Models Excel

| Application | How Diffusion Helps |
|-------------|---------------------|
| **Creative Arts** | DALL-E, Midjourney, Stable Diffusion for art, design, advertising |
| **Content Creation** | Adobe Firefly for integrated creative tools |
| **Medical Imaging** | Enhancing MRI/CT scans, generating synthetic training data |
| **Drug Discovery** | Generating novel molecular structures for pharmaceuticals |
| **Video Generation** | Runway, Pika Labs, Sora for text-to-video |
| **3D Modeling** | Generating 3D objects and scenes from text |
| **Audio & Music** | Generating music, sound effects, voice synthesis |
| **Scientific Visualization** | Creating visualizations of complex data |

### The Diffusion Revolution (2020-Present)

**2020:** DDPM paper introduces the algorithm you learned
**2022:** DALL-E 2 and Stable Diffusion launch publicly
**2023:** Billions of images generated, mainstream adoption
**2024:** Video generation (Sora), 3D generation, multimodal models
**Future:** Real-time generation, personalization, professional tools

### Why Diffusion Won

Compared to previous generative methods (GANs, VAEs):
- **Better quality:** More photorealistic, higher resolution
- **Easier training:** More stable, reliable convergence
- **More controllable:** Text conditioning works exceptionally well
- **Versatile:** Works for images, video, audio, 3D, molecules
- **Interpretable:** Iterative process is easier to understand and debug

**You now understand the algorithm powering this revolution!**

---

## Before You Submit

Make sure you have:

- [ ] Completed all 5 modules (0, 1, 2, 3, 4)
- [ ] Run all code cells and viewed visualizations
- [ ] Answered all 25 questions in the answer sheet
- [ ] Made predictions before viewing results (Q4, Q10, Q16)
- [ ] Trained the toy denoiser in Module 2
- [ ] Generated images from noise in Module 3
- [ ] Used the professional CIFAR-10 model in Module 4
- [ ] Understood the forward diffusion process (noise addition)
- [ ] Understood the reverse diffusion process (image generation)
- [ ] Connected concepts to Lab 7 (CNNs)
- [ ] Understood the scaling path to DALL-E

---

## Submission Instructions

Submit your completed answer sheet according to your instructor's guidelines (Canvas upload, PDF, etc.).

**Congratulations!** You've completed Lab 8 and now understand how diffusion models generate images—the foundation of DALL-E, Midjourney, and Stable Diffusion!

---

## Bridge to Future Topics

### What You've Learned About Generative AI:
- Diffusion models **create images** from noise
- **Iterative refinement** produces high quality
- **Text conditioning** enables creative control
- **Scale** dramatically improves results

### What's Coming Next:
- **Advanced diffusion:** Latent diffusion, classifier-free guidance
- **Other generative models:** GANs, VAEs, autoregressive models
- **Multimodal AI:** Combining text, images, and video
- **Practical applications:** Building your own generative tools

### The Connection:
```
Lab 7 (CNNs):      Understand images
Lab 8 (Diffusion): Create images
Future:            Combine understanding + creation for powerful AI systems
```

**Both analysis and synthesis are essential for artificial intelligence!**

---

## Additional Resources

### Further Reading

- **Original DDPM Paper:** Ho et al. (2020) - "Denoising Diffusion Probabilistic Models"
- **Stable Diffusion:** Rombach et al. (2022) - "High-Resolution Image Synthesis with Latent Diffusion Models"
- **DALL-E 2:** Ramesh et al. (2022) - "Hierarchical Text-Conditional Image Generation with CLIP Latents"

### Tools and Platforms

- **Hugging Face Diffusers:** Python library for diffusion models (used in Module 4)
- **Stable Diffusion:** Open-source text-to-image model you can run locally
- **DALL-E 3:** OpenAI's commercial text-to-image system
- **Midjourney:** Artistic image generation platform

### Datasets to Explore

- **CIFAR-10/100:** 60,000 tiny images, 10 or 100 categories (used in Module 4)
- **ImageNet:** 1.4M images, 1000 categories
- **LAION-5B:** 5 billion text-image pairs (used to train Stable Diffusion)

### Interactive Resources

- **Hugging Face Spaces:** Try diffusion models in your browser
- **Colab Notebooks:** Run Stable Diffusion for free
- **GitHub Repositories:** Explore open-source implementations

---

**Questions or feedback?** Reach out to your instructor or TA!

**Happy generating!** 🎨
