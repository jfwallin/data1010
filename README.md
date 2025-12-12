# DATA 1010: Artificial Intelligence in Action - Lab Materials

Interactive lab materials for teaching introductory students how Machine Learning and Artificial Intelligence work through hands-on experimentation.

## About This Course

**DATA 1010: Artificial Intelligence in Action** is a freshman-level, non-major course designed for general education science credit. The course provides students with a conceptual understanding of AI/ML through interactive exploration, minimizing mathematical formalism and programming requirements while maximizing hands-on learning.

### Course Topics

Beyond these hands-on labs, students learn about:
- History of AI
- Large Language Models (LLMs)
- Multimodal AI systems
- AI ethics and regulation
- The future of AI

## Philosophy

These labs are designed to:
- **Lower barriers to entry** for students with no CS or math background
- **Build conceptual intuition** before introducing formal models
- **Emphasize exploration and discovery** over coding mechanics
- **Use prediction-experiment-explanation cycles** to develop deep understanding
- **Run entirely in Google Colab** with no local installation required

Each lab is designed for **60-75 minutes** of instructor-led, collaborative group work in a classroom setting.

## Lab Structure

All labs follow a consistent **Predict → Experiment → Explain** framework:

1. **Predict**: Students make hypotheses about what will happen
2. **Experiment**: Students run interactive code and adjust parameters
3. **Explain**: Students articulate their understanding of the observations

Each lab typically includes:
- Pre-lab preparation materials (10-20 min, asynchronous)
- In-lab group activities (60-75 min, synchronous)
- Post-lab reflection questions (20-30 min, asynchronous)

## Getting Started

### For Students

Click the "Open in Colab" button for any lab below. The notebooks will open in Google Colab where you can:
1. **File → Save a copy in Drive** to create your own editable version
2. Run cells by clicking the play button or pressing Shift+Enter
3. Experiment with parameters using interactive sliders and widgets
4. Work collaboratively with your group

**No installation required** - everything runs in your browser!

### For Instructors

All materials are designed to be used in a 75-90 minute class session with students working in groups of 2-4. See [DATA_1010_Lab_Framework_Design_Principles.md](DATA_1010_Lab_Framework_Design_Principles.md) for detailed pedagogical guidance, facilitation strategies, and assessment recommendations.

The notebooks are structured to minimize instructor preparation while maximizing student engagement. All code is pre-written; students focus on exploration and understanding rather than programming.

## Labs

### Lab 1: Error, Loss, and Optimization

**Core Concepts**: Understanding how machines measure mistakes, visualizing error in parameter space, and exploring optimization fundamentals

**Learning Objectives**:
- Understand the difference between local and global error
- Explore how model parameters affect total loss
- Visualize error landscapes in parameter space
- Build intuition for optimization as hill-climbing

**Materials**:
- [Student Handout](lab_1_modules/Lab_1_Student_Handout.md) | [Answer Sheet](lab_1_modules/Lab_1_Answer_Sheet.md)

**Modules**:
- [Module 0: Setup](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_1_modules/lab_1_module_0_setup.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_1_modules/lab_1_module_0_setup.ipynb)
- [Module 1: Global Error](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_1_modules/lab_1_module_1_global_error.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_1_modules/lab_1_module_1_global_error.ipynb)
- [Module 2: Line Fitting](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_1_modules/lab_1_module_2_line_fitting.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_1_modules/lab_1_module_2_line_fitting.ipynb)
- [Module 3: Parameter Space](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_1_modules/lab_1_module_3_parameter_space.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_1_modules/lab_1_module_3_parameter_space.ipynb)
- [Module 4: Hidden Function](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_1_modules/lab_1_module_4_hidden_function.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_1_modules/lab_1_module_4_hidden_function.ipynb)
- [Module 5: Mountain Climbing](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_1_modules/lab_1_module_5_mountain.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_1_modules/lab_1_module_5_mountain.ipynb)
- [Lab 1 Narrative](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_1_modules/lab_1_narrative.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_1_modules/lab_1_narrative.ipynb)

---

### Lab 2: Gradient Descent

**Core Concepts**: Automated optimization using gradients, understanding learning rates, and recognizing optimization challenges

**Learning Objectives**:
- Understand how gradient descent automates optimization
- Explore the role of learning rates in convergence
- Identify failure modes (local minima, divergence)
- Connect gradient descent to real ML training

**Materials**:
- [Student Handout](lab_2_modules/Lab_2_Student_Handout.md) | [Answer Sheet](lab_2_modules/Lab_2_Answer_Sheet.md)

**Modules**:
- [Module 0: Setup](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_2_modules/lab_2_module_0_setup.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_2_modules/lab_2_module_0_setup.ipynb)
- [Module 1: Parabola Gradient Descent](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_2_modules/lab_2_module_1_parabola_gd.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_2_modules/lab_2_module_1_parabola_gd.ipynb)
- [Module 2: Parameter Space Gradient Descent](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_2_modules/lab_2_module_2_parameter_space_gd.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_2_modules/lab_2_module_2_parameter_space_gd.ipynb)
- [Module 3: Learning Rates](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_2_modules/lab_2_module_3_learning_rates.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_2_modules/lab_2_module_3_learning_rates.ipynb)
- [Module 4: Mountain Limits](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_2_modules/lab_2_module_4_mountain_limits.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_2_modules/lab_2_module_4_mountain_limits.ipynb)

---

### Lab 3: Neural Networks and Activation Functions

**Core Concepts**: Moving beyond linear models, understanding activation functions, and building perceptrons

**Learning Objectives**:
- Recognize limitations of linear models
- Understand how activation functions enable nonlinear decisions
- Build intuition for how neurons "bend space" to separate data
- Explore the anatomy of a simple perceptron

**Materials**:
- [Student Handout](lab_3_modules/Lab_3_Student_Handout.md) | [Answer Sheet](lab_3_modules/Lab_3_Answer_Sheet.md)

**Modules**:
- [Module 0: Linear Limits](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_3_modules/lab_3_module_0_linear_limits.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_3_modules/lab_3_module_0_linear_limits.ipynb)
- [Module 1: Bending Space](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_3_modules/lab_3_module_1_bending_space.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_3_modules/lab_3_module_1_bending_space.ipynb)
- [Module 2: Activation Details](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_3_modules/lab_3_module_2_activation_details.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_3_modules/lab_3_module_2_activation_details.ipynb)
- [Module 3: Building a Perceptron](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_3_modules/lab_3_module_3_building_perceptron.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_3_modules/lab_3_module_3_building_perceptron.ipynb)
- [Module 4: Testing Limits](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_3_modules/lab_3_module_4_testing_limits.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_3_modules/lab_3_module_4_testing_limits.ipynb)

---

### Lab 4: Deep Neural Networks and Classification

**Core Concepts**: Multi-layer networks, training neural networks, and applying NNs to real classification problems

**Learning Objectives**:
- Understand how multiple layers create complex decision boundaries
- Explore the anatomy of a small neural network
- Train networks on real datasets (Iris, Breast Cancer)
- Evaluate model performance and recognize overfitting

**Materials**:
- [Student Handout](lab_4_modules/Lab_4_Student_Handout.md) | [Answer Sheet](lab_4_modules/Lab_4_Answer_Sheet.md)

**Modules**:
- [Module 0: Lifting Dimensions](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_4_modules/lab_4_module_0_lifting_dimensions.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_4_modules/lab_4_module_0_lifting_dimensions.ipynb)
- [Module 1: Anatomy of a Tiny Neural Network](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_4_modules/lab_4_module_1_anatomy_tiny_nn.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_4_modules/lab_4_module_1_anatomy_tiny_nn.ipynb)
- [Module 2: Training a Neural Network](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_4_modules/lab_4_module_2_training_neural_network-v3.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_4_modules/lab_4_module_2_training_neural_network-v3.ipynb)
- [Module 3: Iris Classification](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_4_modules/lab_4_module_3_iris_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_4_modules/lab_4_module_3_iris_classification.ipynb)
- [Module 4: Breast Cancer Classification](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_4_modules/lab_4_module_4_breast_cancer_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_4_modules/lab_4_module_4_breast_cancer_classification.ipynb)

---

### Lab 5: Embeddings and Representation Learning

**Core Concepts**: How AI represents text, semantic similarity, and high-dimensional vector spaces

**Learning Objectives**:
- Understand embeddings as numerical representations of meaning
- Explore word and sentence embeddings
- Measure semantic similarity using embeddings
- Connect embeddings to LLM capabilities

**Materials**:
- [Student Handout](lab_5_modules/Lab_5_Student_Handout.md) | [Answer Sheet](lab_5_modules/Lab_5_Answer_Sheet.md)

**Modules**:
- [Module 0: Introduction to Embeddings](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_5_modules/lab_5_module_0_introduce_to_embeddings.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_5_modules/lab_5_module_0_introduce_to_embeddings.ipynb)
- [Module 1: Embedding Words](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_5_modules/lab_5_module_1_embedding_words.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_5_modules/lab_5_module_1_embedding_words.ipynb)
- [Module 2: Embedding Sentences](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_5_modules/lab_5_module_2_embedding_sentences.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_5_modules/lab_5_module_2_embedding_sentences.ipynb)

---

### Lab 6: Saliency Maps and Explainability

**Core Concepts**: Understanding what models learn, visualizing important features, and AI ethics

**Learning Objectives**:
- Understand saliency as a measure of feature importance
- Generate saliency maps for text, images, and tabular data
- Recognize the importance of explainability for trust and ethics
- Discuss limitations of saliency-based explanations

**Materials**:
- [Student Handout](lab_6_modules/Lab_6_Student_Handout.md) | [Answer Sheet](lab_6_modules/Lab_6_Answer_Sheet.md)

**Modules**:
- [Module 0: What is Saliency?](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_6_modules/lab_6_module_0_what_is_saliency.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_6_modules/lab_6_module_0_what_is_saliency.ipynb)
- [Module 1: Text Saliency](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_6_modules/lab_6_module_1_text_saliency.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_6_modules/lab_6_module_1_text_saliency.ipynb)
- [Module 2: Image Saliency](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_6_modules/lab_6_module_2_image_saliency.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_6_modules/lab_6_module_2_image_saliency.ipynb)
- [Module 3: Tabular Saliency](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_6_modules/lab_6_module_3_tabular_saliency.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_6_modules/lab_6_module_3_tabular_saliency.ipynb)
- [Module 4: Ethics and Explainability](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_6_modules/lab_6_module_4_ethics_explainability.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_6_modules/lab_6_module_4_ethics_explainability.ipynb)

---

### Lab 7: Convolutional Neural Networks

**Core Concepts**: How computers "see", convolution operations, and hierarchical feature learning

**Learning Objectives**:
- Understand convolution as pattern detection
- Explore filters and their effects on images
- Visualize feature maps in CNNs
- Recognize how CNNs build hierarchical representations
- Train a simple CNN on MNIST digits

**Materials**:
- [Student Handout](lab_7_modules/Lab_7_Student_Handout.md) | [Answer Sheet](lab_7_modules/Lab_7_Answer_Sheet.md)

**Modules**:
- [Module 0: What is Convolution?](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_7_modules/lab_7_module_0_what_is_convolution.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_7_modules/lab_7_module_0_what_is_convolution.ipynb)
- [Module 1: Filters on Real Images](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_7_modules/lab_7_module_1_filters_real_images.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_7_modules/lab_7_module_1_filters_real_images.ipynb)
- [Module 2: CNN Feature Maps](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_7_modules/lab_7_module_2_cnn_feature_maps.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_7_modules/lab_7_module_2_cnn_feature_maps.ipynb)
- [Module 3: Hierarchical Features](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_7_modules/lab_7_module_3_hierarchical_features.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_7_modules/lab_7_module_3_hierarchical_features.ipynb)
- [Module 4: MNIST Training](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_7_modules/lab_7_module_4_mnist_training.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_7_modules/lab_7_module_4_mnist_training.ipynb)

---

### Lab 8: Diffusion Models — From Noise to Images

**Core Concepts**: Generative AI, diffusion models, and the technology behind DALL-E and Stable Diffusion

**Learning Objectives**:
- Understand forward and reverse diffusion processes
- Implement progressive noise addition to images
- Train a denoising U-Net neural network
- Generate images from pure random noise
- Use professional pre-trained diffusion models
- Connect toy implementations to DALL-E and Stable Diffusion
- Understand text-to-image conditioning

**Materials**:
- [Student Handout](lab_8_modules/Lab_8_Student_Handout.md) | [Answer Sheet](lab_8_modules/Lab_8_Answer_Sheet.md) | [Implementation Guide](lab_8_modules/IMPLEMENTATION_GUIDE.md)

**Modules**:
- [Module 0: What is a Diffusion Model?](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_8_modules/lab_8_module_0_what_is_diffusion.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_8_modules/lab_8_module_0_what_is_diffusion.ipynb)
- [Module 1: Forward Diffusion Demo](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_8_modules/lab_8_module_1_forward_diffusion.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_8_modules/lab_8_module_1_forward_diffusion.ipynb)
- [Module 2: Training a Toy Denoiser](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_8_modules/lab_8_module_2_toy_denoiser.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_8_modules/lab_8_module_2_toy_denoiser.ipynb)
- [Module 3: Multi-Step Denoising](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_8_modules/lab_8_module_3_iterative_denoising.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_8_modules/lab_8_module_3_iterative_denoising.ipynb)
- [Module 4: Professional Pre-Trained Model](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_8_modules/lab_8_module_4_pretrained_sampling.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_8_modules/lab_8_module_4_pretrained_sampling.ipynb)

**Key Connection to Lab 7:**
```
Lab 7 (CNNs):      Image → Features → Classification (Analysis: "What is this?")
Lab 8 (Diffusion): Noise → Features → Image (Synthesis: "Create this!")
```
Both use convolutional architectures—one for understanding, one for creating!

---

### Lab 10: AI Self-Assessment and the Hallucination Boundary

**Core Concepts**: AI overconfidence, hallucinations, reliability assessment, and responsible AI usage

**Learning Objectives**:

- Understand that AI confidence ≠ accuracy
- Recognize patterns of AI overconfidence through empirical testing
- Identify systematic AI weaknesses (citation hallucinations, ambiguous queries, etc.)
- Learn verification strategies for AI-generated information
- Develop critical thinking about when to trust vs. verify AI outputs
- Apply responsible AI usage principles to real-world scenarios

**Materials**:

- [Student Handout](lab_10_modules/Lab_10_Student_Handout.md) | [Answer Sheet](lab_10_modules/Lab_10_Answer_Sheet.md)

**Modules**:

- [Module 0: Setup and Understanding AI Self-Assessment](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_10_modules/lab_10_module_0_setup.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_10_modules/lab_10_module_0_setup.ipynb)
- [Module 1: Collecting AI Self-Predictions](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_10_modules/lab_10_module_1_collect_predictions.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_10_modules/lab_10_module_1_collect_predictions.ipynb)
- [Module 2: Evaluating AI Responses](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_10_modules/lab_10_module_2_evaluate_responses.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_10_modules/lab_10_module_2_evaluate_responses.ipynb)
- [Module 3: Analysis and Visualization](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_10_modules/lab_10_module_3_analysis.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_10_modules/lab_10_module_3_analysis.ipynb)
- [Module 4: Synthesis and Implications](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_10_modules/lab_10_module_4_synthesis.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfwallin/data1010/blob/main/lab_10_modules/lab_10_module_4_synthesis.ipynb)

**Unique Feature:**
This lab uses a **hybrid approach**—students interact with real AI systems (ChatGPT, Claude, Gemini, etc.) via free web interfaces while using Jupyter notebooks for data collection, analysis, and visualization. No API keys required!

**Key Pedagogical Message:**
> **AI models are overconfident and make mistakes. Users need caution, testing, and domain knowledge to identify when AI systems fail.**

---

## Adapting for Your Course

This repository is designed to be modular and adaptable:

- **Use individual modules** rather than complete labs
- **Adjust pacing** based on your class length and student needs
- **Combine modules** from different labs to create custom experiences
- **Add your own content** building on these foundations
- **Modify difficulty** by revealing/hiding code cells or changing parameters

All materials are licensed under [MIT License](LICENSE) to encourage adaptation and reuse.

## Technical Requirements

- Google account (for Colab access)
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Stable internet connection

**No installation, no setup, no package management required.**

## Repository Structure

```
data1010/
├── README.md                                    # This file
├── LICENSE                                      # MIT License
├── DATA_1010_Lab_Framework_Design_Principles.md # Detailed pedagogical framework
├── lab_1_modules/                               # Optimization fundamentals
├── lab_2_modules/                               # Gradient descent
├── lab_3_modules/                               # Neural networks & activation
├── lab_4_modules/                               # Deep networks & classification
├── lab_5_modules/                               # Embeddings
├── lab_6_modules/                               # Saliency & explainability
├── lab_7_modules/                               # Convolutional neural networks
├── lab_8_modules/                               # Diffusion models & generative AI
└── lab_10_modules/                              # AI self-assessment & hallucinations
```

## Contributing

If you use these materials and develop improvements or additional content, please consider contributing back through pull requests or by sharing your adaptations.

## Citation

If you use these materials in your teaching or research, please cite:

```
DATA 1010: Artificial Intelligence in Action - Lab Materials
[Your Name/Institution]
https://github.com/jfwallin/data1010
```

## Contact

For questions, suggestions, or to share how you've used these materials:
- Open an issue on this repository
- [Add your contact information here]

## Acknowledgments

Developed for DATA 1010: Artificial Intelligence in Action
[Add any acknowledgments to contributors, funding sources, or inspiration]

---

**Ready to get started?** Click any of the "Open in Colab" buttons above to begin exploring!
