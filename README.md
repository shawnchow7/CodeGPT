# GPT-like Model Implementation

This project is an implementation of a GPT-like model using PyTorch. The model is designed to generate text based on a given context, leveraging a transformer architecture with self-attention and multi-headed attention mechanisms. This was a personal learning project to explore and understand these fundamental concepts in modern natural language processing.

## Project Overview

The goal of this project was to deepen my understanding of transformer models by implementing a simplified version of a GPT (Generative Pre-trained Transformer) model from scratch. Through this hands-on approach, I explored self-attention, multi-headed attention, and the overall transformer architecture, all while using PyTorch to bring these concepts to life. The model generates text by predicting the next character in a sequence, making it a practical exercise in both theory and application.

## What the Code Does

The code implements a character-level GPT-like model that generates text based on an initial context. Here's a breakdown of its key components:

- **Token and Positional Embeddings**: Converts input characters into dense vectors and adds positional information to preserve the sequence order.
- **Transformer Blocks**: The core of the model, each block consists of:
  - **Multi-Headed Self-Attention**: Allows the model to focus on different parts of the input sequence simultaneously, capturing diverse contextual relationships.
  - **Feed-Forward Neural Network**: Processes the attention output to refine the representation.
  - **Layer Normalization and Dropout**: Stabilizes training and prevents overfitting.
- **Vocabulary Projection**: A final linear layer that maps the model's output to the vocabulary size for character prediction.

The `generate` function uses the trained model to produce new text by sampling from the predicted probability distribution of the next character, iteratively building a sequence up to 5000 characters.

## Code Structure

The code is organized into several classes:

- **`GPT`**: The main model class, orchestrating embeddings, transformer blocks, and the final projection layer.
- **`TransformerBlock`**: Represents a single transformer block with multi-headed self-attention and a feed-forward network.
- **`MultiHeadedSelfAttention`**: Implements the multi-headed attention mechanism, splitting the work across multiple attention heads.
- **`SingleHeadAttention`**: Handles a single attention head, computing keys, queries, and values, and applying a masked softmax.
- **`VanillaNeuralNetwork`**: A simple feed-forward network used within each transformer block.

The `generate` function takes a trained model, an initial context, and parameters like the number of characters to generate, producing text as output.

## Running the Code

To run this project, you need PyTorch installed. Follow these steps:

1. **Install PyTorch**:
   ```bash
   pip install torch
   ```

2. **Run the Script**:
   Ensure the pre-trained weights file (`weights.pt`) is in the same directory as the script. The code loads this model and generates 5000 characters starting from an empty context. You can run it with:
   ```python
   python your_script_name.py
   ```
   The script outputs the generated text to the console.

   *Note*: Adjust `WEIGHT_PATH` in the code if your weights file is located elsewhere.

## Learning Experience

This project was a deep dive into the mechanics of transformer models. By implementing self-attention and multi-headed attention from scratch, I gained a practical understanding of how these mechanisms allow the model to weigh different parts of the input sequence and generate coherent text. Working with PyTorch also helped me appreciate its flexibility and power for building neural networks. This hands-on process solidified my grasp of these concepts and their role in modern AI.

## Dependencies

- **PyTorch**: The deep learning framework used to build and run the model.

Feel free to explore the code, tweak the hyperparameters (e.g., `model_dim`, `num_heads`, `num_blocks`), or use your own training data to see how the model performs!
