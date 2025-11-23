# Neural Network Heads Documentation

This document describes the various policy and value heads used in the network, their training targets, and any specific scaling or transformations applied during training.

## Policy Heads

### 1. Vanilla Policy (`vanilla`)
*   **Description**: The standard policy head predicting the best move.
*   **Training Target**: The `probabilities` vector from the training data (MCTS visit counts).
*   **Scaling/Transformation**: No specific scaling is applied to the target. The loss function compares the network output (logits) directly against the target probability distribution.
*   **Loss Function**: Cross-entropy loss (Kullback-Leibler divergence).

### 2. Optimistic Short-Term Policy (`optimistic_st`)
*   **Description**: A policy head trained to be "optimistic" about the outcome, focusing more on moves that lead to better short-term evaluations.
*   **Training Target**: The same `probabilities` vector as the `vanilla` head.
*   **Scaling/Transformation**:
    *   **Mechanism**: The "optimism" is applied as a **sample weight** in the loss function, not as a transformation of the target tensor itself.
    *   **Calculation**: Weights are calculated based on the difference between the short-term value target (`st`) and the network's predicted short-term value, scaled by the predicted error.
    *   **Logic**: Samples where the network underestimates the short-term value (i.e., the position is better than the network thinks) are weighted higher.
*   **Loss Function**: Weighted Cross-entropy loss.

### 3. Soft Policy (`soft`)
*   **Description**: A policy head trained on a "softened" version of the MCTS probabilities, encouraging exploration or capturing more of the distribution's shape.
*   **Training Target**: The `probabilities` vector from the training data.
*   **Scaling/Transformation**:
    *   **Mechanism**: **Temperature scaling** is applied to the target probabilities **inside the loss function**.
    *   **Calculation**: `target = target^(1/temperature)`.
    *   **Location**: This transformation happens in the loss calculation logic (specifically in `correct_policy` helper in `tfprocess.py`), **not** at the tensor generation stage. The tensor generator provides the raw `probabilities`.
*   **Loss Function**: Cross-entropy loss against the temperature-scaled target.

### 4. Opponent Policy (`opponent`)
*   **Description**: Predicts the move the opponent actually played in the game.
*   **Training Target**: `opp_played_idx` (Integer index of the move played by the opponent).
*   **Scaling/Transformation**: The integer index is converted to a one-hot vector inside the loss function.
*   **Loss Function**: Cross-entropy loss.

## Value Heads

### 1. Winner (`winner`)
*   **Description**: Predicts the final game outcome (Win/Draw/Loss).
*   **Training Target**: A 3-element probability vector derived from `result_q` and `result_d` in the training data.
    *   `Win = (1 + result_q - result_d) / 2`
    *   `Loss = (1 - result_q - result_d) / 2`
    *   `Draw = result_d`
*   **Scaling/Transformation**: None.
*   **Loss Function**: Cross-entropy or MSE depending on configuration.

### 2. Q-Value (`q`)
*   **Description**: Predicts the expected value of the position based on the MCTS search (best Q).
*   **Training Target**: A 3-element probability vector derived from `best_q` and `best_d` in the training data.
    *   `Win = (1 + best_q - best_d) / 2`
    *   `Loss = (1 - best_q - best_d) / 2`
    *   `Draw = best_d`
*   **Scaling/Transformation**: None.
*   **Loss Function**: MSE or Cross-entropy.

### 3. Short-Term Value (`st`)
*   **Description**: Predicts the short-term evaluation of the position (e.g., from a shallow search or static eval).
*   **Training Target**: A 3-element probability vector derived from `q_st` and `d_st` in the training data.
    *   `Win = (1 + q_st - d_st) / 2`
    *   `Loss = (1 - q_st - d_st) / 2`
    *   `Draw = d_st`
*   **Scaling/Transformation**: None.
*   **Loss Function**: MSE or Cross-entropy.

## Auxiliary Heads

### 1. Moves Left (`moves_left`)
*   **Description**: Predicts the number of moves remaining in the game.
*   **Training Target**: `plies_left` (Float).
*   **Scaling/Transformation**:
    *   **Mechanism**: The target and output are scaled down by a factor (e.g., 20.0) **inside the loss function** to bring the loss magnitude into a similar range as other losses.
*   **Loss Function**: Huber loss.
