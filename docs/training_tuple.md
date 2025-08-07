# Training Tuple Format

The `convert_v6_to_tuple` function in `tf/chunkparser.py` processes training
data and produces a 5-element tuple:
`(planes, probs, winner, best_q, plies_left)`.

When these raw byte strings are interpreted as NumPy arrays, they have the
following shapes for each training example:

1.  **`planes`**: `(112, 64)` as a `float32` array.
    * This represents the board state as 112 feature planes, each of size 8x8
        (64). The original 104 planes from the input are augmented with 8
        additional planes for information like castling rights, side to move,
        the rule 50 count, and board edge detection.

2.  **`probs`**: `(1858,)` as a `float32` array.
    * This corresponds to the `float probabilities[1858]` member in the
        `V6TrainingData` C++ struct, representing the policy probabilities for
        all possible moves.

3.  **`winner`**: `(3,)` as a `float32` array.
    * This holds the game's outcome from the current player's perspective,
        representing the probabilities for a win, draw, and loss, respectively.

4.  **`best_q`**: `(3,)` as a `float32` array.
    * Similar to `winner`, this stores the value of the position after search
        (the Q-value), also represented as win, draw, and loss probabilities.

5.  **`plies_left`**: A scalar `float32`.
    * This value represents the estimated number of plies remaining until the
        end of the game.
