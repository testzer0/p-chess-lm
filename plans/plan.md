The goal of this next experiment is the next sanity check on our way to produce a model that is able to play chess and explain its moves. Pasted below is the entire project proposal:

ChessLM Project Proposal
============================
High Level Goal: 
Train a language model to play chess at a high level and be able to explain its reasoning about any positions in natural language.
Relevant Vocabulary:
Position: Refers to a configuration of pieces on the board along with an indication of which player’s turn it is.
Ply / Half-move: Refers to a single turn made by one player.
Move: Refers to a single turn made by one player and the subsequent response made by their opponent.
Line / Variation: Refers to a sequence of moves (does not necessarily need to be a whole number of moves)
Variation Tree: Refers to the tree of possible evaluations rooted at a certain position. The nodes of depth 1 are candidate moves for the player whose turn it is, and subsequent depths correspond to the response by the opponent.
Principle Variation: Refers to the “oracle” best sequence of moves that can be made by each player according to an oracle engine.
Material: Refers to the pieces that each player currently has on the board. Each piece has an approximate value (but can vary depending on the current position).
Pawn: 1 point
Knight: 3 points
Bishop: 3.5 points
Rook: 5 points
Queen: 9 points
FEN: A shorthand notation that encapsulates the current position and relevant information (whose turn to move, legality of castling, number of moves until 50-move rules).
Stockfish / Evaluations: a chess engine that is capable of providing a real-valued evaluation of a position. This real-valued eval in units called centi-pawn, and positive values (+x) indicate that White has an advantage of x-centipawns while negative values (-y) indicate that Black has an advantage of y-centipawns. The centi-pawn evaluation roughly correlates with the material piece value dictionary above (+3 is roughly equivalent to White having an extra bishop on the board).

Metrics: 
Playing: We will use ELO as a metric that we calibrate by playing all models pairwise 10x.
Explaining: Two options that we can continue to iterate on. The challenge is to find a canonical way of evaluating all explanations of positions. In some positions, it may make sense to analyze high level ideas whereas in other positions it may make sense to analyze forced lines. 
Rubric scoring system: given a model’s explanation, how many points does it score based on an extensive rubric? 
Sample rubric:
1. Core Evaluation & Material (Max 2 Points)
[1 Point] The Starting Baseline: Does the text explicitly state the current material balance (e.g., "White is down a full exchange," "Black has an extra pawn")?

[1 Point] The Psychological Reality: Does the text give a realistic assessment of the position's vibe (e.g., acknowledging that a position is "completely winning," "worse but fightable," or "deeply uncomfortable") instead of treating every position as a dry math puzzle?

2. Prophylaxis & The Opponent's Plan (Max 2 Points)
[1 Point] The Threat First: Before pitching its own recommended move, does the text dedicate space to explaining what the opponent wants to do if left to their own devices?

[1 Point] Specific Targets: Does it name the specific squares, files, or weak pawns the opponent is trying to exploit?

3. Alternative Candidate Analysis (Max 2 Points)
[1 Point] Testing "Natural" Moves: Does the text evaluate at least one or two other logical, tempting moves that a human player might instinctively want to make?

[1 Point] The Concrete Refutation: For those alternative moves, does it explain exactly why they fail or fall short, rather than just brushing them off as "suboptimal"?

4. Calculation Mechanics & Safety (Max 2 Points)
[1 Point] The Forcing Line: Does the calculation follow a clear path of forced moves (checks, captures, direct threats) that are easy for a human mind to calculate over the board?

[1 Point] The "In-Between" Check: If the line involves a counter-attack or a "desperado" capture, does the text explicitly account for tactical curveballs (like intermediate checks or counter-threats)?

5. Clear Takeaway & End Goal (Max 2 Points)
[1 Point] The Visual Landing Spot: Does the final calculated line describe what the board actually looks like when the dust settles (e.g., "White ends up in a simplified endgame up two pawns")?

[1 Point] The Actionable Rule: Does the explanation conclude with a practical strategic takeaway or rule of thumb that the player can carry forward into future games?
Tree similarity metrics: a (slightly outdated) idea is that we can compute similarity between the gold variation trees and the variation trees that we can extract from model generated outputs.
Quartet: It is defined as the number of subsets of four leaves that are not related by the same topology in both trees. 
RF: It is defined as (A + B) where A is the number of partitions of data implied by the first tree but not the second tree and B is the number of partitions of data implied by the second tree but not the first tree.

Notes on generating gold variation trees:
Self-play through weak networks
Chess book scraping
Baselines:
Play and explanations produced by GPT-5.5 (high), Gemini-3.1 (Pro, high)
Can play around with prompt optimization to see if playing strength / explanation quality improves dramatically. Previous models saw high variance depending on how the current position was passed to models (i.e. picture of board, FEN
Methodology:
Architecture:
An encoder-decoder framework to tackle this task where:
The encoder is a Transformer model that can commit to a near-optimal action at every state,but neither the action nor the reasoning behind it is expressed in natural language.The encoder weights stay frozen throughout the training process.
The decoder is a pretrained LLM. We aim to feed the hidden states of the encoder by projecting them through (learnable) key and value projections (at each layer) and augmenting the KV cache of the decoder with these projections. Essentially, this provides the decoder with a cached context that we can update with every new position. We call learnable projections the KV bridge. 


Possible alterations:
Currently we are only projecting the last layer hidden states of the encoder to the decoder’s KV cache. We can consider either:
concatenating all encoder hidden state layers along the hidden dimension axis and then projecting the concatenated tensor
staggering the projections (i.e. if there are N encoder layers and L decoder layers, we map the first layer encoder hidden states to the [0, L/N] decoder kv cache layers, the second layer encoder hidden states to the [L/N, 2L/N] decoder kv cache layers, etc.)
Proposed training:

TLDR: A 3 stage approach to train the pretrained LLM:
Midtraining stage where the decoder learns to interpret the projected encoder hidden states.
SFT stage where the decoder is exposed to low quality, gemini annotated positional explanations and (potentially) high quality human annotated positional explanations
RL stage where the model is able to self-improve based on RLHF with a LLM as a judge.
Midtraining Stage
The goal of the midtraining stage is to train the decoder to be able to extract a board representation from the encoder. We will evaluate this stage with the proxy of playing strength. The idea is that the decoder needs to be able to successfully be able to interpret the hidden states that encode the current position. 

Possible strategies (all these are done with SFT):
Train the model to output the principal variation as per the LC0 network.
Train the model to output the best ply as per the LC0 network. ❌
Train the model on QA pairs to interpret the board state (i.e. what piece is on what square) ❌
Train linear probes on the encoder hidden states to identify relevant motifs and features (i.e. passed pawn, weak square, kingside attack). Verbalize motifs in natural language to train the decoder.

Notes:
The proxy of playing strength is not a perfect evaluation. We found that when we trained the model to output the best ply, it achieved very strong playing strength, but subsequent training on top of this checkpoint (SFT stage) was not very good. This is likely due to the fact that the next best ply is encoded very close to the last layer hidden state at the last index (one linear head projection away).
SFT Stage
The goal of the SFT stage is to be able to verbalize a faithful explanation to a predicted move. This move can be suboptimal and we aim to recover playing strength with the subsequent RL stage. We will evaluate this stage using tree similarity metrics between the model generated analysis trees and a held out set of gold analysis trees. We aim to stop when the model explanations are good enough to be able to bootstrap with RL training (empirical evaluations).

Possible strategies:
Prompt gemini-3.1-flash-lite to explain a position given the analysis tree, and train on these positional explanations.
Use natural language autoencoders (haiku3.5, opus4.6, code) to extract model cot from the hidden states. 

RL Stage
The goal of the RL stage is to recover playing strength while maintaining the faithfulness of the explanations to the predicted move. We will evaluate these checkpoints with the two metrics defined above.

Possible strategies:
RL with LLM as a judge with the rubric evaluations above to maintain faithful explanations
RLVR with the best move from LC0 to recover playing strength.
=====================================

Chess is a complex game. The game itself can be decomposed into multiple stages, each requiring different methods of analyzing positions. For example, each chess game can be composed (roughly) into
Opening: rote memorization of existing opening theory.
Middlegame: more complex tactical calculations and long term goal setting to transition into a favorable endgame state
Endgame: fewer pieces on the board, requires ability to convert small advantages to victory through strategic placement of those few pieces.

And even within any of these categories, there are multiple sub-categories each involving many different strategies and analysis modes. Just to illustrate for endgames:
King and Pawn endgames: the simplest type of endgame involving just Kings and Pawns
Rook and Pawn endgames: the most frequent type of endgame involving
(Same color) Bishops and Pawn endgames
(Opposite color) Bishop and Pawn endgames
Etc…

We can greatly simplify our scope by focusing on a subset of the chess game, such as only looking at endgames, or even just a certain type of endgame. This is the task that we will aim to solve in ./chess/. We will start with just King and Pawn endgames and slowly increase complexity to see if we are able to apply the method that worked for Tower of Hanoi (the key exception being that the complexity is greatly increased and the explanations can no longer be programatically generated) to chess endgames. 

Once you have read this document, please create chess_plan.md and write some initial summary plan of this proposal (you can use the start of hanoi_plan.md as template). Do not touch this file, we will continue to write and update chess_plan.md as we setup the data files, load in LC0 weights, and setup SFT and RL training.
