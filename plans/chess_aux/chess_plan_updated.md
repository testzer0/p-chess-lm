## Overall Goals

The overall goals are the same as the original project, with the possibility of reduced scope. Fill in some details here from the prior chess\_plan.md overall goals / intro tab.

## Proposed Roadmap

I have done some thinking regarding how we are going to get a decoder LM to start verbalizing its thoughts. I have come to a few conclusions. The LM should start verbalizing explanations very early on. These initial explanations can be very verbose, i.e. an explanation of checkmate can be (broken down into atomic move rules, sans the special tokenization):
    - If I play Qg7 in this position, there will be a white queen on g7 and a black king on g8. The black king can move to h8, h7, g7, f7, f8. However, all these square are being attacked by at least one white piece. The white queen is also attacking the g8 square. Thus, black is in checkmate.

The idea is that the language model should learn (as humans do) to express higher level concepts in very verbose language, before kind of distilling their knowledge into fewer tokens. This is like how when humans take proof based classes, at first each individual step is very small, and then we learn to synthesize and recognize patterns to compress an extremely verbose description into a more compressed format. To this end, we first identify four necessary skills that the model needs to perform well on in order to verbalize reasonable explanations for more complex tasks (identifying checkmate, identifying forks / pins / skewers, etc).

These four tasks are:
    - Stage 1: Current Position Understanding: Conditioned on an encoder position from LC0, identify what piece is on a queried square; identify what square a queried piece is on; identify how many white / black pieces. 
    - Stage 2: Current Position Movement/Attacks: Conditioned on an encoder position, identify how many attackers / defenders exist for a queried square; identify all square a queried piece can move to / attack
    - Stage 3: Future Position understanding: Conditioned on an encoder position from LC0 and a series of moves (but crucially the end position is not encoded with LC0 and provided to the LM), identify what piece is on a queried square; identify what square a queried piece is on; identify how many white / black pieces 
    - Stage 4: Future Position Movement/Attacks: Conditioned on an encoder position and sequence of moves, identify how many attackers / defenders exist for a queried square; identify all square a queried piece can move to / attack.

## Possible Architectures

These three architectures are marked out in kv\_bridge.md. We should do some initial testing on stage 1 to see the architecture that is most expressive / learns quickest with fewest amount of data.

## Data

We have also a few data schemes, constructed in our various dataset folders. Eventually, we will need datasets for stage2, 3, 4 as well. 



