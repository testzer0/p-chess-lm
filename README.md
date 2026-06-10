```
├── models/            # Contains code for specific ChessLM architectures
│   ├── base.py            # Base protocol that fits into training and inference code
│   ├── flamingo.py        # Implementation of Flamingo XAttention arch
|   ├── kv_proj.py         # Implementation of KV Projection arch
│   └── llava.py           # Implementation of LLaVa embedding projection arch
|
├── utils/                    # Contains code for supplementary util functions
│   ├── sample_positions.py       # Sample random positions from raw lichess dumps;
|   |                             # stores in (start_fen, move_list, end_fen) triples
|   |
│   ├── generate_sft_data.py      # Creates SFT QA pairs from (s_fen, mv_lst, e_fen) list;
|   |                             # supports multiple question and format types
|   |
│   ├── create_sft_dataset.py     # Create SFT datasetfor train and eval datasets that
|   |                             # can mix question types; saves in HF arrow format
|   |
|   ├── utils.py                  # General utility functions; encoder pass, custom tokens
|   ├── prompt_utils.py           # Defines question templates for QA generation
|   ├── training_utils.py         # Training utility functions for data, model, tokenizer, optim
|   └── eval_utils.py             # Evaluation utility functions for generation, answer checking
|
├── train.py          # Main training loop (details below)
├── scripts/          # Contains scripts for training and inference
└── smoke_tests/      # Contains unit tests for testing code correctness

```
