Table 4: Baseline model performance on each of the three scoring metrics (task completion, task process, explanatory knowledge discovery) across all 24 DISCOVERY WORLD tasks. Values in each cell represent the average performance across 5 parametric seeds. Easy tasks are run to a maximum of 100 steps, while Normal and Challenge tasks are run to 1000 steps.

| #   | Topic          | Task                          | ReACT Baseline | Plan+Execute Baseline | Hypothesizer Baseline |
|-----|----------------|-------------------------------|----------------|-----------------------|-----------------------|
|     |                |                               | Pressure       | Completion            | Knowledge             |
|     |                |                               |                |                       |                       |
| 1   | Proteomics     | Clustering                    | 0.87           | 0.20                  | 0.20                  | 0.90                 | 0.40                  | 1.00                  |
| 2   | Chemistry      | Exploring Combinations and Hill Climbing | 0.88           | 0.40                  | 0.40                  | 0.68                 | 0.20                  | 0.93                 | 0.40                  | 0.40                  |
| 3   | Archaeology    | Correlations                  | 0.88           | 0.40                  | 0.40                  | 0.55                 | 0.20                  | 0.93                 | 0.40                  | 0.60                  |
| 4   | Reactor Lab    | Regression                    | 0.87           | 1.00                  | 1.00                  | 0.70                 | 0.60                  | 0.40                 | 0.90                 | 0.40                  | 0.40                  |
| 5   | Plant Nutrients| Uncovering systems of rules   | 0.82           | 0.00                  | 0.00                  | 0.87                 | 0.40                  | 0.00                 | 0.93                 | 0.60                  | 0.40                  |
| 6   | Space Sick     | Single instrument             | 0.90           | 0.40                  | 0.40                  | 0.90                 | 0.40                  | 0.00                 | 0.87                 | 0.00                  | 0.00                  |
| 7   | Archaeology    | Open-ended discovery           | 0.72           | 0.40                  | 0.30                  | 0.74                 | 0.00                  | 0.00                 | 0.64                 | 0.40                  | 0.40                  |
| 8   | Nuclear Lab    | Single instrument             | 0.89           | 0.20                  | 0.00                  | 0.46                 | 0.00                  | 0.05                 | 0.55                 | 0.20                  | 0.05                  |
| 9   | Plant Nutrients| Multiple instruments          | 0.42           | 0.00                  | 0.40                  | 0.44                 | 0.00                  | 0.10                 | 0.38                 | 0.00                  | 0.20                  |
| 10  | Reaction       | Linear regression             | 0.44           | 0.00                  | 0.00                  | 0.49                 | 0.00                  | 0.00                 | 0.51                 | 0.00                  | 0.00                  |
| 11  | Plant Nutrients| Quadratic regression          | 0.43           | 0.00                  | 0.20                  | 0.39                 | 0.00                  | 0.00                 | 0.39                 | 0.00                  | 0.00                  |
| 12  | Plant Nutrients| Novel instruments             | 0.70           | 0.20                  | 0.20                  | 0.70                 | 0.20                  | 0.20                 | 0.60                 | 0.00                  | 0.00                  |
| 13  | Reaction       | Presence rules                | 0.91           | 0.60                  | 0.00                  | 0.84                 | 0.40                  | 0.00                 | 0.56                 | 0.00                  | 0.00                  |
| 14  | Space Sick     | Look-up variables             | 0.33           | 0.00                  | 0.00                  | 0.53                 | 0.00                  | 0.07                 | 0.13                 | 0.40                  | 0.00                  |
| 15  | Challenge      | Measure 2 variables           | 0.51           | 0.00                  | 0.05                  | 0.34                 | 0.00                  | 0.00                 | 0.11                 | 0.00                  | 0.00                  |
| 16  | Challenge      | Measure 5 variables           | 0.43           | 0.00                  | 0.00                  | 0.43                 | 0.00                  | 0.00                 | 0.22                 | 0.00                  | 0.03                  |
| 17  | Translation    | Rosetta stone style linguistic discovery of alien language | 0.40           | 0.40                  | 0.20                  | 0.30                 | 0.00                  | 0.00                 | 0.20                 | 0.20                  | 0.00                  |
| 18  | Translation    | Rosetta stone style linguistic discovery of alien language | 0.20           | 0.00                  | 0.00                  | 0.68                 | 0.40                  | 0.00                 | 0.84                 | 0.40                  | 0.00                  |
| 19  | Translation    | Rosetta stone style linguistic discovery of alien language | 0.06           | 0.00                  | 0.04                  | 0.55                 | 0.20                  | 0.05                 | 0.16                 | 0.00                  | 0.00                  |

Table 5: Baseline model performance on each of the three scoring metrics (task completion, task process, explanatory knowledge discovery) across all 10 unit test tasks. Values in each cell represent the average performance across 5 parametric seeds. Unit tests tasks are run to a maximum of 100 steps.

| #   | Unit Test Topic               | ReACT Baseline | Plan+Execute Baseline | Hypothesizer Baseline |
|-----|-------------------------------|----------------|-----------------------|-----------------------|
|     |                               | Pressure       | Completion            | Knowledge             |
| 25  | Multi-turn dialog with an agent | 1.00           | 1.00                  | 1.00                  | 1.00                  |
| 26  | Measure an object with an instrument | 0.87           | 0.60                  | 0.73                  | 0.40                  | 1.00                  | 1.00                  |
| 27  | Pick-and-place object         | 0.90           | 0.80                  | 0.80                  | 0.60                  | 1.00                  | 1.00                  |
| 28  | Pick-and-give object          | 0.55           | 0.50                  | 0.70                  | 0.80                  | 1.00                  | 1.00                  |
| 29  | Read Discovery Feed posts     | 1.00           | 1.00                  | 0.90                  | 0.80                  | 1.00                  | 1.00                  |
| 30  | Move through doors            | 0.58           | 0.20                  | 0.25                  | 0.00                  | 0.30                  | 0.00                  |
| 31  | Using keys with doors         | 0.69           | 0.20                  | 0.54                  | 0.00                  | 0.69                  | 0.00                  |
| 32  | Navigate to a specific room   | 0.20           | 0.20                  | 0.20                  | 0.00                  | 0.20                  | 0.20                  |
| 33  | Search an environment for an object | 0.80           | 0.80                  | 0.60                  | 0.60                  | 1.00                  | 1.00                  |
| 34  | Interact with a moving agent  | 0.60           | 0.20                  | 0.53                  | 0.00                  | 0.53                  | 0.20                  |

4.2 Baseline Agent Models

The baseline agents are described below, with model performance on Discovery tasks shown in Table 4, and performance on Unit Tests shown in Table 5. We use the GPT-4O model for all our agents due to its higher performance and lower cost compared to other models. For space we provide