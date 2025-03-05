Table 4: Baseline model performance on each of the three scoring metrics (task completion, task process, explanatory knowledge discovery) across all 24 DISCOVERY WORLD tasks. Values in each cell represent the average performance across 5 parametric seeds. Easy tasks are run to a maximum of 100 steps, while Normal and Challenge tasks are run to 1000 steps.

| # | Topic    | Task               | ReACT Pressure | Plan+Execute Pressure | Hypothesizer Pressure | ReACT Completion | Plan+Execute Completion | Hypothesizer Completion |
|---|----------|--------------------|----------------|-----------------------|-----------------------|-------------------|----------------------|------------------------|
| 1 | Proteomics | Clustering        | 0.87 0.20 0.20 | 0.80 0.00 0.00 | 0.90 0.40 1.00 | 0.87 0.20 0.20 | 0.80 0.00 0.00 | 0.90 0.40 1.00 |
| 2 | Chemistry | Exploring Combos   | 0.88 0.40 0.40 | 0.68 0.20 0.00 | 0.93 0.40 0.40 | 0.68 0.20 0.00 | 0.93 0.40 0.40 | 0.68 0.20 0.00 |
| 3 | Chemistry | Hill Climbing     | 0.88 0.40 0.40 | 0.55 0.20 0.00 | 0.93 0.40 0.60 | 0.55 0.20 0.00 | 0.93 0.40 0.60 | 0.55 0.20 0.00 |
| 4 | Archaeology | Correlations | 0.87 1.00 1.00 | 0.70 0.60 0.40 | 0.90 0.00 0.40 | 0.70 0.60 0.40 | 0.90 0.00 0.40 | 0.70 0.60 0.40 |
| 5 | Archaeology | Inferences | 0.82 0.00 0.00 | 0.87 0.40 0.00 | 0.93 0.60 0.40 | 0.87 0.40 0.00 | 0.93 0.60 0.40 | 0.87 0.40 0.00 |
| 6 | Archaeology | Correlations | 0.90 0.40 0.00 | 0.90 0.40 0.00 | 0.97 0.00 0.00 | 0.90 0.40 0.00 | 0.97 0.00 0.00 | 0.90 0.40 0.00 |

Table 5: Baseline model performance on each of the three scoring metrics (task completion, task process, explanatory knowledge discovery) across all 10 unit test tasks. Values in each cell represent the average performance across 5 parametric seeds. Unit tests tasks are run to a maximum of 100 steps.

| # | Unit Test Topic                  | ReACT Pressure | Plan+Execute Pressure | Hypothesizer Pressure | ReACT Completion | Plan+Execute Completion | Hypothesizer Completion |
|---|----------------------------------|----------------|-----------------------|-----------------------|-------------------|----------------------|------------------------|
| 25 | Multi-turn dialog with an agent  | 1.00 1.00      | 1.00 1.00             | 1.00 1.00             | 1.00 1.00         | 1.00 1.00             | 1.00 1.00             |
| 26 | Measure an object with an instrument | 0.87 0.60 | 0.73 0.40             | 1.00 1.00             | 1.00 1.00         | 1.00 1.00             | 1.00 1.00             |
| 27 | Pick-and-place object            | 0.90 0.80      | 0.80 0.60             | 1.00 1.00             | 1.00 1.00         | 1.00 1.00             | 1.00 1.00             |
| 28 | Pick-and-place object            | 0.20 0.00      | 0.20 0.00             | 0.20 0.00             | 0.20 0.00         | 0.20 0.00             | 0.20 0.00             |
| 29 | Read Discovery/Feed posts        | 0.55 0.20      | 0.55 0.20             | 0.55 0.20             | 0.55 0.20         | 0.55 0.20             | 0.55 0.20             |
| 30 | Move through doors               | 0.58 0.20      | 0.25 0.00             | 0.30 0.00             | 0.30 0.00         | 0.30 0.00             | 0.30 0.00             |
| 31 | Using keys with doors            | 0.69 0.20      | 0.54 0.00             | 0.69 0.00             | 0.69 0.00         | 0.69 0.00             | 0.69 0.00             |
| 32 | Navigate to a specific room in a house | 0.20 0.20 | 0.20 0.00             | 0.20 0.20             | 0.20 0.20         | 0.20 0.20             | 0.20 0.20             |
| 33 | Search an environment for an object | 0.80 0.80 | 0.60 0.60             | 0.90 1.00             | 0.90 1.00         | 0.90 1.00             | 0.90 1.00             |
| 34 | Interact with a moving agent     | 0.60 0.20      | 0.53 0.00             | 0.53 0.20             | 0.53 0.20         | 0.53 0.20             | 0.53 0.20             |

4.2 Baseline Agent Models

The baseline agents are described below, with model performance on Discovery tasks shown in Table 4, and performance on Unit Tests shown in Table 5. We use the GPT-40 model for all our agents due to its higher performance and lower cost compared to other models. For space we provide...