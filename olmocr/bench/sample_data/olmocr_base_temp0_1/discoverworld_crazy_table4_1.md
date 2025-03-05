Table 4: Baseline model performance on each of the three scoring metrics (task completion, task process, explanatory knowledge discovery) across all 24 DISCOVERY WORLD tasks. Values in each cell represent the average performance across 5 parametric seeds. Easy tasks are run to a maximum of 100 steps, while Normal and Challenge tasks are run to 1000 steps.

| # | Topic       | Task                  | ReACT Procedure | ReACT Completion | Plan+Execute Procedure | Plan+Execute Completion | Hypothesizer Procedure | Hypothesizer Completion |
|---|-------------|-----------------------|-----------------|-----------------|------------------------|------------------------|------------------------|------------------------|
| 1 | Proteomics  | Clustering            | 0.87 0.20 0.20  | 0.80 0.00 0.00  | 0.90 0.40 1.00        |                        |                        |                        |
| 2 | Chemistry   | Exploring Combinations and Hill Climbing | 0.88 0.40 0.60 | 0.55 0.20 0.00  | 0.93 0.40 0.60        |                        |                        |                        |
| 3 | Archaeology | Correlations          | 0.87 1.00 1.00  | 0.70 0.60 0.40  | 0.90 0.00 0.40        |                        |                        |                        |
| 4 | Reactor Lab | Regression            | 0.82 0.00 0.00  | 0.87 0.40 0.00  | 0.93 0.60 0.40        |                        |                        |                        |
| 5 | Space Sick  | Open-ended discovery   | 0.89 0.40 0.00  | 0.73 0.40 0.00  | 0.87 0.00 0.00        |                        |                        |                        |
| 6 | Plant Nutrients | Uncovering systems of rules | 0.80 0.20 0.20  | 0.70 0.20 0.20  | 0.60 0.00 0.00        |                        |                        |                        |
| 7 | Reactor Lab | Regression            | 0.91 0.60 0.00  | 0.84 0.40 0.00  | 0.56 0.00 0.00        |                        |                        |                        |
| 8 | Space Sick  | Open-ended discovery   | 0.89 0.40 0.00  | 0.73 0.40 0.00  | 0.62 0.00 0.00        |                        |                        |                        |
| 9 | Archaeology | Correlations          | 0.87 1.00 1.00  | 0.70 0.60 0.40  | 0.90 0.00 0.40        |                        |                        |                        |
| 10| Reactor Lab | Regression            | 0.82 0.00 0.00  | 0.87 0.40 0.00  | 0.93 0.60 0.40        |                        |                        |                        |
| 11| Space Sick  | Open-ended discovery   | 0.89 0.40 0.00  | 0.73 0.40 0.00  | 0.62 0.00 0.00        |                        |                        |                        |
| 12| Archaeology | Correlations          | 0.87 1.00 1.00  | 0.70 0.60 0.40  | 0.90 0.00 0.40        |                        |                        |                        |
| 13| Reactor Lab | Regression            | 0.91 0.60 0.00  | 0.84 0.40 0.00  | 0.56 0.00 0.00        |                        |                        |                        |
| 14| Space Sick  | Open-ended discovery   | 0.89 0.40 0.00  | 0.73 0.40 0.00  | 0.62 0.00 0.00        |                        |                        |                        |
| 15| Archaeology | Correlations          | 0.87 1.00 1.00  | 0.70 0.60 0.40  | 0.90 0.00 0.40        |                        |                        |                        |
| 16| Reactor Lab | Regression            | 0.91 0.60 0.00  | 0.84 0.40 0.00  | 0.56 0.00 0.00        |                        |                        |                        |
| 17| Space Sick  | Open-ended discovery   | 0.89 0.40 0.00  | 0.73 0.40 0.00  | 0.62 0.00 0.00        |                        |                        |                        |
| 18| Archaeology | Correlations          | 0.87 1.00 1.00  | 0.70 0.60 0.40  | 0.90 0.00 0.40        |                        |                        |                        |
| 19| Reactor Lab | Regression            | 0.91 0.60 0.00  | 0.84 0.40 0.00  | 0.56 0.00 0.00        |                        |                        |                        |
| 20| Space Sick  | Open-ended discovery   | 0.89 0.40 0.00  | 0.73 0.40 0.00  | 0.62 0.00 0.00        |                        |                        |                        |
| 21| Archaeology | Correlations          | 0.87 1.00 1.00  | 0.70 0.60 0.40  | 0.90 0.00 0.40        |                        |                        |                        |
| 22| Reactor Lab | Regression            | 0.91 0.60 0.00  | 0.84 0.40 0.00  | 0.56 0.00 0.00        |                        |                        |                        |
| 23| Space Sick  | Open-ended discovery   | 0.89 0.40 0.00  | 0.73 0.40 0.00  | 0.62 0.00 0.00        |                        |                        |                        |
| 24| Archaeology | Correlations          | 0.87 1.00 1.00  | 0.70 0.60 0.40  | 0.90 0.00 0.40        |                        |                        |                        |

Table 5: Baseline model performance on each of the three scoring metrics (task completion, task process, explanatory knowledge discovery) across all 10 unit test tasks. Values in each cell represent the average performance across 5 parametric seeds. Unit tests tasks are run to a maximum of 100 steps.

| # | Unit Test Topic | ReACT Procedure | ReACT Completion | Plan+Execute Procedure | Plan+Execute Completion | Hypothesizer Procedure | Hypothesizer Completion |
|---|----------------|-----------------|-----------------|------------------------|------------------------|------------------------|------------------------|
| 25| Multi-turn dialog with an agent | 1.00 1.00 | 1.00 1.00 | 1.00 1.00 |                        |                        |                        |
| 26| Measure an object with an instrument | 0.87 0.60 | 0.73 0.40 | 1.00 1.00 |                        |                        |                        |
| 27| Pick-and-place object | 0.90 0.80 | 0.80 0.60 | 1.00 1.00 |                        |                        |                        |
| 28| Read DiscoveryFeed posts | 1.00 1.00 | 0.90 0.80 | 1.00 1.00 |                        |                        |                        |
| 29| Move through doors | 0.58 0.20 | 0.25 0.00 | 0.30 0.00 |                        |                        |                        |
| 30| Using keys with doors | 0.69 0.20 | 0.54 0.00 | 0.69 0.00 |                        |                        |                        |
| 31| Navigate to a specific room in a house | 0.20 0.20 | 0.20 0.00 | 0.20 0.20 |                        |                        |                        |
| 32| Search an environment for an object | 0.80 0.80 | 0.60 0.60 | 1.00 1.00 |                        |                        |                        |
| 33| Interact with a moving agent | 0.60 0.20 | 0.53 0.00 | 0.53 0.20 |                        |                        |                        |
| 34| Average (Unit Tests) | 0.76 0.60 | 0.66 0.44 | 0.77 0.64 |                        |                        |                        |

4.2 Baseline Agent Models

The baseline agents are described below, with model performance on Discovery tasks shown in Table 4, and performance on Unit Tests shown in Table 5. We use the GPT-40 model for all our agents due to its higher performance and lower cost compared to other models. For space we provide