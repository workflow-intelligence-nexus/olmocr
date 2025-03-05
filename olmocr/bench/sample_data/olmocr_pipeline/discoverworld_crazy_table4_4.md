Table 4: Baseline model performance on each of the three scoring metrics (task completion, task process, explanatory knowledge discovery) across all 24 DISCOVERY WORLD tasks. Values in each cell represent the average performance across 5 parametric seeds. Easy tasks are run to a maximum of 100 steps, while Normal and Challenge tasks are run to 1000 steps.

| # | Topic         | Task            | ReACT Procedure | ReACT Completion | ReACT Knowledge | Plan+Execute Procedure | Plan+Execute Completion | Plan+Execute Knowledge | Hypothesizer Procedure | Hypothesizer Completion | Hypothesizer Knowledge |
|---|---------------|-----------------|----------------|----------------|----------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|
| 1 | Proteomics    | Clustering      | 0.87 0.20 0.20 | 0.80 0.00 0.00 | 0.90 0.40 1.00 | 0.88 0.40 0.60         | 0.55 0.20 0.00         | 0.93 0.40 0.40         | 0.90 0.40 1.00         | 0.90 0.40 1.00         |                        |
| 2 | Chemistry     | Exploring       |                |                |                | 0.87 1.00 1.00         | 0.70 0.60 0.40         | 0.90 0.00 0.40         |                        |                        |                        |
| 3 | Archaeology   | Correlations    | 0.82 0.00 0.00 | 0.87 0.40 0.00 | 0.93 0.60 0.40 | 0.90 0.40 0.00         | 0.90 0.40 0.00         | 0.90 0.00 0.40         |                        |                        |                        |
| 4 | Reactor Lab   | Regression      | 0.87 0.20 0.20 | 0.70 0.20 0.20 | 0.60 0.00 0.00 | 0.87 0.20 0.20         | 0.87 0.00 0.40         | 0.60 0.20 0.50         |                        |                        |                        |
| 5 | Space Sick    | Single instrument| 0.72 0.40 0.30 | 0.74 0.00 0.00 | 0.64 0.40 0.40 | 0.72 0.40 0.30         | 0.72 0.00 0.00         | 0.64 0.40 0.40         |                        |                        |                        |
| 6 | Plant Nutrients| Uncovering     |                |                |                | 0.87 0.20 0.20         | 0.87 0.00 0.40         | 0.60 0.00 0.40         |                        |                        |                        |
| 7 | Plant Nutrients| Systems of rules|                |                |                | 0.87 0.20 0.20         | 0.87 0.00 0.40         | 0.60 0.00 0.40         |                        |                        |                        |
| 8 | Space Sick    | Open-ended      |                |                |                | 0.87 0.20 0.20         | 0.87 0.00 0.40         | 0.60 0.00 0.40         |                        |                        |                        |
| 9 | Reactor Lab   | Regression      |                |                |                | 0.87 0.20 0.20         | 0.87 0.00 0.40         | 0.60 0.00 0.40         |                        |                        |                        |
| 10 | Rocket Science| Multi-step      |                |                |                | 0.87 0.20 0.20         | 0.87 0.00 0.40         | 0.60 0.00 0.40         |                        |                        |                        |
| 11 | Rocket Science| Measurements    |                |                |                | 0.87 0.20 0.20         | 0.87 0.00 0.40         | 0.60 0.00 0.40         |                        |                        |                        |
| 12 | Rocket Science| Application     |                |                |                | 0.87 0.20 0.20         | 0.87 0.00 0.40         | 0.60 0.00 0.40         |                        |                        |                        |
| 13 | Translation   | Rosetta-stone   |                |                |                | 0.87 0.20 0.20         | 0.87 0.00 0.40         | 0.60 0.00 0.40         |                        |                        |                        |
| 14 | Translation   | Linguistic      |                |                |                | 0.87 0.20 0.20         | 0.87 0.00 0.40         | 0.60 0.00 0.40         |                        |                        |                        |
| 15 | Translation   | Discovery       |                |                |                | 0.87 0.20 0.20         | 0.87 0.00 0.40         | 0.60 0.00 0.40         |                        |                        |                        |
| 16 | Translation   | Style discovery |                |                |                | 0.87 0.20 0.20         | 0.87 0.00 0.40         | 0.60 0.00 0.40         |                        |                        |                        |
| 17 | Translation   | Alien language  |                |                |                | 0.87 0.20 0.20         | 0.87 0.00 0.40         | 0.60 0.00 0.40         |                        |                        |                        |
| 18 | Average       | Easy            |                |                |                | 0.87 0.20 0.20         | 0.87 0.00 0.40         | 0.60 0.00 0.40         |                        |                        |                        |
| 19 | Average       | Normal          |                |                |                | 0.87 0.20 0.20         | 0.87 0.00 0.40         | 0.60 0.00 0.40         |                        |                        |                        |
| 20 | Average       | Challenge       |                |                |                | 0.87 0.20 0.20         | 0.87 0.00 0.40         | 0.60 0.00 0.40         |                        |                        |                        |

Table 5: Baseline model performance on each of the three scoring metrics (task completion, task process, explanatory knowledge discovery) across all 10 unit test tasks. Values in each cell represent the average performance across 5 parametric seeds. Unit tests tasks are run to a maximum of 100 steps.

| # | Unit Test Topic                  | ReACT Procedure | ReACT Completion | Plan+Execute Procedure | Plan+Execute Completion | Hypothesizer Procedure | Hypothesizer Completion |
|---|----------------------------------|----------------|----------------|------------------------|------------------------|------------------------|------------------------|
| 25 | Multi-turn dialog with an agent  |                |                |                        |                        |                        |                        |
| 26 | Measure an object with an instrument | 0.87 0.60 | 0.73 0.40 | 1.00 1.00              |                        |                        |                        |
| 27 | Pick-and-place object           |                |                |                        |                        |                        |                        |
| 28 | Pick-and-give object            |                |                |                        |                        |                        |                        |
| 29 | Read DiscoveryFeed posts        | 0.87 0.20 | 0.25 0.00 | 0.30 0.00              |                        |                        |                        |
| 30 | Move through doors              |                |                |                        |                        |                        |                        |
| 31 | Using keys with doors           | 0.87 0.20 | 0.54 0.00 | 0.69 0.00              |                        |                        |                        |
| 32 | Navigate to a specific room in a house | 0.87 0.20 | 0.20 0.00 | 0.20 0.20              |                        |                        |                        |
| 33 | Search an environment for an object | 0.87 0.80 | 0.60 0.60 | 0.90 0.10               |                        |                        |                        |
| 34 | Interact with a moving agent    |                |                |                        |                        |                        |                        |
| 35 | Average (Unit Tests)            |                |                |                        |                        |                        |                        |

4.2 Baseline Agent Models

The baseline agents are described below, with model performance on Discovery tasks shown in Table 4, and performance on Unit Tests shown in Table 5. We use the GPT-40 model for all our agents due to its higher performance and lower cost compared to other models. For space we provide...