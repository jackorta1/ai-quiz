# COMP237 - INTRODUCTION TO AI
# MIDTERM STUDY NOTES

**Centennial College | Modules 1-5 Complete Review**

---
---

# MODULE 1: INTRODUCTION TO AI

---

## 1.1 The Definition of AI

There are **4 categories** of AI definitions organized along **2 dimensions**:

|  | **Human** | **Rational** |
|---|-----------|-------------|
| **Thought** | Think Humanly (cognitive modeling) | Think Rationally (laws of thought / logic) |
| **Behavior** | Act Humanly (Turing Test) | Act Rationally (doing the right thing) |

- **2 Dimensions:** Human vs Rational AND Thought vs Behavior
- **Agent** = something that acts and can operate autonomously
- "Act Rationally" = doing the right thing (Nilsson's definition)

---

## 1.2 The Turing Test

| Feature | Detail |
|---------|--------|
| **Proposed by** | Alan Turing (1950) |
| **Main question** | "Can machines think?" |
| **How it works** | Interrogator sends typed messages, must guess if respondent is human or computer |
| **Pass condition** | Machine fools the interrogator 30%+ of the time |
| **Famous example** | ELIZA chatbot |

### Standard Turing Test requires 4 capabilities:
1. **Natural Language Processing (NLP)** — communicate in English
2. **Knowledge Representation** — store what it knows
3. **Automated Reasoning** — answer questions, draw conclusions
4. **Machine Learning** — adapt to new patterns

### Total Turing Test adds 2 more:
5. **Computer Vision** — perceive objects
6. **Robotics** — manipulate objects, move around

---

## 1.3 The 5 AI Disciplines

| # | Discipline | Purpose |
|---|-----------|---------|
| 1 | Natural Language Processing | Communicate in human language |
| 2 | Knowledge Representation & Automated Reasoning | Store knowledge, draw conclusions |
| 3 | Machine Learning | Adapt to new circumstances, detect patterns |
| 4 | Computer Vision | Perceive and interpret images/video |
| 5 | Robotics | Physical manipulation and movement |

---

## 1.4 History of AI

- **1950:** Great expectations, Turing's paper
- **Contributing disciplines:** Philosophy, Mathematics, Psychology, Neuroscience, Control Theory, Computer Engineering, Linguistics
- **Neuroscience contribution:** Understanding how brains process information
- **Logicist tradition:** Building systems using logical notation (obstacle: hard to express informal knowledge formally)

---

## 1.5 AI Applications

- Robotic vehicles
- Game playing (chess, Go)
- Machine translation (100+ languages)
- Recommender systems (Amazon, Netflix, Walmart)
- Spam filtering (a form of dis-recommendation)
- Deep learning analyzes: text, music, videos, history, metadata

---

## 1.6 AI Risks

Major risks of AI:
1. **Biased decision making** (loans, employment, credit cards)
2. **Surveillance and persuasion**
3. **Lethal autonomous weapons**
4. **Economic disruption** (job displacement)
5. **Misuse of ML algorithms**
6. **Bias in training data**

Affected sectors: Economic, social, military, financial, scientific

---
---

# MODULE 2: INTELLIGENT AGENTS

---

## 2.1 Core Definitions

| Term | Definition |
|------|-----------|
| **Agent** | An entity that perceives its environment through **sensors** and acts through **actuators** |
| **Agent function** | Maps any given **percept sequence** to an **action** |
| **Percept** | The agent's perceptual input at any given instant |
| **Percept sequence** | The complete history of everything the agent has perceived |
| **Rational agent** | Selects actions that **maximize expected performance** |

**Key difference: Agent vs Program**
- Agents interact with their environment, perceive and act **autonomously**
- Programs just execute instructions

---

## 2.2 The 5 Agent Types

| Agent Type | How It Works | Key Phrase |
|-----------|-------------|------------|
| **Simple Reflex** | "If condition, then action" — direct mapping from percept to action | No memory, no model |
| **Model-Based Reflex** | Maintains **internal state** of how the world evolves | Handles partial observability |
| **Goal-Based** | Has specific **goals** to achieve, plans actions toward them | "What do I want to achieve?" |
| **Utility-Based** | Evaluates quality of outcomes — sequences may be **quicker, safer, more reliable, or cheaper** | Compares and optimizes |
| **Learning** | Improves from experience, has a **problem generator** element | Any agent type can be a learning agent |

### Important Matching Points:
- "Problem generator element" = **Learning agent**
- "If condition then action" = **Simple Reflex agent**
- "How the world evolves independently" = **Model-based agent**
- "Quicker, safer, more reliable, cheaper" = **Utility-based agent**
- "Partially observed and stochastic environments" = **Model-based agent**
- "Any type of agent can be this type" = **Learning agent**

---

## 2.3 Environments & PEAS Framework

### PEAS = Performance, Environment, Actuators, Sensors

### Environment Properties:

| Property | Option 1 | Option 2 |
|----------|----------|----------|
| Observable | **Fully** (agent sees complete state) | **Partially** (limited/noisy sensors) |
| Deterministic | **Deterministic** (next state fully determined by current state + action) | **Stochastic** (randomness involved) |
| Agents | **Single-agent** | **Multi-agent** (competitive or cooperative) |
| Change | **Static** (doesn't change while agent decides) | **Dynamic** (changes while agent thinks) |
| Values | **Discrete** (finite states/actions) | **Continuous** (infinite range) |

### PEAS Examples:

**Soccer Game:**
| Component | Description |
|-----------|-------------|
| P | Goals scored, wins, passes completed, possession % |
| E | Soccer field, ball, opponents, teammates, weather |
| A | Legs (kick, run), head, body movement |
| S | Eyes (vision), ears (hearing), touch (ball contact) |
| Type | Partially observable, stochastic, multi-agent, dynamic, continuous |

**Autonomous Vehicle:**
| Component | Description |
|-----------|-------------|
| P | Safe arrival, travel time, fuel efficiency, obey traffic laws |
| E | Roads, traffic, pedestrians, weather, signs |
| A | Steering, brakes, accelerator, signals, horn |
| S | Cameras, LIDAR, radar, GPS, speedometer |
| Type | Partially observable, stochastic, multi-agent, dynamic, continuous |

**Chess:**
| Component | Description |
|-----------|-------------|
| P | Win/loss/draw, pieces captured, efficiency |
| E | 8x8 board, chess pieces, opponent, clock |
| A | Move pieces on the board |
| S | Camera/screen to see board state |
| Type | Fully observable, deterministic, multi-agent, sequential, discrete |

**Medical Diagnosis:**
| Component | Description |
|-----------|-------------|
| P | Correct diagnosis %, patient outcomes, cost |
| E | Hospital, patient data, medical records |
| A | Display diagnosis, recommend tests, prescribe |
| S | Keyboard input, symptoms, lab results, history |
| Type | Partially observable, stochastic, single-agent, static, discrete |

---
---

# MODULE 3: UNINFORMED SEARCH

---

## 3.1 Search Problem Components

A search problem is defined by:
1. **Initial state** — where the agent starts
2. **Actions** — what the agent can do
3. **Transition model** — what happens when an action is taken
4. **Goal test** — is this the goal state?
5. **Path cost** — cost of the path from start to current node

**State space** = the set of all possible configurations the system can be in
**Branching factor (b)** = maximum number of successors of any node

---

## 3.2 BFS (Breadth-First Search)

| Property | Value |
|----------|-------|
| **Data structure** | Queue (FIFO) — `pop(0)` |
| **Strategy** | Explore ALL nodes at depth d before depth d+1 |
| **Complete?** | Yes |
| **Optimal?** | Yes (for unweighted graphs / equal step costs) |
| **Time complexity** | O(b^d) |
| **Space complexity** | O(b^d) — stores entire frontier |

### BFS Pseudocode:
```
def BFS(start, goal):
    queue = [start]
    visited = {start}

    while queue is not empty:
        current = queue.pop(0)       # DEQUEUE first element
        if current == goal:
            return success
        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return failure
```

### BFS Step-by-Step:
1. Dequeue the front node
2. Check if it's the goal
3. If NOT goal: **enqueue neighbors and update predecessors**
4. Mark neighbors as visited
5. Repeat

---

## 3.3 DFS (Depth-First Search)

| Property | Value |
|----------|-------|
| **Data structure** | Stack (LIFO) — `pop()` |
| **Strategy** | Go as DEEP as possible before backtracking |
| **Complete?** | No (can get stuck in infinite loops) |
| **Optimal?** | No |
| **Time complexity** | O(b^m) — m = max depth |
| **Space complexity** | O(b*m) — LINEAR! Only stores current path |

### DFS Pseudocode:
```
def DFS(start, goal):
    stack = [start]
    visited = set()

    while stack is not empty:
        current = stack.pop()        # POP last element
        if current == goal:
            return success
        if current not in visited:
            visited.add(current)
            for neighbor in get_neighbors(current):
                if neighbor not in visited:
                    stack.append(neighbor)
    return failure
```

### Key: Neighbor Push Order
- Neighbors pushed onto stack in a **specific order** (check your lab notes!)
- Since stack is LIFO, the **last pushed = first explored**
- If you want to explore Up first, push Up **last**

---

## 3.4 BFS vs DFS Side-by-Side

| Feature | BFS | DFS |
|---------|-----|-----|
| Data structure | Queue (FIFO) | Stack (LIFO) |
| Python operation | `pop(0)` — removes FIRST | `pop()` — removes LAST |
| Explores | Level by level (breadth) | Deep first (depth) |
| Complete | Yes | No |
| Optimal | Yes (unweighted) | No |
| Memory | O(b^d) — HIGH | O(b*m) — LOW |
| Best for | Shortest path (unweighted) | Memory-limited, deep solutions |

---

## 3.5 Uniform Cost Search (UCS)

| Property | Value |
|----------|-------|
| **Data structure** | Priority Queue (sorted by path cost) |
| **Strategy** | Always expand the node with lowest total path cost |
| **Complete?** | Yes |
| **Optimal?** | Yes |
| **When to use** | Edge costs vary, want minimum cost path |
| **Special case** | Reduces to BFS when all step costs are equal |

---

## 3.6 Why We Need a Visited Set

- **Without visited set:** Algorithm revisits nodes endlessly in graphs with cycles = **infinite loop**
- **With visited set:** Each node is explored only once
- This is the difference between **tree search** (no visited check, can revisit) and **graph search** (visited check, no revisits)

---

## 3.7 Complexity Quick Reference

| Metric | BFS | DFS |
|--------|-----|-----|
| Time | O(b^d) | O(b^m) |
| Space | O(b^d) | O(b*m) |

- **b** = branching factor
- **d** = depth of shallowest solution
- **m** = maximum depth of search tree
- Search tree with b=10, d=5: max size = 10^5 = 100,000 nodes

---
---

# MODULE 4: INFORMED SEARCH

---

## 4.1 Key Concepts

| Term | Definition |
|------|-----------|
| **Informed search** | Uses domain-specific knowledge (**heuristics**) to guide search |
| **Uninformed search** | No domain knowledge (BFS, DFS, UCS) |
| **Heuristic h(n)** | An estimate of the cost from node n to the goal |
| **Admissible heuristic** | **Never overestimates** the actual cost: h(n) <= true cost |
| **Consistent heuristic** | h(n) <= cost(n,n') + h(n') for all n,n' — triangle inequality |

Every consistent heuristic is also admissible.

---

## 4.2 Greedy Search

| Property | Value |
|----------|-------|
| **Evaluation function** | **f(n) = h(n)** — heuristic only |
| **Strategy** | Always expand node that APPEARS closest to goal |
| **Complete?** | No (can get stuck in loops) |
| **Optimal?** | No (ignores actual path cost) |
| **Problem** | May pick a path that seems good but is expensive overall |

---

## 4.3 A* Search

| Property | Value |
|----------|-------|
| **Evaluation function** | **f(n) = g(n) + h(n)** |
| **g(n)** | Actual path cost from START to node n |
| **h(n)** | Heuristic estimate from node n to GOAL |
| **Complete?** | Yes (with finite nodes) |
| **Optimal?** | **Yes, IF heuristic is admissible** |
| **Data structure** | Priority Queue (sorted by f-value) |

### A* is optimal because:
It considers BOTH the cost already spent (g) AND the estimated remaining cost (h), ensuring it finds the cheapest total path.

### A* Example:
```
Node A: g=2, h=3  -->  f(A) = 2+3 = 5
Node B: g=4, h=1  -->  f(B) = 4+1 = 5
--> Tie! Either can be expanded first.
```

---

## 4.4 Greedy vs A* Comparison

| Feature | Greedy | A* |
|---------|--------|----|
| Uses g(n)? | No | Yes |
| Uses h(n)? | Yes | Yes |
| Formula | f(n) = h(n) | f(n) = g(n) + h(n) |
| Complete? | No | Yes |
| Optimal? | No | Yes (with admissible h) |
| To convert A* to Greedy | Remove g(n) from f(n) | — |

---

## 4.5 Complete Algorithm Comparison

| Algorithm | Data Structure | Complete? | Optimal? | Time | Space |
|-----------|---------------|-----------|----------|------|-------|
| **BFS** | Queue (FIFO) | Yes | Yes (unweighted) | O(b^d) | O(b^d) |
| **DFS** | Stack (LIFO) | No | No | O(b^m) | O(bm) |
| **UCS** | Priority Queue | Yes | Yes | O(b^d) | O(b^d) |
| **Greedy** | Priority Queue | No | No | O(b^m) | O(b^m) |
| **A*** | Priority Queue | Yes | Yes (admissible h) | O(b^d) | O(b^d) |

---

## 4.6 Heuristic Function Features

A good heuristic should:
1. Be **easy to compute**
2. **Not overestimate** the cost (admissible)
3. **Guide the search efficiently** toward the goal
4. Satisfy the **triangle inequality** (consistent)

---
---

# MODULE 5: MACHINE LEARNING & LINEAR REGRESSION

---

## 5.1 Machine Learning Overview

**Definition:** The ability of a machine to expand its knowledge without human intervention. An agent is learning when it improves its performance after making observations.

**Why ML is needed:**
1. Designers cannot anticipate all possible future situations
2. Designers cannot anticipate all changes over time

**Traditional vs ML:**
| Traditional Approach | Machine Learning Approach |
|---------------------|--------------------------|
| Focuses on **rules** to build logic | Focuses on **previous data** to identify rules/logic |
| Explicit programming | Learns from examples |
| Expert systems, rule-based | Data-driven |

---

## 5.2 Three Types of Learning

| Type | How It Works | Feedback | Example |
|------|-------------|----------|---------|
| **Supervised** | Learns from **labeled input-output pairs**; environment is the "teacher" | Explicit labels | Classification, Regression |
| **Unsupervised** | Finds **patterns without labels**; no explicit feedback | None | Clustering |
| **Reinforcement** | Learns from **rewards and punishments** | Rewards/penalties | Q-learning, Game AI |

### Supervised Learning Detail:
- Given N examples: (x1,y1), (x2,y2),...(xN,yN)
- Find hypothesis h that approximates true function f
- h is drawn from hypothesis space H
- **4 Steps:** Split data -> Train model -> Test model -> Compare results

### Classification vs Regression:
| | Classification | Regression |
|---|---------------|-----------|
| Output | **Categorical/discrete** (sunny, cloudy, rainy) | **Continuous** (33.5, 37.2) |
| Example | Is this email spam? Yes/No | What will the temperature be? |

---

## 5.3 ML Challenges (CONFIRMED EXAM TOPIC)

**Valid challenges:**
- Insufficient quantity of data ✓
- Poor data quality ✓
- Underfitting the data ✓
- Overfitting the data ✓

**NOT valid challenges:**
- Scarcity of algorithms ✗
- Scarcity of performance measures ✗

---

## 5.4 Overfitting vs Underfitting

| | Overfitting | Underfitting |
|---|-----------|-------------|
| **Cause** | Model is too complex | Model is too simple |
| **Bias** | Low | High |
| **Variance** | High | Low |
| **Training performance** | Good | Poor |
| **Test performance** | Poor | Poor |
| **What happens** | Memorizes noise in training data | Can't capture underlying pattern |
| **Fix** | Simplify model, more data, regularization | Use more complex model, more features |

### Bias-Variance Tradeoff:
- **Goal:** Find the model that optimally fits future examples
- Balance between too simple (high bias) and too complex (high variance)
- A linear function often generalizes better than a polynomial that perfectly fits training data

---

## 5.5 Feature Engineering

- **70-80%** of time in ML project is spent on building the feature space
- Involves: Data transformation, standardization/normalization, missing values, correlation analysis, feature selection
- **Entropy and information gain** are used to select the most relevant features

---

## 5.6 Loss Functions

| Name | Formula | Use |
|------|---------|-----|
| **L1 (Absolute)** | \|y - y_hat\| | Regression |
| **L2 (Squared)** | (y - y_hat)^2 | Regression (most common) |
| **L0/1 (Zero-One)** | 0 if y=y_hat, else 1 | Classification |

---

## 5.7 Statistics Review

### Correlation:
- Range: **-1 to +1**
- 0 = no correlation
- Above +0.5 or below -0.5 = notable correlation
- **Positive:** both increase together
- **Negative (inverse):** one increases, other decreases

### Hypothesis Testing:
| Term | Definition |
|------|-----------|
| **Null hypothesis (H0)** | The hypothesis assumed to be true |
| **Alternate hypothesis (Ha)** | The new hypothesis being proposed |
| **P-value** | Probability of finding observed results when H0 is true |
| **Reject H0 when** | p-value < significance level (e.g., < 0.05) |

### T-test vs Z-test:
| | T-test | Z-test |
|---|--------|--------|
| Distribution | Student-t | Normal |
| Population variance | Unknown | Known |
| Sample size | < 30 | > 30 |

### Chi-square test:
- Formula: **chi^2 = sum((O - E)^2 / E)**
- Used for: testing independence, checking fairness, validating data

---

## 5.8 Linear Regression

### Equation:
**y = w1*x + w0**
- w0 = intercept (y-axis)
- w1 = slope

### Finding w0 and w1:
- Use **Ordinary Least Squares (OLS)** method
- Minimize the sum of squared differences: Loss = sum(y - y_hat)^2
- Set partial derivatives to zero: dLoss/dw0 = 0 and dLoss/dw1 = 0

### R-Squared (R^2):
- **R^2 = 1 - (SSE/SST)**
- Measures how well the model explains variability
- Closer to 1 = better fit
- SST = Total Sum of Squares (total variability)
- SSE = Error Sum of Squares (unexplained)
- SSR = Regression Sum of Squares (explained)
- SST = SSR + SSE

### Null Hypothesis in Linear Regression:
- **H0: w1 = 0** (no relationship between x and y)
- **Ha: w1 != 0** (there is a relationship)

### Multivariate Linear Regression:
- h(x) = w0 + w1*x1 + w2*x2 + ... + wn*xn
- Risk: **Overfitting** — irrelevant dimensions may appear useful by chance

---
---

# KEY FORMULAS CHEAT SHEET

| Formula | Name | Usage |
|---------|------|-------|
| f(n) = g(n) + h(n) | A* evaluation | A* search |
| f(n) = h(n) | Greedy evaluation | Greedy search |
| y = w1*x + w0 | Linear regression | Prediction |
| R^2 = 1 - (SSE/SST) | R-squared | Model fit quality |
| L2 = (y - y_hat)^2 | Squared loss | Regression loss |
| L1 = \|y - y_hat\| | Absolute loss | Regression loss |
| Z = (Am - A0) / (sigma/sqrt(n)) | Z-test | Hypothesis testing |
| chi^2 = sum((O-E)^2 / E) | Chi-square | Independence test |
| O(b^d) | BFS complexity | Time & space for BFS |
| O(b*m) | DFS space | Space for DFS |

---
---

# PYTHON CODE CHEAT SHEET

## BFS — Queue (FIFO)
```python
def bfs(start, goal):
    queue = [start]           # Initialize queue
    visited = {start}         # Track visited nodes

    while queue:
        current = queue.pop(0)    # FIFO: remove FIRST element
        if current == goal:
            return True
        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return False
```

## DFS — Stack (LIFO)
```python
def dfs(start, goal):
    stack = [start]           # Initialize stack
    visited = set()           # Track visited nodes

    while stack:
        current = stack.pop()     # LIFO: remove LAST element
        if current == goal:
            return True
        if current not in visited:
            visited.add(current)
            for neighbor in get_neighbors(current):
                if neighbor not in visited:
                    stack.append(neighbor)
    return False
```

## Key Python Difference:
```python
queue.pop(0)   # Removes FIRST element (index 0) --> BFS (FIFO)
stack.pop()    # Removes LAST element            --> DFS (LIFO)
```

## Tracing Code Output:
```python
graph = {'A':['B','C'], 'B':['D','E'], 'C':['F'], 'D':[], 'E':[], 'F':[]}

# BFS output: A, B, C, D, E, F  (level by level)
# DFS output: A, C, F, B, E, D  (deep first, last neighbor popped first)
```

**Why DFS visits C before B:**
Stack after visiting A = [B, C]. `pop()` takes C (last element).

---
---

# LAST-MINUTE REVIEW CHECKLIST

Before the exam, make sure you can:

- [ ] Define "agent function" and fill in the blank
- [ ] Match all 5 agent types to their key phrases
- [ ] Write a complete PEAS description for ANY activity
- [ ] Classify environments (observable, deterministic, agents, dynamic, continuous)
- [ ] Explain why BFS uses queue and DFS uses stack
- [ ] Know `pop(0)` = FIFO (BFS) vs `pop()` = LIFO (DFS)
- [ ] Trace BFS and DFS code to predict output
- [ ] Identify a missing visited set as a bug
- [ ] Write the A* formula: f(n) = g(n) + h(n)
- [ ] Define admissible heuristic (never overestimates)
- [ ] Explain Greedy vs A* (h only vs g+h)
- [ ] Know the complete algorithm comparison table
- [ ] State the 3 types of ML (Supervised, Unsupervised, Reinforcement)
- [ ] Identify ML challenges (NOT: scarcity of algorithms/performance measures)
- [ ] Distinguish overfitting (good train, bad test) from underfitting (bad both)
- [ ] Know traditional approach (rules) vs ML approach (data)
- [ ] Write the linear regression equation: y = w1*x + w0
- [ ] Know L2 loss = (y - y_hat)^2
- [ ] Know null hypothesis in regression: H0: w1 = 0
- [ ] Know when to reject H0: p-value < 0.05
- [ ] Know time/space complexity for BFS and DFS

---

**Good luck on your midterm!**
