# COMP237 - INTRODUCTION TO AI
# MOCK MIDTERM EXAM

**Centennial College | Winter 2026**
**Time Allowed: 60 minutes | Total Marks: 100 points**
**Instructions:** Answer ALL questions. Read carefully before answering.

---

## SECTION A: FILL IN THE BLANK (12 points)

---

**Question 1** (3 pts)

An agent's behavior is described by the ________________ that maps any given **percept sequence** to an **action**.

---

**Question 2** (3 pts)

In A* search, the evaluation function is f(n) = ________ + ________, where the first term is the actual path cost from start to node n, and the second is the heuristic estimate from n to the goal.

---

**Question 3** (3 pts)

A model that performs well on training data but poorly on test data is experiencing ________________. This happens because the model has high ________________ and memorizes noise.

---

**Question 4** (3 pts)

The three main types of machine learning are: ________________, ________________, and ________________ learning.

---

## SECTION B: MATCHING (12 points)

---

**Question 5** (6 pts)

Match the best choice (statement) to the type of agent.
**Note:** Many statements might be linked to one type.

| # | Statement | Your Answer |
|---|-----------|-------------|
| 1 | "Problem generator element" | __________ |
| 2 | "Partially observed and stochastic environments are good candidates for the ___" | __________ |
| 3 | "How the world evolves independently of the agent" | __________ |
| 4 | "If condition then action" | __________ |
| 5 | "Sequences of actions may be quicker, safer, more reliable, or cheaper" | __________ |
| 6 | "Any type of agent can be this type" | __________ |

**Options:** Simple Reflex Agent | Model-Based Agent | Goal-Based Agent | Utility-Based Agent | Learning Agent

---

**Question 6** (6 pts)

Match each search algorithm to its correct data structure and property:

| # | Algorithm | Data Structure | Complete? | Optimal? |
|---|-----------|---------------|-----------|----------|
| 1 | BFS | __________ | __________ | __________ |
| 2 | DFS | __________ | __________ | __________ |
| 3 | A* (admissible h) | __________ | __________ | __________ |

**Data Structure Options:** Queue (FIFO) | Stack (LIFO) | Priority Queue
**Complete/Optimal Options:** Yes | No

---

## SECTION C: MULTIPLE CHOICE (45 points)
**Circle the BEST answer for each question.**

---

**Question 7** (3 pts)

What are the two dimensions on which all AI definitions revolve?

- A) Speed and efficiency
- B) Human versus Rational and Thought versus Behavior
- C) Hardware and Software
- D) Agents and systems

---

**Question 8** (3 pts)

For a machine to pass the **standard** Turing Test, it needs which capabilities?

- A) NLP, Knowledge Representation, Automated Reasoning, Machine Learning
- B) Only NLP and Machine Learning
- C) Computer Vision and Robotics
- D) All 6 AI capabilities

---

**Question 9** (3 pts)

The Turing **Total** Test differs from the standard Turing Test by additionally requiring:

- A) More time for conversation
- B) Computer Vision and Robotics (interaction with physical objects)
- C) Multiple interrogators
- D) Higher accuracy levels

---

**Question 10** (3 pts)

Which of the following is NOT one of the 5 AI disciplines?

- A) Natural Language Processing
- B) Automated Reasoning
- C) Quantum Computing
- D) Machine Learning

---

**Question 11** (3 pts)

An intelligent agent is best defined as:

- A) A robot that can move
- B) A system that perceives its environment through sensors and acts through actuators to achieve goals
- C) A computer program only
- D) A device that learns continuously

---

**Question 12** (3 pts)

After checking if a dequeued cell is the goal in BFS (and it is NOT the goal), what comes next?

- A) Update visited
- B) Dequeue again
- C) Exit algorithm
- D) Enqueue neighbors and update predecessors

---

**Question 13** (3 pts)

What is the order we used to push neighbors onto the stack in DFS?

- A) up, down, left, right
- B) up, right, down, left
- C) left, right, up, down
- D) left, down, right, up

---

**Question 14** (3 pts)

Which search algorithm uses the LEAST memory?

- A) Depth-First Search
- B) Breadth-First Search
- C) A* Search
- D) Uniform Cost Search

---

**Question 15** (3 pts)

Why do we maintain a 'visited' set in graph search algorithms?

- A) To speed up the search
- B) To prevent revisiting nodes and creating infinite loops
- C) To determine the goal state
- D) To calculate the cost function

---

**Question 16** (3 pts)

What is an "admissible heuristic"?

- A) A heuristic that is easy to compute
- B) A heuristic that never overestimates the actual cost to reach the goal
- C) A heuristic that uses machine learning
- D) A heuristic that is always correct

---

**Question 17** (3 pts)

What is the main difference between Greedy search and A* search?

- A) Greedy uses f(n) = h(n) only; A* uses f(n) = g(n) + h(n)
- B) Greedy is always optimal; A* is not
- C) A* uses no heuristic; Greedy does
- D) There is no difference

---

**Question 18** (3 pts)

The main difference between the traditional approach and machine learning is:

- A) Traditional is faster
- B) Traditional focuses on rules to build logic; ML focuses on previous data to identify rules/logic
- C) ML requires more hardware
- D) There is no difference

---

**Question 19** (3 pts)

Mark the VALID challenges of machine learning (select all that apply):

- [ ] Insufficient quantity of data
- [ ] Poor data quality
- [ ] Scarcity of algorithms
- [ ] Underfitting the data
- [ ] Overfitting the data
- [ ] Scarcity of performance measures

---

**Question 20** (3 pts)

The L2 loss function is defined as:

- A) |y - y_hat|
- B) (y - y_hat)^2
- C) 0 if y = y_hat, else 1
- D) log(y - y_hat)

---

**Question 21** (3 pts)

In linear regression, y = w1*x + w0. The null hypothesis H0 is:

- A) w1 = 1
- B) w1 = 0 (no relationship between x and y)
- C) w0 = 0
- D) R^2 = 1

---

## SECTION D: CODE-BASED QUESTIONS (16 points)

---

**Question 22** (4 pts)

What is the difference between `queue.pop(0)` and `stack.pop()` in Python?

- A) No difference
- B) `pop(0)` removes the first element (FIFO); `pop()` removes the last element (LIFO)
- C) `pop(0)` removes the last element; `pop()` removes the first element
- D) Both remove random elements

---

**Question 23** (4 pts)

What is wrong with this BFS implementation?

```python
def bfs_broken(start, goal):
    queue = [start]
    while queue:
        current = queue.pop(0)
        if current == goal:
            return True
        for neighbor in get_neighbors(current):
            queue.append(neighbor)
    return False
```

- A) Nothing is wrong
- B) Missing a visited set — will create infinite loops in graphs with cycles
- C) Should use stack instead of queue
- D) Should pop from the end

---

**Question 24** (4 pts)

What does this code output?

```python
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [], 'E': [], 'F': []
}

def mystery(start):
    result = []
    ds = [start]
    while ds:
        node = ds.pop()
        if node not in result:
            result.append(node)
            ds.extend(graph[node])
    return result

print(mystery('A'))
```

- A) ['A', 'B', 'C', 'D', 'E', 'F']
- B) ['A', 'C', 'F', 'B', 'E', 'D']
- C) ['A', 'B', 'D', 'E', 'C', 'F']
- D) ['F', 'E', 'D', 'C', 'B', 'A']

---

**Question 25** (4 pts)

Complete this BFS implementation by filling in the blanks:

```python
def bfs(start, goal):
    queue = [start]
    visited = {start}

    while queue:
        current = ___________          # BLANK 1
        if current == goal:
            return True
        for neighbor in get_neighbors(current):
            if ___________:            # BLANK 2
                ___________            # BLANK 3
                ___________            # BLANK 4
    return False
```

BLANK 1: __________________
BLANK 2: __________________
BLANK 3: __________________
BLANK 4: __________________

---

## SECTION E: WRITTEN / SHORT ANSWER (15 points)

---

**Question 26** (5 pts)

For the activity of **"Playing a Chess Game"**, give a PEAS description of the task environment and characterize it in terms of the following properties:

1. Performance Measure: _______________________________________________
2. Environment: ______________________________________________________
3. Actuators: ________________________________________________________
4. Sensors: __________________________________________________________
5. Type of environment (fully/partially observed, deterministic/stochastic, single/multi-agent, static/dynamic, discrete/continuous): ___________________________

---

**Question 27** (5 pts)

Explain why DFS uses a **stack** while BFS uses a **queue**. How does the data structure choice affect the order in which nodes are explored? Include the time and space complexity of each algorithm.

---

**Question 28** (5 pts)

Compare and contrast **Greedy Search** and **A* Search**:

1. What evaluation function does each use?
2. Is Greedy search complete? Is it optimal?
3. Is A* search complete? Is it optimal? Under what condition?
4. Give a real-world scenario where A* would be preferred over Greedy.

---
---

# ANSWER KEY

---

## SECTION A: FILL IN THE BLANK

**Q1:** agent function

**Q2:** g(n) + h(n)

**Q3:** overfitting; variance

**Q4:** Supervised, Unsupervised, Reinforcement

---

## SECTION B: MATCHING

**Q5:**
1. Learning agent
2. Model-based agent
3. Model-based agent
4. Simple Reflex agent
5. Utility-based agent
6. Learning agent

**Q6:**
| # | Algorithm | Data Structure | Complete? | Optimal? |
|---|-----------|---------------|-----------|----------|
| 1 | BFS | Queue (FIFO) | Yes | Yes (unweighted) |
| 2 | DFS | Stack (LIFO) | No | No |
| 3 | A* (admissible h) | Priority Queue | Yes | Yes |

---

## SECTION C: MULTIPLE CHOICE

| Q | Answer | Q | Answer |
|---|--------|---|--------|
| 7 | B | 14 | A |
| 8 | A | 15 | B |
| 9 | B | 16 | B |
| 10 | C | 17 | A |
| 11 | B | 18 | B |
| 12 | D | 19 | See below |
| 13 | Check lab notes | 20 | B |
|    |        | 21 | B |

**Q19 Answer:** Check: Insufficient data, Poor data quality, Underfitting, Overfitting.
Do NOT check: Scarcity of algorithms, Scarcity of performance measures.

---

## SECTION D: CODE-BASED

**Q22:** B — `pop(0)` removes the first element (FIFO for BFS); `pop()` removes the last element (LIFO for DFS).

**Q23:** B — Missing a visited set. Without it, the algorithm will revisit nodes endlessly in graphs with cycles, causing an infinite loop.

**Q24:** B — `['A', 'C', 'F', 'B', 'E', 'D']`
This is DFS (uses `pop()` = LIFO).
- Pop A, extend [B, C]
- Pop C (last element), extend [F]
- Pop F (no neighbors)
- Pop B, extend [D, E]
- Pop E (no neighbors)
- Pop D (no neighbors)

**Q25:**
- BLANK 1: `queue.pop(0)`
- BLANK 2: `neighbor not in visited`
- BLANK 3: `visited.add(neighbor)`
- BLANK 4: `queue.append(neighbor)`

---

## SECTION E: WRITTEN / SHORT ANSWER

**Q26 — Chess PEAS:**
1. **P:** Win/loss/draw outcome, number of pieces captured, tournament rating, efficiency of moves
2. **E:** 8x8 chess board, chess pieces (kings, queens, rooks, bishops, knights, pawns), opponent, chess clock
3. **A:** Move pieces on the board (via screen display or robotic arm)
4. **S:** Camera/screen input to see board state, clock sensor for time remaining
5. **Type:** Fully observable, deterministic, multi-agent, sequential (static between turns), discrete

**Q27 — DFS vs BFS:**
- **DFS uses a stack (LIFO):** The most recently discovered node is explored first. This means the algorithm goes as deep as possible along one branch before backtracking. When a node is expanded, its children are pushed onto the stack, and the last child pushed is the first one popped.
- **BFS uses a queue (FIFO):** Nodes are explored in the order they were discovered. This means all nodes at depth d are explored before any node at depth d+1. When a node is expanded, its children are added to the back of the queue.
- **DFS complexity:** Time O(b^m), Space O(b*m) — where b = branching factor, m = max depth. Space is linear because only the current path is stored.
- **BFS complexity:** Time O(b^d), Space O(b^d) — where d = depth of shallowest solution. Space is exponential because the entire frontier level must be stored.

**Q28 — Greedy vs A*:**
1. **Greedy:** f(n) = h(n) — uses only the heuristic estimate to the goal. **A*:** f(n) = g(n) + h(n) — uses actual path cost + heuristic estimate.
2. **Greedy:** Not complete (can get stuck in loops), Not optimal (may find a path that looks good locally but is expensive overall).
3. **A*:** Complete (yes, with finite nodes), Optimal (yes, IF the heuristic is admissible — meaning it never overestimates the actual cost).
4. **Real-world scenario:** GPS navigation — A* is preferred because it considers both the distance already traveled (g) and the estimated remaining distance (h), ensuring the shortest total route. Greedy might pick a road that seems close to the destination but takes a longer detour overall.

---

**END OF EXAM**

**Total: 100 points**
| Section | Points |
|---------|--------|
| A: Fill in the Blank | 12 |
| B: Matching | 12 |
| C: Multiple Choice | 45 |
| D: Code-Based | 16 |
| E: Written/Short Answer | 15 |
| **Total** | **100** |
