# COMP237 - CENTENNIAL COLLEGE MIDTERM PREDICTION
## Comprehensive Exam Preparation Guide (Modules 1-5)

---

# PART 1: QUIZ PATTERN ANALYSIS

Based on your last quiz screenshots (16 questions, 30 minutes), the exam uses these question types:
- **Fill-in-the-blank** (e.g., Q1: "agent function" maps percept sequence to action)
- **Matching** (e.g., Q2: match agent characteristics to agent types, 3 pts)
- **Multiple Choice** (e.g., Q6: BFS algorithm steps, Q14: neighbor push ordering)
- **Written/Short Answer** (e.g., Q9: PEAS description for "Playing a Soccer game", 5 pts)
- **Code-based MC** (pseudocode reading, output prediction, fill-in-code)

---

# PART 2: CONFIRMED QUESTIONS (From Studocu Preview)

### Confirmed Q1: Traditional vs Machine Learning
"The main difference between the traditional approach and the machine learning approach is:"

**Answer:** Traditional approach focuses on rules to build the logic, while the machine learning approach focuses on previous data to identify the rules/logic.

### Confirmed Q2: Machine Learning Challenges
"Mark the valid challenges of machine learning from the below list:"
- Insufficient quantity of data ✓
- Poor data quality ✓
- Scarcity of algorithms ✗
- Under fitting the data ✓
- Over fitting the data ✓
- Scarcity of performance measures ✗

---

# PART 3: PREDICTED QUESTIONS BY MODULE

---

## MODULE 1: Introduction to AI

### M1-Q1 (MC): AI Definition Categories
"Which of the following is NOT one of the four AI definition categories?"
- A) Think humanly
- B) Act humanly
- C) Learn humanly
- D) Think rationally

**Answer: C** - The four categories are: Think Humanly, Act Humanly, Think Rationally, Act Rationally.

### M1-Q2 (MC): AI Definition Dimensions
"What are the two dimensions on which all AI definitions revolve?"
- A) Speed and efficiency
- B) Human versus Rational and Thought versus Behavior
- C) Hardware and Software
- D) Agents and systems

**Answer: B**

### M1-Q3 (MC): Turing Test
"What is the main question the Turing Test tries to answer?"
- A) Can machines be faster than humans?
- B) Can machines think?
- C) Can machines see?
- D) Can machines move?

**Answer: B**

### M1-Q4 (MC): Turing Test Capabilities
"For a machine to pass the Turing Test, it needs:"
- A) 1-2 capabilities
- B) All 4 required capabilities (NLP, Knowledge Representation, Automated Reasoning, Machine Learning)
- C) At least 3 capabilities
- D) Just one major capability

**Answer: B**

### M1-Q5 (MC): Turing Test Pass Threshold
"What does it mean if a machine 'passes' the Turing Test?"
- A) It can solve all problems
- B) It fools the interrogator 30% of the time or more
- C) It is smarter than humans
- D) It can learn new languages instantly

**Answer: B**

### M1-Q6 (MC): Total Turing Test
"The Turing Total Test differs from the standard Turing Test by requiring:"
- A) More time for conversation
- B) Interaction with physical objects (video signal and passing objects)
- C) Multiple interrogators
- D) Higher accuracy levels

**Answer: B** - Requires Computer Vision and Robotics in addition to the standard 4 capabilities.

### M1-Q7 (MC): AI Disciplines
"Which of the following is NOT one of the 5 AI disciplines?"
- A) Natural Language Processing
- B) Automated Reasoning
- C) Quantum Computing
- D) Machine Learning

**Answer: C** - The 5 disciplines are: NLP, Automated Reasoning, Machine Learning, Computer Vision, Robotics.

### M1-Q8 (MC): AI Risks
"Which of the following is a major risk of AI development?"
- A) Biased decision making
- B) Surveillance and persuasion
- C) Lethal autonomous weapons
- D) All of the above

**Answer: D**

### M1-Q9 (MC): History of AI
"Which of the following disciplines has contributed to AI development?"
- A) Philosophy and Mathematics
- B) Psychology and Neuroscience
- C) Control Theory and Computer Engineering
- D) All of the above

**Answer: D**

### M1-Q10 (Fill-in-blank):
"An _____ in AI is defined as something that acts and can operate autonomously."

**Answer:** Agent

---

## MODULE 2: Intelligent Agents

### M2-Q1 (Fill-in-the-blank) [HIGH PROBABILITY - Same as Quiz Q1]:
"An agent's behavior is described by the _____ that maps any given percept sequence to an action."

**Answer:** agent function

### M2-Q2 (MC): Agent Definition
"An intelligent agent is best defined as:"
- A) A robot that can move
- B) A system that perceives its environment and acts to achieve goals
- C) A computer program only
- D) A device that learns continuously

**Answer: B**

### M2-Q3 (MC): Agent vs Program
"What is the primary difference between an agent and a program?"
- A) Agents are faster
- B) Agents are more intelligent
- C) Agents interact with their environment and can perceive and act autonomously
- D) There is no difference

**Answer: C**

### M2-Q4 (MC): Agent Interaction
"How does an AI agent interact with its environment?"
- A) Using sensors and actuators
- B) Using only sensors
- C) Using only actuators
- D) None of the above

**Answer: A** - An AI agent perceives using Sensors and acts using Actuators.

### M2-Q5 (MC): Rational Agent
"Which of the following best describes a rational agent?"
- A) Makes random decisions to explore the environment
- B) Attempts to maximize expected performance
- C) Always succeeds in achieving its goals
- D) Requires no information about the environment

**Answer: B**

### M2-Q6 (MC): Agent Types
"Which of the following is a type of artificial intelligence agent?"
- A) Learning AI Agent
- B) Simple Reflex AI Agent
- C) Goal-Based AI Agent
- D) All of the above

**Answer: D** - Five agent types exist: Simple Reflex, Model-Based Reflex, Goal-Based, Utility-Based, Learning.

### M2-Q7 (Matching) [HIGH PROBABILITY - Same as Quiz Q2]:
Match the best choice (statement) to the type of agent:

| # | Statement | Answer |
|---|-----------|--------|
| 1 | "Problem generator element" | Learning agent |
| 2 | "Partially observed and stochastic environments are good candidates for the___" | Model-based agent |
| 3 | "How the world evolves independently of the agent" | Model-based agent |
| 4 | "If condition then action" | Reflex agent |
| 5 | "Sequences of actions may be quicker, safer, more reliable, or cheaper than others" | Utility-based agent |
| 6 | "Any type of agent can be this type" | Learning agent |

### M2-Q8 (Matching - Alternate Version):
| # | Statement | Answer |
|---|-----------|--------|
| 1 | "Uses past performance feedback to improve future decisions" | Learning agent |
| 2 | "Maintains an internal representation of how the world changes" | Model-based agent |
| 3 | "Generates multiple solutions and evaluates them for cost/quality" | Utility-based agent |
| 4 | "Responds immediately to percepts with hard-coded rules" | Simple Reflex agent |
| 5 | "Works towards a specific objective" | Goal-based agent |

### M2-Q9 (Fill-in-blank):
"A _____ agent maintains an internal model of the world to handle partially observable environments."

**Answer:** model-based

### M2-Q10 (MC): Environment Types
"A chess game environment is best described as:"
- A) Fully observable, deterministic, multi-agent, static
- B) Partially observable, stochastic, single-agent, dynamic
- C) Fully observable, stochastic, multi-agent, dynamic
- D) Partially observable, deterministic, single-agent, static

**Answer: A** - Chess is fully observable (you see the whole board), deterministic (moves have predictable outcomes), multi-agent (two players), and static (doesn't change while you think... though some argue sequential).

### M2-Q11 (Written - PEAS) [HIGH PROBABILITY - Same as Quiz Q9]:
"For the activity of 'Playing a Soccer game', give a PEAS description:"

**Sample Answer:**
1. **P (Performance Measure):** Goals scored, wins, passes completed, possession %
2. **E (Environment):** Soccer field, ball, opposing team, teammates, weather
3. **A (Actuators):** Legs (kick, run), head, body movement
4. **S (Sensors):** Eyes (vision), ears (hearing referee/teammates), touch (ball contact)
5. **Type:** Partially observable, stochastic, multi-agent, dynamic, continuous

### M2-Q12 (Written - PEAS Alternate Scenarios):
Be prepared for PEAS on ANY of these:

**Autonomous Vehicle:**
- P: Safe arrival, time, fuel efficiency, passenger comfort
- E: Roads, traffic, pedestrians, weather, signs
- A: Steering, brakes, accelerator, signals, horn
- S: Cameras, LIDAR, GPS, speedometer, radar
- Type: Partially observable, stochastic, multi-agent, dynamic, continuous

**Medical Diagnosis System:**
- P: Correct diagnosis %, patient outcomes, cost
- E: Hospital, patient data, medical records
- A: Display diagnosis, recommend tests, prescribe treatment
- S: Keyboard input, patient symptoms, lab results, medical history
- Type: Partially observable, stochastic, single-agent, static, discrete

---

## MODULE 3: Uninformed Search

### M3-Q1 (MC): State Space Definition
"In search problems, what is the 'state space'?"
- A) The physical space where the agent operates
- B) The set of all possible states the system can be in
- C) Only the starting state
- D) Only the goal state

**Answer: B**

### M3-Q2 (MC): Uninformed Search
"Which of the following is an uninformed search strategy?"
- A) Best-first search
- B) A* search
- C) Breadth-first search (BFS)
- D) Greedy search

**Answer: C** - Uninformed = BFS, DFS, UCS, DLS, Iterative Deepening. Informed = Greedy, A*.

### M3-Q3 (MC) [HIGH PROBABILITY - Same as Quiz Q6]:
"After checking if a dequeued cell is the goal, what comes next in the BFS algorithm, assuming it isn't the goal?"
- A) Update visited
- B) Dequeue again
- C) Exit algorithm
- D) Enqueue neighbors and update predecessors

**Answer: D**

### M3-Q4 (MC): DFS Exploration
"Depth-first search (DFS) explores:"
- A) All nodes at the current depth before going deeper
- B) The deepest path first
- C) Only the shortest path
- D) All paths simultaneously

**Answer: B**

### M3-Q5 (MC): BFS Advantage
"What is the main advantage of Breadth-first search?"
- A) It's faster than DFS
- B) It guarantees finding the shortest path (in unweighted graphs)
- C) It uses less memory
- D) It's easier to implement

**Answer: B**

### M3-Q6 (MC): Data Structures
"What data structure is used in Depth-First Search?"
- A) Queue (FIFO)
- B) Stack (LIFO)
- C) Priority Queue
- D) Linked List

**Answer: B**

"What data structure is used in Breadth-First Search?"
- A) Stack (LIFO)
- B) Queue (FIFO)
- C) Priority Queue
- D) Array

**Answer: B**

### M3-Q7 (MC) [HIGH PROBABILITY - Same as Quiz Q14]:
"What is the order we used to push neighbors on to the stack?"
- A) up, down, left, right
- B) up, right, down, left
- C) left, right, up, down
- D) left, down, right, up

**Answer:** Depends on your course convention - review your lab tutorial notes. Typically the reverse order of how you want to explore them (since stack is LIFO).

### M3-Q8 (MC): Memory Usage
"Which search algorithm uses the LEAST memory?"
- A) DFS
- B) BFS
- C) Both use the same
- D) Cannot be compared

**Answer: A** - DFS only stores nodes on the current path. BFS stores the entire frontier level.

### M3-Q9 (MC): Visited Set
"Why do we maintain a 'visited' set/list in graph search algorithms?"
- A) To speed up the search
- B) To prevent revisiting nodes and creating infinite loops
- C) To determine the goal state
- D) To calculate the cost function

**Answer: B**

### M3-Q10 (MC): Algorithm Comparison
"Which search algorithm uses FIFO queue ordering?"
- A) Depth-First Search
- B) Breadth-First Search
- C) Depth-Limited Search
- D) Iterative Deepening

**Answer: B**

### M3-Q11 (Fill-in-blank): Complexity
"For BFS with branching factor b and maximum depth d:"
- Time complexity: **O(b^d)**
- Space complexity: **O(b^d)**

"For DFS with branching factor b and maximum depth m:"
- Time complexity: **O(b^m)**
- Space complexity: **O(b*m)** (linear!)

### M3-Q12 (MC): UCS
"Uniform Cost Search (UCS) is appropriate when:"
- A) All edge costs are equal
- B) Edge costs vary, and we want the minimum cost path
- C) We only care about the number of steps
- D) Memory is unlimited

**Answer: B**

### M3-Q13 (MC): Search Tree Size
"Given a state space with 10 possible actions per state and a maximum depth of 5, what is the maximum size of the search tree?"
- A) 10^5
- B) 5^10
- C) 10 x 5
- D) 10 + 5

**Answer: A** (100,000 nodes)

### M3-Q14 (Conceptual):
"Explain why DFS uses a stack while BFS uses a queue. How does this data structure choice affect the order in which nodes are explored?"

**Answer:** DFS uses a stack (LIFO) because it needs to explore the most recently discovered node first, going deep before backtracking. BFS uses a queue (FIFO) because it explores nodes in the order they were discovered, processing all nodes at depth d before moving to depth d+1. Stack = depth-first exploration. Queue = breadth-first (level-by-level) exploration.

---

## MODULE 4: Informed Search [NEW - WAS MISSING FROM PREDICTIONS]

### M4-Q1 (MC): Heuristic Definition
"What is a heuristic in AI search?"
- A) A guarantee to find the optimal solution
- B) An estimate that guides search toward the goal
- C) A random search method
- D) A type of algorithm that learns

**Answer: B**

### M4-Q2 (MC): Greedy Search
"Greedy search always:"
- A) Finds the optimal solution
- B) Uses the least memory
- C) Selects the path with the best heuristic value at each step
- D) Explores all paths

**Answer: C** - Greedy search uses only h(n) (heuristic estimate to goal). It is NOT optimal and NOT complete.

### M4-Q3 (MC): A* Search Formula
"A* search combines:"
- A) The actual path cost g(n) and the heuristic estimate h(n)
- B) Depth and breadth
- C) Two different heuristics
- D) Random selection and deterministic search

**Answer: A** - f(n) = g(n) + h(n), where g(n) = cost from start to n, h(n) = estimated cost from n to goal.

### M4-Q4 (MC): A* Optimality
"What makes A* search optimal?"
- A) It always finds the shortest path
- B) It uses admissible heuristics (never overestimate) and combines actual cost with estimated cost
- C) It's the fastest algorithm
- D) It uses the least memory

**Answer: B**

### M4-Q5 (MC): Admissible Heuristic
"What is an 'admissible heuristic'?"
- A) A heuristic that is easy to compute
- B) A heuristic that never overestimates the actual cost to reach the goal
- C) A heuristic that uses machine learning
- D) A heuristic that is always correct

**Answer: B** - h(n) <= actual cost from n to goal, for all n.

### M4-Q6 (MC): Optimal Algorithm
"Which search algorithm guarantees finding the shortest path with minimum cost?"
- A) Greedy search
- B) Depth-first search
- C) A* search with an admissible heuristic
- D) Breadth-first search

**Answer: C**

### M4-Q7 (MC): Heuristic Function Features
"The 'features of a heuristic function' include:"
- A) It should be easy to compute
- B) It should not overestimate the cost
- C) It should guide the search efficiently
- D) All of the above

**Answer: D**

### M4-Q8 (MC): Informed vs Uninformed
"What is the key difference between informed and uninformed search?"
- A) Informed search uses more memory
- B) Informed search uses domain-specific knowledge (heuristics) to guide the search
- C) Uninformed search is always faster
- D) They are the same thing

**Answer: B**

### M4-Q9 (MC): Greedy vs A*
"What is the main difference between Greedy search and A* search?"
- A) Greedy uses f(n) = h(n) only; A* uses f(n) = g(n) + h(n)
- B) Greedy is always optimal; A* is not
- C) A* uses no heuristic; Greedy does
- D) There is no difference

**Answer: A** - Greedy only looks at estimated distance to goal. A* considers both the cost so far AND the estimated remaining cost.

### M4-Q10 (Fill-in-blank):
"In A* search, f(n) = _____ + _____, where the first term is the path cost from start to n and the second is the heuristic estimate from n to the goal."

**Answer:** g(n) + h(n)

### M4-Q11 (MC): Algorithm Comparison Table
| Property | BFS | DFS | Greedy | A* |
|----------|-----|-----|--------|-----|
| Complete? | Yes | No (can loop) | No | Yes (with admissible h) |
| Optimal? | Yes (unweighted) | No | No | Yes (with admissible h) |
| Time | O(b^d) | O(b^m) | O(b^m) | O(b^d) |
| Space | O(b^d) | O(bm) | O(b^m) | O(b^d) |
| Uses heuristic? | No | No | Yes h(n) | Yes g(n)+h(n) |

### M4-Q12 (MC): A* Computation
"Given: Start node S, Goal node G. Path S->A has g=2, h(A)=3. Path S->B has g=4, h(B)=1. Which node does A* expand first?"
- A) Node A because f(A) = 2+3 = 5
- B) Node B because f(B) = 4+1 = 5
- C) Both have equal f-value, either can be expanded
- D) Neither, the algorithm terminates

**Answer: C** - Both have f=5, so ties are broken arbitrarily.

### M4-Q13 (MC): Consistency
"A heuristic is consistent (monotone) if for every node n and successor n':"
- A) h(n) <= cost(n, n') + h(n')
- B) h(n) >= cost(n, n') + h(n')
- C) h(n) = h(n')
- D) h(n) = 0

**Answer: A** - This is the triangle inequality. Every consistent heuristic is also admissible.

---

## MODULE 5: Machine Learning & Linear Regression

### M5-Q1 (MC): ML Definition
"Machine learning is best described as:"
- A) Programming a computer with explicit rules
- B) The ability of a machine to expand its knowledge without human intervention
- C) A type of database management
- D) A hardware improvement technique

**Answer: B**

### M5-Q2 (MC) [CONFIRMED]: Traditional vs ML
"The main difference between the traditional approach and the machine learning approach is:"
- A) Traditional is faster
- B) Traditional focuses on rules to build logic; ML focuses on previous data to identify rules/logic
- C) ML requires more hardware
- D) There is no difference

**Answer: B**

### M5-Q3 (MC): Three Types of Learning
"What are the three main types of machine learning?"
- A) Supervised, Unsupervised, and Reinforcement
- B) Classification, Regression, and Clustering
- C) Linear, Non-linear, and Neural
- D) Training, Testing, and Validation

**Answer: A**

### M5-Q4 (MC): Supervised Learning
"In supervised learning, what does the agent observe?"
- A) Random data without labels
- B) Input-output pairs, learning a function that maps input to output
- C) Only rewards and punishments
- D) Patterns without guidance

**Answer: B** - The environment acts as the "teacher" providing labeled examples.

### M5-Q5 (MC): Unsupervised Learning
"The most common unsupervised learning task is:"
- A) Classification
- B) Regression
- C) Clustering
- D) Reinforcement

**Answer: C** - Detecting useful clusters of input examples without explicit feedback.

### M5-Q6 (MC): Reinforcement Learning
"In reinforcement learning, the agent learns from:"
- A) Labeled examples
- B) A series of rewards and punishments
- C) Explicit rules
- D) Unsupervised patterns

**Answer: B**

### M5-Q7 (MC): Classification vs Regression
"If the predicted output is categorical (e.g., sunny, cloudy, rainy), this is a _____ problem. If continuous (e.g., 33.5 degrees), it's a _____ problem."

**Answer:** Classification; Regression

### M5-Q8 (Checkbox) [CONFIRMED]: ML Challenges
"Mark the valid challenges of machine learning:"
- Insufficient quantity of data ✓
- Poor data quality ✓
- Scarcity of algorithms ✗
- Under fitting the data ✓
- Over fitting the data ✓
- Scarcity of performance measures ✗

### M5-Q9 (MC): Overfitting
"A model that performs well on training data but poorly on test data is experiencing:"
- A) Underfitting
- B) Overfitting
- C) Good generalization
- D) Random error

**Answer: B** - High variance, the model memorizes noise in training data.

### M5-Q10 (MC): Underfitting
"A model that has poor performance on BOTH training and test data is experiencing:"
- A) Overfitting
- B) Underfitting
- C) Perfect fit
- D) Bias-variance balance

**Answer: B** - High bias, the model is too simple to capture patterns.

### M5-Q11 (MC): Bias-Variance Tradeoff
"The goal of the bias-variance tradeoff is to:"
- A) Always minimize bias
- B) Always minimize variance
- C) Find the model that optimally fits future examples, balancing simplicity and complexity
- D) Use the most complex model possible

**Answer: C**

### M5-Q12 (MC): Feature Engineering
"What percentage of time in an ML project is typically spent on building the feature space?"
- A) 10-20%
- B) 30-40%
- C) 50-60%
- D) 70-80%

**Answer: D**

### M5-Q13 (MC): Loss Functions
"The L2 loss function is defined as:"
- A) |y - y_hat|
- B) (y - y_hat)^2
- C) 0 if y = y_hat, else 1
- D) log(y - y_hat)

**Answer: B** - L1 = absolute, L2 = squared, L0/1 = zero-one loss.

### M5-Q14 (MC): Hypothesis
"In supervised learning, the function h that approximates the true function f is called:"
- A) The loss function
- B) The hypothesis
- C) The training set
- D) The test set

**Answer: B** - Drawn from a hypothesis space H.

### M5-Q15 (MC): Steps of Supervised Learning
"What are the 4 main steps of supervised learning?"
1. Split data into training and testing
2. Use training data to learn parameters/weights of h
3. Run the model on test data (without labels)
4. Compare predicted results with actual test labels

### M5-Q16 (MC): Linear Regression
"The uni-variate linear regression equation is:"
- A) y = w1*x + w0
- B) y = x^2 + w0
- C) y = e^x + w0
- D) y = log(x) + w0

**Answer: A** - w0 is the intercept, w1 is the slope.

### M5-Q17 (MC): R-Squared
"R-squared (R^2) measures:"
- A) The total error
- B) How well the regression model explains the variability in data
- C) The number of features
- D) The sample size

**Answer: B** - R^2 = 1 - (SSR/SST). Closer to 1 = better fit.

### M5-Q18 (MC): Correlation
"A correlation coefficient of -0.85 indicates:"
- A) No relationship
- B) A strong positive relationship
- C) A strong negative (inverse) relationship
- D) A weak relationship

**Answer: C** - Values below -0.5 or above 0.5 indicate notable correlation.

### M5-Q19 (MC): Null Hypothesis in Regression
"The null hypothesis in linear regression is:"
- A) H0: w1 = 1
- B) H0: w1 = 0 (no relationship between x and y)
- C) H0: w0 = 0
- D) H0: R^2 = 1

**Answer: B**

### M5-Q20 (MC): P-Value
"When do we reject the null hypothesis?"
- A) When p-value > 0.05
- B) When p-value < significance level (e.g., < 0.05)
- C) When p-value = 1
- D) Never

**Answer: B**

---

# PART 4: CODE-BASED MULTIPLE CHOICE QUESTIONS

These are the types of code questions likely on the midterm, based on your lab tutorials.

---

### CODE-Q1: BFS Pseudocode Fill-in [HIGH PROBABILITY]
"Fill in the missing steps in this BFS pseudocode:"
```python
def BFS(start, goal):
    queue = []
    queue.append(start)
    visited = set()
    visited.add(start)

    while len(queue) > 0:
        current = queue.pop(0)          # ← What does pop(0) do?
        if current == goal:
            return True                  # ← Found the goal
        for neighbor in get_neighbors(current):
            if _____________:            # ← BLANK 1
                visited.add(neighbor)    # ← BLANK 2 purpose?
                queue.append(neighbor)   # ← BLANK 3 purpose?
```

**BLANK 1 Answer:** `neighbor not in visited`
**BLANK 2 Purpose:** Mark neighbor as visited to prevent revisiting
**BLANK 3 Purpose:** Add neighbor to queue for future exploration

### CODE-Q2: DFS Pseudocode Fill-in [HIGH PROBABILITY]
"Fill in the missing steps in this DFS pseudocode:"
```python
def DFS(start, goal):
    stack = []
    stack.append(start)
    visited = set()

    while len(stack) > 0:
        current = stack.pop()           # ← What does pop() do? (removes LAST element)
        if current == goal:
            return True
        if current not in visited:
            visited.add(current)
            for neighbor in get_neighbors(current):
                if _____________:        # ← BLANK
                    stack.append(neighbor)
```

**BLANK Answer:** `neighbor not in visited`
**Key difference from BFS:** `stack.pop()` removes the LAST element (LIFO), while `queue.pop(0)` removes the FIRST element (FIFO).

### CODE-Q3: What Does This Code Output?
```python
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [], 'E': [], 'F': []
}

def bfs(start):
    visited = []
    queue = [start]
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.append(node)
            queue.extend(graph[node])
    return visited

print(bfs('A'))
```
- A) ['A', 'B', 'D', 'E', 'C', 'F']
- B) ['A', 'B', 'C', 'D', 'E', 'F']
- C) ['A', 'C', 'F', 'B', 'E', 'D']
- D) ['F', 'E', 'D', 'C', 'B', 'A']

**Answer: B** - BFS visits level by level: A (level 0), B,C (level 1), D,E,F (level 2).

### CODE-Q4: What Does This Code Output?
```python
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [], 'E': [], 'F': []
}

def dfs(start):
    visited = []
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.append(node)
            stack.extend(graph[node])
    return visited

print(dfs('A'))
```
- A) ['A', 'B', 'C', 'D', 'E', 'F']
- B) ['A', 'C', 'F', 'B', 'E', 'D']
- C) ['A', 'B', 'D', 'E', 'C', 'F']
- D) ['A', 'C', 'B', 'F', 'E', 'D']

**Answer: B** - DFS with stack: pop A, push [B,C]. Pop C (last), push [F]. Pop F. Pop B, push [D,E]. Pop E. Pop D. Result: A, C, F, B, E, D.

### CODE-Q5: Identify the Bug
"What is wrong with this BFS implementation?"
```python
def bfs_broken(start, goal):
    queue = [start]
    while queue:
        current = queue.pop(0)
        if current == goal:
            return True
        for neighbor in get_neighbors(current):
            queue.append(neighbor)     # <-- What's missing?
    return False
```
- A) Nothing is wrong
- B) Missing a visited set - will create infinite loops
- C) Should use stack instead of queue
- D) Should pop from the end

**Answer: B** - Without a visited set, the algorithm will revisit nodes endlessly in graphs with cycles.

### CODE-Q6: pop(0) vs pop()
"What is the difference between `queue.pop(0)` and `stack.pop()`?"
- A) No difference
- B) `pop(0)` removes the first element (FIFO); `pop()` removes the last element (LIFO)
- C) `pop(0)` removes the last element; `pop()` removes the first element
- D) Both remove random elements

**Answer: B** - This is why `pop(0)` is used in BFS (queue) and `pop()` is used in DFS (stack).

### CODE-Q7: A* Search Code Reading
```python
def a_star(start, goal, h):
    open_set = [(0 + h(start), start)]  # (f_score, node)
    g_score = {start: 0}
    visited = set()

    while open_set:
        open_set.sort()
        f, current = open_set.pop(0)    # Get node with lowest f
        if current == goal:
            return g_score[goal]
        visited.add(current)
        for neighbor, cost in get_neighbors(current):
            if neighbor not in visited:
                new_g = g_score[current] + cost
                if new_g < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = new_g
                    f_score = new_g + h(neighbor)
                    open_set.append((f_score, neighbor))
```
"In this A* implementation, what does `f_score = new_g + h(neighbor)` compute?"
- A) Only the heuristic estimate
- B) Only the actual path cost
- C) The total estimated cost: actual cost from start + heuristic estimate to goal
- D) The distance between two neighbors

**Answer: C** - f(n) = g(n) + h(n) is the core of A*.

### CODE-Q8: Greedy vs A* in Code
"What change would convert this A* search into a Greedy search?"
- A) Change `f_score = new_g + h(neighbor)` to `f_score = h(neighbor)`
- B) Change `f_score = new_g + h(neighbor)` to `f_score = new_g`
- C) Remove the visited set
- D) Use a stack instead of priority queue

**Answer: A** - Greedy uses only f(n) = h(n), ignoring the actual cost g(n).

### CODE-Q9: Linear Regression Code
```python
import numpy as np

X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 6, 8, 10])

w1 = np.sum((X - X.mean()) * (Y - Y.mean())) / np.sum((X - X.mean())**2)
w0 = Y.mean() - w1 * X.mean()

print(f"w1={w1}, w0={w0}")
```
"What will this code output?"
- A) w1=2.0, w0=0.0
- B) w1=1.0, w0=1.0
- C) w1=0.5, w0=1.5
- D) w1=3.0, w0=-1.0

**Answer: A** - The data is perfectly linear: Y = 2X + 0. So slope w1=2, intercept w0=0.

### CODE-Q10: Understanding Queue Operations
```python
queue = ['A']
visited = {'A'}
# Neighbors of A are B, C
# After processing A:
queue.append('B')
queue.append('C')
visited.add('B')
visited.add('C')
# Queue is now: ['B', 'C']

current = queue.pop(0)  # What is current?
```
- A) 'C'
- B) 'B'
- C) 'A'
- D) Error

**Answer: B** - `pop(0)` removes and returns the FIRST element. Queue goes from ['B','C'] to ['C'].

---

# PART 5: INTERNET-SOURCED SAMPLE QUESTIONS (From AI Exam Banks)

These questions are sourced from university AI exams and MCQ banks (InterviewBit, Sanfoundry, university midterms).

### INET-Q1: Search Memory
"Which of the following search methods takes less memory?"
- A) Depth First Search
- B) Breadth-First Search
- C) Linear Search
- D) Optimal Search

**Answer: A** - DFS stores only nodes on the current path, unlike BFS which stores the entire frontier.

### INET-Q2: Informed Search
"Which search algorithm is commonly used in AI problem-solving?"
- A) Blind Search
- B) Informed Search
- C) Depth-First Search
- D) Breadth-First Search

**Answer: B** - Informed search uses heuristics to make better decisions.

### INET-Q3: Unsupervised Learning
"Exploratory Learning is another name for:"
- A) Supervised learning
- B) Unsupervised learning
- C) Reinforcement learning
- D) None of the above

**Answer: B** - No external supervision is required.

### INET-Q4: Agent Perception
"How does an AI agent interact with its environment?"
- A) Using sensors and actuators
- B) Using only sensors
- C) Using only actuators
- D) None of the above

**Answer: A**

### INET-Q5: Search Completeness
"A search algorithm is said to be complete if:"
- A) It finds the optimal solution
- B) It terminates in finite time
- C) It guarantees to find a solution if one exists
- D) It uses minimum memory

**Answer: C**

### INET-Q6: Search Optimality
"A search algorithm is optimal if:"
- A) It runs in the least time
- B) It uses the least memory
- C) It finds the solution with the lowest cost
- D) It expands the fewest nodes

**Answer: C**

### INET-Q7: Heuristic Properties
"For A* to be optimal, the heuristic must be:"
- A) Consistent only
- B) Admissible (never overestimates)
- C) Always equal to actual cost
- D) Greater than actual cost

**Answer: B**

### INET-Q8: State Space Components
"A search problem is defined by which components?"
- A) Initial state, actions, transition model, goal test, path cost
- B) Only initial state and goal state
- C) Only actions and costs
- D) Only the search algorithm

**Answer: A**

### INET-Q9: Branching Factor
"The branching factor in a search tree is:"
- A) The depth of the tree
- B) The maximum number of successors of any node
- C) The total number of nodes
- D) The number of goal states

**Answer: B**

### INET-Q10: UCS vs BFS
"Uniform Cost Search reduces to Breadth-First Search when:"
- A) The heuristic is zero
- B) All step costs are equal
- C) The graph has no cycles
- D) The depth is limited

**Answer: B**

---

# PART 6: COMPLETE MOCK MIDTERM EXAM

## COMP237 - Introduction to AI - Mock Midterm
**Time: 60 minutes | Total: 80 points | 20 questions**

---

**Question 1 (3 pts) - Fill in the blank:**
An agent's behavior is described by the __________ that maps any given percept sequence to an action.

**Answer:** agent function

---

**Question 2 (5 pts) - Matching:**
Match each statement to the correct agent type:

| Statement | Options |
|-----------|---------|
| 1. "Problem generator element" | A) Simple Reflex |
| 2. "If condition then action" | B) Model-based |
| 3. "Maintains internal state of how world evolves" | C) Goal-based |
| 4. "Evaluates sequences for quality: quicker, safer, cheaper" | D) Utility-based |
| 5. "Any agent can be this type" | E) Learning |

**Answers:** 1-E, 2-A, 3-B, 4-D, 5-E

---

**Question 3 (3 pts) - MC:**
What are the two dimensions on which all AI definitions revolve?
- A) Speed and efficiency
- B) Human versus Rational and Thought versus Behavior
- C) Hardware and Software
- D) Agents and systems

**Answer: B**

---

**Question 4 (3 pts) - MC:**
For a machine to pass the standard Turing Test, it needs:
- A) NLP, Knowledge Representation, Automated Reasoning, Machine Learning
- B) Only NLP and Machine Learning
- C) Computer Vision and Robotics
- D) All 6 AI capabilities

**Answer: A**

---

**Question 5 (3 pts) - MC:**
After checking if a dequeued cell is the goal in BFS (and it is NOT the goal), what comes next?
- A) Update visited
- B) Dequeue again
- C) Exit algorithm
- D) Enqueue neighbors and update predecessors

**Answer: D**

---

**Question 6 (3 pts) - MC:**
Which search algorithm uses the least memory?
- A) DFS
- B) BFS
- C) A*
- D) UCS

**Answer: A** - DFS space complexity is O(bm) vs O(b^d) for BFS.

---

**Question 7 (3 pts) - MC:**
What is an admissible heuristic?
- A) A heuristic that is easy to compute
- B) A heuristic that never overestimates the actual cost to reach the goal
- C) A heuristic that always equals the true cost
- D) A heuristic that overestimates to be safe

**Answer: B**

---

**Question 8 (3 pts) - MC:**
In A* search, f(n) = g(n) + h(n). What does g(n) represent?
- A) The heuristic estimate from n to goal
- B) The actual path cost from start to n
- C) The total estimated cost
- D) The branching factor

**Answer: B**

---

**Question 9 (5 pts) - Written (PEAS):**
For the activity of "Autonomous Vehicle Navigation", give a PEAS description and characterize the environment:
1. Performance Measure
2. Environment
3. Actuators
4. Sensors
5. Type of environment

**Sample Answer:**
1. P: Safe arrival, minimize travel time, fuel efficiency, obey traffic laws, passenger comfort
2. E: Roads, highways, intersections, other vehicles, pedestrians, traffic signs, weather conditions
3. A: Steering wheel, brakes, accelerator, turn signals, horn, display screen
4. S: Cameras, LIDAR, radar, GPS, speedometer, ultrasonic sensors, microphones
5. Type: Partially observable, stochastic, multi-agent, dynamic, continuous

---

**Question 10 (3 pts) - MC:**
The main difference between the traditional approach and machine learning is:
- A) Traditional is faster
- B) Traditional focuses on rules to build logic; ML focuses on data to identify rules
- C) ML requires more hardware
- D) No difference

**Answer: B**

---

**Question 11 (4 pts) - Checkbox:**
Mark the valid challenges of machine learning:
- [ ] Insufficient quantity of data
- [ ] Poor data quality
- [ ] Scarcity of algorithms
- [ ] Under fitting the data
- [ ] Over fitting the data
- [ ] Scarcity of performance measures

**Answer:** Check: Insufficient data, Poor data quality, Underfitting, Overfitting. Do NOT check: Scarcity of algorithms, Scarcity of performance measures.

---

**Question 12 (3 pts) - MC:**
What are the three main types of machine learning?
- A) Supervised, Unsupervised, and Reinforcement
- B) Classification, Regression, and Clustering
- C) Linear, Non-linear, and Neural
- D) Training, Testing, and Validation

**Answer: A**

---

**Question 13 (3 pts) - MC:**
A model that performs well on training data but poorly on test data is experiencing:
- A) Underfitting
- B) Overfitting
- C) Good generalization
- D) Optimal performance

**Answer: B**

---

**Question 14 (3 pts) - MC:**
What is the order we used to push neighbors onto the stack in DFS?
- A) up, down, left, right
- B) up, right, down, left
- C) left, right, up, down
- D) left, down, right, up

**Answer:** (Check your lab notes for your course's specific convention)

---

**Question 15 (5 pts) - Code Reading:**
```python
graph = {'A':['B','C'], 'B':['D','E'], 'C':['F'], 'D':[], 'E':[], 'F':[]}

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
What does this code output?
- A) ['A', 'B', 'C', 'D', 'E', 'F']
- B) ['A', 'C', 'F', 'B', 'E', 'D']
- C) ['A', 'B', 'D', 'E', 'C', 'F']
- D) ['F', 'E', 'D', 'C', 'B', 'A']

**Answer: B** - This is DFS (uses `pop()` = LIFO). Pop A, extend [B,C]. Pop C, extend [F]. Pop F. Pop B, extend [D,E]. Pop E. Pop D.

---

**Question 16 (3 pts) - MC:**
What change converts A* search into Greedy search?
- A) Change f(n) = g(n) + h(n) to f(n) = h(n) only
- B) Change f(n) = g(n) + h(n) to f(n) = g(n) only
- C) Remove the heuristic
- D) Use a stack instead of priority queue

**Answer: A**

---

**Question 17 (3 pts) - MC:**
In linear regression, y = w1*x + w0. The null hypothesis H0 is:
- A) w1 = 1
- B) w1 = 0 (no relationship)
- C) w0 = 0
- D) R^2 = 1

**Answer: B**

---

**Question 18 (3 pts) - MC:**
The L2 loss function is:
- A) |y - y_hat|
- B) (y - y_hat)^2
- C) 0 if y=y_hat, else 1
- D) log(y - y_hat)

**Answer: B**

---

**Question 19 (3 pts) - Fill in the blank:**
"For BFS, time complexity is O(___) and space complexity is O(___), where b is the branching factor and d is the depth."

**Answer:** O(b^d) and O(b^d)

---

**Question 20 (5 pts) - Code Fill-in:**
Complete this BFS implementation:
```python
def bfs(start, goal):
    queue = [start]
    visited = {start}

    while queue:
        current = ___________          # BLANK 1: get next node
        if current == goal:
            return True
        for neighbor in get_neighbors(current):
            if ___________:            # BLANK 2: check condition
                ___________            # BLANK 3: mark visited
                ___________            # BLANK 4: add to explore
    return False
```

**Answers:**
- BLANK 1: `queue.pop(0)`
- BLANK 2: `neighbor not in visited`
- BLANK 3: `visited.add(neighbor)`
- BLANK 4: `queue.append(neighbor)`

---

# PART 7: STUDY NOTES (Quick Reference)

## Key Formulas
| Formula | Meaning |
|---------|---------|
| f(n) = g(n) + h(n) | A* evaluation function |
| f(n) = h(n) | Greedy search |
| y = w1*x + w0 | Linear regression |
| R^2 = 1 - (SSE/SST) | R-squared (model fit) |
| L2 = (y - y_hat)^2 | Squared loss |
| Z = (Am - A0) / (sigma/sqrt(n)) | Z-test |
| chi^2 = sum((O-E)^2 / E) | Chi-square test |

## Key Definitions
| Term | Definition |
|------|-----------|
| Agent function | Maps percept sequences to actions |
| Rational agent | Maximizes expected performance |
| PEAS | Performance, Environment, Actuators, Sensors |
| Admissible heuristic | Never overestimates actual cost |
| State space | Set of all possible configurations |
| Overfitting | High variance, fits noise, poor on test data |
| Underfitting | High bias, too simple, poor on both train & test |
| Supervised learning | Learns from labeled input-output pairs |
| Unsupervised learning | Finds patterns without labels (clustering) |
| Reinforcement learning | Learns from rewards and punishments |

## Algorithm Quick Reference
| Algorithm | Data Structure | Complete? | Optimal? | Space |
|-----------|---------------|-----------|----------|-------|
| BFS | Queue (FIFO) | Yes | Yes (unweighted) | O(b^d) |
| DFS | Stack (LIFO) | No | No | O(bm) |
| UCS | Priority Queue | Yes | Yes | O(b^d) |
| Greedy | Priority Queue | No | No | O(b^m) |
| A* | Priority Queue | Yes | Yes (admissible h) | O(b^d) |

## Agent Types Quick Reference
| Agent Type | Key Feature |
|-----------|------------|
| Simple Reflex | If-then rules, no memory |
| Model-Based Reflex | Internal model of world state |
| Goal-Based | Works toward specific goals |
| Utility-Based | Evaluates quality (quicker, safer, cheaper) |
| Learning | Improves from experience, has problem generator |

## 5 AI Disciplines
1. Natural Language Processing (NLP)
2. Knowledge Representation & Automated Reasoning
3. Machine Learning
4. Computer Vision
5. Robotics

## Turing Test Capabilities
- Standard: NLP, Knowledge Representation, Automated Reasoning, Machine Learning
- Total Turing Test adds: Computer Vision, Robotics

---

# PART 8: HIGH-CONFIDENCE PREDICTIONS

These are ALMOST CERTAINLY on the midterm based on quiz patterns + Studocu previews:

1. ✅ Agent function fill-in-the-blank
2. ✅ Agent type matching (Learning, Reflex, Model-based, Utility-based)
3. ✅ PEAS description for an activity (Soccer, Autonomous Vehicle, Chess, or Medical)
4. ✅ BFS/DFS algorithm operation questions
5. ✅ Neighbor ordering on stack
6. ✅ Traditional vs ML comparison
7. ✅ ML challenges checklist (overfitting, underfitting, data quality, insufficient data)
8. ✅ Code-based question (BFS or DFS pseudocode fill-in or output prediction)
9. ✅ A* search formula f(n) = g(n) + h(n)
10. ✅ Admissible heuristic definition
11. ✅ Supervised vs Unsupervised vs Reinforcement learning
12. ✅ Overfitting vs Underfitting definitions
13. ✅ Search algorithm comparison (BFS vs DFS vs A*)
14. ✅ Data structure choice (Queue for BFS, Stack for DFS)

---

## Sources Used for Internet Questions
- [COMP237 Studocu Materials](https://www.studocu.com/en-ca/course/centennial-college/introduction-to-ai/5127465)
- [COMP237 Midterm MCQs Review](https://www.studocu.com/en-ca/document/centennial-college/introduction-to-ai/midtermmcqs-mid-term-test/49568569)
- [InterviewBit AI MCQs](https://www.interviewbit.com/artificial-intelligence-mcq/)
- [Sanfoundry 1000 AI MCQs](https://www.sanfoundry.com/artificial-intelligence-questions-answers/)
- [University of Pittsburgh CS2710 Midterm](https://people.cs.pitt.edu/~litman/courses/cs2710/lectures/cs2710mid.pdf)
- [UW-Madison CS540 Past Exams](https://pages.cs.wisc.edu/~dyer/cs540/exams-toc.html)
