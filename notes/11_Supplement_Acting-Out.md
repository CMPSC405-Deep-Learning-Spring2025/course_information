# 🎯 Be the Memory Network!

## Objective:
Work together to act out how an LSTM or GRU processes information step by step. Each person will play a specific role in the network, deciding what to remember, forget, and output.

---

## Background: What Are LSTMs and GRUs?

When working with sequences—like sentences, time-series data, or sensor readings—we often want a model to remember important information from earlier and use it later.

### 📊 LSTM (Long Short-Term Memory)
LSTMs manage information using three gates:
- **Forget Gate**: Decides how much past information to forget.
- **Input Gate**: Chooses how much of the new input to store.
- **Output Gate**: Decides what information to share as output.

💡 *Think of LSTMs like a complex filing system—they keep important memories and discard irrelevant ones.*

### 🔄 GRU (Gated Recurrent Unit)
GRUs are a simplified version of LSTMs with two gates:
- **Update Gate**: Controls how much old memory to keep.
- **Reset Gate**: Determines how much past information to mix with the new input.

💡 *GRUs are like a quick memory filter—faster but less detailed than LSTMs.*

---

## 🎭 Your Roles in the Network

In groups of 4-5 people, you will act out an LSTM and a GRU. Each person will take on one of these roles:

### For LSTM:
- **Input Provider (xₜ)**: Gives a new piece of information.
- **Forget Gate**: Decides how much of the previous memory to erase.
- **Input Gate**: Chooses how much of the new input to keep.
- **Output Gate**: Decides what information is passed to the next step.
- **Memory Tracker**: Updates and carries the memory (cell state and hidden state).

### For GRU:
- **Input Provider (xₜ)**: Gives a new piece of information.
- **Reset Gate**: Decides how much old memory to mix with the new input.
- **Update Gate**: Controls how much of the previous memory to keep.
- **Memory Tracker**: Updates and carries the memory (hidden state).

---

## 📚 How to Play the Simulation

1. Form a group and assign roles.
2. Follow the step-by-step process for either the LSTM or GRU below.
3. Communicate clearly—each gate should announce its decision before passing the information to the next person.

---

## 🔍 Example: Acting Out an LSTM (5 People)

Let’s walk through a specific example using the phrase:

**Input Sequence**: `"I love myself."`

### Step 1: Time Step 1 ("I")
- **Input Provider (xₜ)**:  
  `"I"` is the first input. Passes `xₜ = "I"` to the gates.

- **Forget Gate**:  
  No previous memory exists yet, so there is nothing to forget.  
  **Decision**: Forget 0%.

- **Input Gate**:  
  New information is valuable at the start.  
  **Decision**: Keep 100% of `"I"`.

- **Output Gate**:  
  Share `"I"` as output since it is the only word.  
  **Decision**: Output = `"I"`.

- **Memory Tracker**:  
  - **Cell State**: `"I"`  
  - **Hidden State**: `"I"` (output)

✅ **Output so far**: `"I"`

---

### Step 2: Time Step 2 ("love")
- **Input Provider (xₜ)**:  
  Passes the next input: `xₜ = "love"`.

- **Forget Gate**:  
  Since `"I"` is still relevant, keep most of it.  
  **Decision**: Forget 10% of previous memory.

- **Input Gate**:  
  `"Love"` is important new information.  
  **Decision**: Keep 90% of `"love"`.

- **Output Gate**:  
  We want to highlight both `"I"` and `"love."`  
  **Decision**: Output = `"I love"`.

- **Memory Tracker**:  
  - **Cell State**: `"I + 0.9(love)"`  
  - **Hidden State**: `"I love"` (output)

✅ **Output so far**: `"I love"`

---

### Step 3: Time Step 3 ("myself")
- **Input Provider (xₜ)**:  
  Passes the final input: `xₜ = "myself"`.

- **Forget Gate**:  
  `"I"` is less important now.  
  **Decision**: Forget 50% of the first memory.

- **Input Gate**:  
  `"Myself"` is very important.  
  **Decision**: Keep 100% of `"myself"`.

- **Output Gate**:  
  Pass the combined knowledge.  
  **Decision**: Output = `"I love myself"`.

- **Memory Tracker**:  
  - **Cell State**: `0.5("I") + 0.9("love") + 1("myself")`  
  - **Hidden State**: `"I love myself"` (output)

✅ **Final Output**: `"I love myself"`

---

## 🔍 Example: Acting Out a GRU (5 People)

Let’s walk through the GRU process using the input sequence:

**Input Sequence**: `"I love myself."`

### Roles (5 People)
- **Input Provider (xₜ)** – Feeds words into the model.
- **Reset Gate (rₜ)** – Decides how much previous memory to erase.
- **Update Gate (zₜ)** – Decides how much new information to keep.
- **Candidate Memory (h̃ₜ)** – Suggests the updated memory using new input and old memory.
- **Memory Tracker (hₜ)** – Stores and shares the hidden state (output).

---

### Step 1: Time Step 1 ("I")
- **Input Provider (xₜ)**:  
  Passes the first input: `xₜ = "I"`

- **Reset Gate (rₜ)**:  
  No prior information—allow all input to flow.  
  ✅ **Decision**: Reset = 1 (keep everything).

- **Update Gate (zₜ)**:  
  This is the first word, so keep it fully.  
  ✅ **Decision**: Update = 1 (100% new information).

- **Candidate Memory (h̃ₜ)**:  
  Since the reset gate is open, use `"I"` as new memory.  
  ✅ **Candidate Memory**: `"I"`.

- **Memory Tracker (hₜ)**:  
  Update memory with the new input.  
  ✅ **Output**: `"I"`

✅ **Output so far**: `"I"`

---

### Step 2: Time Step 2 ("love")
- **Input Provider (xₜ)**:  
  Passes the next input: `xₜ = "love"`

- **Reset Gate (rₜ)**:  
  `"I"` is still relevant—keep it.  
  ✅ **Decision**: Reset = 1 (keep previous information).

- **Update Gate (zₜ)**:  
  New input is important—partially update.  
  ✅ **Decision**: Update = 0.6 (60% new, 40% old).

- **Candidate Memory (h̃ₜ)**:  
  Mix `"I"` and `"love"` to form updated memory.  
  ✅ **Candidate Memory**: `"I love"`.

- **Memory Tracker (hₜ)**:  
  Blend the current memory with the candidate memory:  
  `hₜ = 0.6("I love") + 0.4("I") = "I love"`

✅ **Output so far**: `"I love"`

---

### Step 3: Time Step 3 ("myself")
- **Input Provider (xₜ)**:  
  Passes the next input: `xₜ = "myself"`

- **Reset Gate (rₜ)**:  
  Keep most of the past but reduce earlier context slightly.  
  ✅ **Decision**: Reset = 0.8 (keep 80% of memory).

- **Update Gate (zₜ)**:  
  `"Myself"` is important—update memory significantly.  
  ✅ **Decision**: Update = 0.7 (70% new, 30% old).

- **Candidate Memory (h̃ₜ)**:  
  Combine `"I love"` with `"myself."`  
  ✅ **Candidate Memory**: `"I love myself"`

- **Memory Tracker (hₜ)**:  
  Update by blending old and new:  
  `hₜ = 0.7("I love myself") + 0.3("I love") = "I love myself"`

✅ **Final Output**: `"I love myself"`
