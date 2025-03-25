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

## 🔍 Example 1: GRU with Input Sequence: "I love myself"

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
  ✅ **Candidate Memory**: `"I"`

- **Memory Tracker (hₜ)**:  
  Update memory with the new input.  
  ✅ **Output**: `"I"`

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
  ✅ **Candidate Memory**: `"I love"`

- **Memory Tracker (hₜ)**:  
  Blend the current memory with the candidate memory:  
  `hₜ = 0.6("I love") + 0.4("I") = "I love"`

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

---

## 🔍 Example 2: LSTM with Input Sequence: "I love myself"

### Roles (5 People)
- **Input Provider (xₜ)** – Feeds words into the model.
- **Forget Gate (fₜ)** – Decides how much previous memory to forget.
- **Input Gate (iₜ)** – Decides how much of the new information to add to memory.
- **Candidate Memory (ĉₜ)** – Suggests the new memory from the current input and previous memory.
- **Memory Tracker (cₜ)** – Stores the cell state (memory) and shares the hidden state (hₜ).

---

### Step 1: Time Step 1 ("I")
- **Input Provider (xₜ)**:  
  Passes the first input: `xₜ = "I"`

- **Forget Gate (fₜ)**:  
  No prior memory, so nothing needs to be forgotten.  
  ✅ **Decision**: Forget = 0 (keep all old memory).

- **Input Gate (iₜ)**:  
  Fully accept new information.  
  ✅ **Decision**: Input = 1 (100% new memory).

- **Candidate Memory (ĉₜ)**:  
  Use `"I"` as new memory.  
  ✅ **Candidate Memory**: `"I"`

- **Memory Tracker (cₜ)**:  
  Update the memory by combining the previous memory (none) and the new input.  
  ✅ **Updated Memory (cₜ)**: `"I"`

- **Hidden State (hₜ)**:  
  Based on memory, output `"I"`.  
  ✅ **Output**: `"I"`

---

### Step 2: Time Step 2 ("love")
- **Input Provider (xₜ)**:  
  Passes the next input: `xₜ = "love"`

- **Forget Gate (fₜ)**:  
  `"I"` is relevant, but let's decide to retain it.  
  ✅ **Decision**: Forget = 0.4 (keep 60% of `"I"`).

- **Input Gate (iₜ)**:  
  New information should partially update memory.  
  ✅ **Decision**: Input = 0.6 (keep 60% new input).

- **Candidate Memory (ĉₜ)**:  
  Combine `"I"` and `"love"` to create the new memory.  
  ✅ **Candidate Memory**: `"I love"`

- **Memory Tracker (cₜ)**:  
  Update memory:  
  `cₜ = 0.4("I") + 0.6("I love") = "I love"`

- **Hidden State (hₜ)**:  
  Output based on updated memory.  
  ✅ **Output**: `"I love"`

---

### Step 3: Time Step 3 ("myself")
- **Input Provider (xₜ)**:  
  Passes the next input: `xₜ = "myself"`

- **Forget Gate (fₜ)**:  
  Keep most of the previous memory, but reduce a little bit.  
  ✅ **Decision**: Forget = 0.2 (keep 80% of `"I love"`).

- **Input Gate (iₜ)**:  
  New input is highly relevant—significant update.  
  ✅ **Decision**: Input = 0.8 (keep 80% new information).

- **Candidate Memory (ĉₜ)**:  
  Combine `"I love"` with `"myself."`  
  ✅ **Candidate Memory**: `"I love myself"`

- **Memory Tracker (cₜ)**:  
  Update the memory:  
  `cₜ = 0.2("I love") + 0.8("I love myself") = "I love myself"`

- **Hidden State (hₜ)**:  
  Output based on updated memory.  
  ✅ **Output**: `"I love myself"`

✅ **Final Output**: `"I love myself"`