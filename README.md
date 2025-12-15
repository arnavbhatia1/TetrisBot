# TetrisBot

A Q-learning agent that learns to play Tetris using reinforcement learning with a neural network.

## Prerequisites

- Java 8 or higher
- Git

To check if you have Java installed:
```bash
java -version
```

## Getting Started

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd TetrisBot
```

### 2. Compile the Agent

```bash
javac -cp "lib/tetris.jar:lib/argparse4j-0.9.0.jar" src/pas/tetris/agents/TetrisQAgent.java
```

### 3. Run the Tetris Bot

To see the bot in action with a GUI:

```bash
java -cp "lib/tetris.jar:lib/argparse4j-0.9.0.jar:src" edu.bu.tetris.Main -a edu.bu.tetris.agents.TrainerAgent -q pas.tetris.agents.TetrisQAgent -t 5 -v 2
```

This will:
- Train the agent for 5 games
- Evaluate it for 2 games
- Display the game in a GUI window

### Run Options

**Silent Mode (no GUI):**
```bash
java -cp "lib/tetris.jar:lib/argparse4j-0.9.0.jar:src" edu.bu.tetris.Main -a edu.bu.tetris.agents.TrainerAgent -q pas.tetris.agents.TetrisQAgent -t 5 -v 2 -s
```

**More Training Phases:**
```bash
java -cp "lib/tetris.jar:lib/argparse4j-0.9.0.jar:src" edu.bu.tetris.Main -a edu.bu.tetris.agents.TrainerAgent -q pas.tetris.agents.TetrisQAgent -t 10 -v 5 -p 3
```
- `-t 10`: Train for 10 games per phase
- `-v 5`: Evaluate for 5 games per phase
- `-p 3`: Repeat for 3 phases

## How It Works

The TetrisQAgent uses:
- **6 features** to evaluate board states: holes, bumpiness, vertical difference, column holes, bottom mino position, and completed lines
- **Neural network** with Dense layers and ReLU activation
- **Q-learning** with epsilon-greedy exploration
- **Reward shaping** that encourages line clears and penalizes poor board states

## Project Structure

```
TetrisBot/
├── src/pas/tetris/agents/
│   └── TetrisQAgent.java    # Q-learning agent implementation
├── lib/
│   ├── tetris.jar           # Tetris game framework
│   └── argparse4j-0.9.0.jar # Command-line argument parsing
└── doc/                     # API documentation
```
