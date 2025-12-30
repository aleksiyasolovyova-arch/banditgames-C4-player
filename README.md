# Connect4 AI Player Service

This repository contains the **AI Player Service** for a Connect Four platform.  
The service is responsible for making intelligent move decisions during games and acting as the AI opponent.

At its core, the system uses **Monte Carlo Tree Search (MCTS)** to explore future game states and select strong moves under time and difficulty constraints. It integrates with the wider platform through HTTP and event-based communication.

---

## Purpose

The AI Player Service is designed to:
- Act as an autonomous AI opponent for Connect Four games
- Respond to live game events with valid and competitive moves
- Support multiple difficulty levels and adaptive gameplay
- Generate detailed gameplay data for analysis and machine learning

---

## How It Works

- The service listens to game events from the backend (via RabbitMQ)
- When a move is required, it reconstructs the current game state
- An MCTS-based agent searches possible continuations and selects a move
- The selected move is sent back to the backend API
- Detailed statistics (visit counts, Q-values, rollouts) are published for logging and learning

---

## AI Architecture

- **MCTS Engine**  
  Implements full Monte Carlo Tree Search with tree reuse, rollouts, and UCB-based selection.

- **AI Manager**  
  Manages per-game agents, skill levels, and dynamic difficulty adjustment (DDA).

- **Adaptive and Reference Agents**  
  The adaptive agent plays the actual move, while an optional reference (expert) agent runs in parallel to provide baseline statistics for evaluation and training.

---

## System Integration

This service integrates with:
- A Connect4 backend (HTTP API) for submitting moves and retrieving game state
- RabbitMQ for receiving game events and publishing gameplay logs
- Downstream ML pipelines that use logged data for policy imitation and win-probability modeling

---

## Role in the Platform

The AI Player Service is used for:
- Human vs AI matches
- AI vs AI self-play games
- Continuous evaluation and data generation
- Supporting learning-based improvements to the AI system

It is designed to be robust, observable, and suitable for long-running production deployments.

---

## Summary

This project provides a production-ready AI player for Connect Four, combining classic search-based decision making (MCTS) with modern service architecture and data-driven learning workflows.
