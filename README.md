# 🏆 FIFA World Cup 2026 Predictor

A hybrid Python/Rust machine learning and Monte Carlo simulation platform designed to forecast outcomes, group standings, and knockout paths for the enlarged 48-team **FIFA World Cup 2026**.

By leveraging high-performance Rust execution for intensive match simulations alongside Python’s analytical ecosystem for data modeling, this repository provides data-driven tournament insights at scale.

---

## 📌 Features

- **Hybrid Architecture**: Blends data preprocessing and visualization in Python (`Jupyter Notebook`) with a blazing-fast core engine in Rust (`cargo`).
- **Monte Carlo Tournament Engine**: Simulates tens of thousands of tournament runs in seconds to calculate probabilistic outcomes (e.g., reaching the Round of 32, Quarter-finals, or winning the final).
- **Match Event Generation**: Simulates discrete in-game events, goals, and penalty shootouts based on team ratings and custom statistical models.
- **Automated CI Pipeline**: Continuous Integration configured via GitHub Actions (`.github/workflows/rust.yml`) ensures robust, error-free Rust builds.

---

## 📁 Repository Structure

```text
fifa-wc2026-predictor/
├── .github/
│   └── workflows/
│       └── rust.yml          # CI build and test pipeline
├── data/
│   └── README.md             # Data schemas and source documentation
├── rust/
│   ├── Cargo.toml            # Rust dependencies & manifest
│   └── src/
│       ├── data.rs           # Data loading & team record structures
│       ├── events.rs         # In-game match event logic
│       ├── main.rs           # CLI entry point
│       ├── model.rs          # Probability and prediction models
│       ├── rng.rs            # Custom fast RNG seeding
│       └── sim.rs            # Tournament bracket & simulation loop
├── FIFA_World_Cup_2026.ipynb # Data exploration & model building
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have the following installed on your machine:
* **Python**: 3.9+ 
* **Rust & Cargo**: 1.70+ (`rustup`)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/fifa-wc2026-predictor.git
   cd fifa-wc2026-predictor
   ```

2. **Set up the Python Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Build the Rust Simulation Engine:**
   ```bash
   cd rust
   cargo build --release
   cd ..
   ```

---

## 💻 Usage

### 1. Data Exploration & Modeling (Python)
Launch the Jupyter Notebook to explore underlying historical data, tune team Elo/expected goal ($xG$) models, and export structured outputs:

```bash
jupyter notebook FIFA_World_Cup_2026.ipynb
```

### 2. High-Speed Tournament Simulation (Rust)
Run the release-compiled Rust engine to perform thousands of tournament simulations:

```bash
cd rust
cargo run --release
```

---

## 🧪 Testing

To run unit tests across the Rust engine and verify event simulation logic:

```bash
cd rust
cargo test
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more details.
