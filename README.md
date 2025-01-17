# Neurevolution for Algo Trading Testing

## Overview
**Neurevolution for Algo Trading Testing** is a project that leverages the power of NeuroEvolution, specifically the NEAT (NeuroEvolution of Augmenting Topologies) framework, to optimize algorithmic trading strategies in volatile markets like cryptocurrency trading. The system evolves neural networks to adaptively predict and react to market dynamics, helping users make better trading decisions.

## Features
- **NEAT Implementation**: Evolutionary neural networks optimized for buy, sell, and hold decisions.
- **Dynamic Configuration**: Easily adjustable NEAT hyperparameters via configuration files.
- **Reinforcement Learning Integration**: Combined neural predictions with reinforcement learning strategies to enhance profitability.
- **Custom Risk Management**: Stop loss and take profit strategies implemented with EMA crossover logic.
- **Real-Time Data Sync**: Automatic updates of training data with real-time market data.
- **SMOTE Implementation**: Addressed class imbalance using Synthetic Minority Over-sampling Technique.
- **Visualization Tools**: Heatmaps and MPLfinance for analyzing historical and forecasted trends.

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - `neat-python`: For NeuroEvolution framework
  - `ccxt`: For market data integration
  - `pandas`, `numpy`: Data manipulation and preprocessing
  - `matplotlib`, `mplfinance`: Visualization
  - `pickle`: Saving and loading models

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Neurevolution-for-Algo-Trading-Testing.git
   cd Neurevolution-for-Algo-Trading-Testing
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the following installed:
   - Python 3.8 or higher
   - Necessary API keys for market data (e.g., from OKX or Binance)

## Usage
### 1. Configure the NEAT Parameters
Adjust the settings in the `neat_config6.txt` file to match your trading requirements:
```ini
[NEAT]
pop_size = 100
fitness_criterion = max
fitness_threshold = 999999999999999999999
...
```

### 2. Run the Main Script
Run the `main.py` file to start the training and testing process:
```bash
python main.py
```

### 3. Analyze Results
- Review logs and visualizations generated during testing.
- The model outputs the best genome and its associated neural network.

### 4. Live Trading (Optional)
Integrate the trained model into your live trading system using real-time data from the configured exchange.

## Key Files
- `main.py`: Main script to run the training and testing processes.
- `neat_config6.txt`: Configuration file for NEAT hyperparameters.
- `data/`: Folder containing historical market data.
- `utils.py`: Utility functions for preprocessing, visualization, and logging.
- `results/`: Directory for storing trained models and logs.

## Example Outputs
- **Heatmap Visualization**: Shows the probability distribution of future prices.
- **Performance Metrics**: Evaluates buy, sell, and hold signals based on profitability.
- **Live Trading**: Generates orders for real-time execution using leveraged trading.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact
For questions or collaborations, reach out to:
- **Author**: Erald
- **Email**: [your-email@example.com]
- **GitHub**: [your-github-profile-link]

---

Happy Trading 🚀
