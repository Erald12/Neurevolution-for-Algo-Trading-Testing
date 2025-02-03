Neuroevolution for Algo Trading Testing
Overview
Neuroevolution for Algo Trading Testing leverages the power of NeuroEvolution, specifically the NEAT (NeuroEvolution of Augmenting Topologies) framework, to optimize algorithmic trading strategies in volatile markets like cryptocurrency. The system evolves neural networks to predict and react to market dynamics, helping users make better trading decisions.

Features
NEAT Implementation: Evolutionary neural networks optimized for buy, sell, and hold decisions.
Dynamic Configuration: Easily adjustable NEAT hyperparameters via configuration files.
Reinforcement Learning Integration: Combined neural predictions with reinforcement learning strategies to enhance profitability.
Custom Risk Management: Stop loss and take profit strategies implemented with EMA crossover logic.
Real-Time Data Sync: Automatic updates of training data with real-time market data.
SMOTE Implementation: Addressed class imbalance using Synthetic Minority Over-sampling Technique.
Visualization Tools: Heatmaps and MPLfinance for analyzing historical and forecasted trends.
Live Trading Testing: Use the Test_Live_Trading.py script to simulate live trading with real-time data.

Technologies Used
Programming Language: Python

Libraries:
neat-python: For the NeuroEvolution framework.
ccxt: For market data integration.
ta: For technical analysis indicators.
tvDatafeed: For retrieving data from TradingView.
numpy, pandas: Data manipulation and preprocessing.
scipy: For statistical operations like linear regression.
rich: For colorful terminal output.
pickle: For saving and loading models.
matplotlib, mplfinance: Visualization.

Installation
Clone the repository:
git clone https://github.com/Erald12/Neuroevolution-for-Algo-Trading-Testing.git
cd Neuroevolution-for-Algo-Trading-Testing

Install the required dependencies:
pip install -r requirements.txt

Ensure you have the following installed:

Python 3.8 or higher
Necessary API keys for market data (e.g., from OKX or Binance)

Usage
Configure the NEAT Parameters
Adjust the settings in the neat_config6.txt file to match your trading requirements:

[NEAT]
pop_size = 100
fitness_criterion = max
fitness_threshold = 999999999999999999999
...

Run the Training and Testing
To start the training and testing process, run the neat_model_17_train.py script:
python neat_model_17_train.py

Test Live Trading
For live trading, you can use the Test_Live_Trading.py file. This script simulates live trading and executes trades based on the trained model:
python Test_Live_Trading.py

Analyze Results
Review logs and visualizations generated during testing. The model will output the best genome and its associated neural network for evaluation.

Key Files
neat_model_17_train.py: Main script for running the training and testing processes.
Test_Live_Trading.py: Script to simulate live trading with real-time data.
neat_config6.txt: Configuration file for NEAT hyperparameters.
data/: Folder containing historical market data.
utils.py: Utility functions for preprocessing, visualization, and logging.
results/: Directory for storing trained models and logs.
Example Outputs
Heatmap Visualization: Shows the probability distribution of future prices.
Performance Metrics: Evaluates buy, sell, and hold signals based on profitability.
Live Trading: Generates orders for real-time execution using leveraged trading.
Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Contact
For questions or collaborations, reach out to:

Author: Erald
Email: [erald.almoete@gmail.com]
GitHub: [https://github.com/Erald12]
Happy Trading 🚀
