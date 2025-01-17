
# ScorePredict-AI

**ScorePredict-AI** is a machine learning-based system designed to predict the performance scores of football players using historical game data. The project focuses on analyzing player statistics, clustering them into performance groups, and ultimately predicting future performance based on various features.

The project uses advanced machine learning techniques to cluster players based on performance metrics and make predictions about their future contributions in games. By leveraging statistical data, the system helps fantasy football players and analysts make more informed decisions.

---

## What is ScorePredict-AI?

**ScorePredict-AI** is a machine learning-based system aimed at predicting the performance of NFL players. The system processes historical data to analyze and predict player performance based on various features such as touchdowns, yards, receptions, and other key performance metrics. 

The project uses techniques like **KMeans Clustering** and **Principal Component Analysis (PCA)** to group players by their performance. Clustering players with similar statistics helps in identifying performance trends and relationships among different player types.

- **KMeans Clustering**: This method groups players based on their historical performance, such as touchdowns, yards, receptions, and other important metrics. The clusters represent groups of players with similar performance profiles.
- **Principal Component Analysis (PCA)**: PCA is used to reduce the dimensionality of the dataset. It helps in transforming the complex data into a simpler format while retaining the most important features, making it easier to visualize and analyze.

By analyzing these clusters and player features, **ScorePredict-AI** can identify patterns in player performance and predict future results based on historical trends.

---

## Why was ScorePredict-AI Created?

The primary goal of **ScorePredict-AI** is to provide better insights into player performance through data-driven analysis and predictions. The project is particularly useful in the following contexts:

- **Fantasy Football**: By predicting player performance, **ScorePredict-AI** helps fantasy football players make informed decisions when selecting players. The predictions can assist in identifying high-performing players or predicting potential breakout candidates.
- **Coaches and Analysts**: Football teams, coaches, and analysts can use the system to predict a player's future contributions based on their historical performance, enabling better decision-making for team strategies.
- **Sports Data Scientists**: Data scientists working in sports analytics can leverage this project to apply machine learning techniques and gain valuable insights into player performance and trends.

The system uses historical player data to generate predictions, allowing teams and analysts to plan more effectively for upcoming games. It enhances the decision-making process by using clustering to identify players with similar performance metrics.

---

## How Does ScorePredict-AI Work?

**ScorePredict-AI** works by following a structured process to predict player performance and analyze player statistics:

1. **Data Collection**:
   - **ScorePredict-AI** loads historical data from various CSV files that contain player stats, including:
     - `gamesUpdated.csv`: Contains football game data, which is essential for training the model.
     - `K_analysis.csv`, `QB_analysis.csv`, `RB_analysis.csv`, etc.: These files contain performance data specific to different player positions such as Quarterback, Running Back, and Kicker.

2. **Data Preprocessing**:
   - The data is cleaned and prepared for analysis by selecting relevant features such as games played, predicted points, and position-specific stats like touchdowns, yards, receptions, etc.

3. **Clustering**:
   - The model applies **KMeans Clustering** to group players based on their performance metrics. The clustering process is designed to group players with similar characteristics, which helps in identifying performance patterns.
   - **PCA** is then applied to reduce the dimensionality of the dataset and visualize it in 2D or 3D, making it easier to understand the relationships between players' performance.

4. **Analysis**:
   - Once the clusters are formed, **ScorePredict-AI** performs a detailed analysis of each cluster:
     - Average predicted points for each cluster.
     - The range of predicted points within each cluster.
     - The number of players in each cluster.
   
5. **Prediction**:
   - The system uses the results from the clustering process to predict future player performance based on their historical data. This helps in forecasting how players might perform in upcoming games.

---

## How to Run:

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/emad-8869/ScorePredict-AI.git
   ```

2. **Install required dependencies (make sure Python 3.x is installed):**
   - Create a virtual environment (optional but recommended):
     ```bash
     python3 -m venv venv
     ```
   - Activate the virtual environment:
     - On macOS/Linux:
       ```bash
       source venv/bin/activate
       ```
     - On Windows:
       ```bash
       .env\Scriptsctivate
       ```
   - Install the required libraries:
     ```bash
     pip install -r requirements.txt
     ```

3. Run the main script:
   ```bash
   python main.py
   ```

4. View the analysis and visualizations generated from the clustering process.

---

## Files:

- `main.py`: Main script for merging datasets, processing, and clustering players.
- `KMeans.ipynb`: Jupyter notebook for clustering analysis based on player position.
- `ScorePredict-AI.ipynb`: Jupyter notebook for loading and exploring game data.
- `gamesUpdated.csv`: Contains football game data used for training.
- `K_analysis.csv`, `QB_analysis.csv`, `RB_analysis.csv`, `TE_analysis.csv`, `WR_analysis.csv`: Player performance data for different positions.
- `clustered_players_with_pca.csv`: Output file with clustered player data and PCA analysis.

---

## License:
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments:
- The dataset `gamesUpdated.csv` was sourced from publicly available football data.
- Thanks to the contributors and open-source community for providing useful machine learning and data analysis libraries.
