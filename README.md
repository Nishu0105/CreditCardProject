# Credit Card Fraud Detection

A web application that uses machine learning to detect fraudulent credit card transactions.

## Overview

This project implements a credit card fraud detection system using a Random Forest classifier. The application is built with Flask and can be deployed on Render or run locally. It provides a simple web interface where users can input transaction features to check if a transaction is fraudulent or legitimate.

## Features

- Machine learning model to detect credit card fraud
- Web interface for easy interaction
- Balanced dataset handling for better model performance
- Optimized for deployment with reduced data size

## Tech Stack

- **Python 3.9**
- **Flask**: Web framework
- **Pandas & NumPy**: Data processing
- **Scikit-learn**: Machine learning (Random Forest classifier)
- **Gunicorn**: WSGI HTTP Server for deployment

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd CreditCard-2
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   python app.py
   ```

5. Open your browser and navigate to `http://localhost:5000`

## Dataset

The application uses a compressed version of the credit card fraud detection dataset. The original dataset has been reduced in size while maintaining a balance between fraud and non-fraud cases to ensure optimal model performance and faster loading times.

To prepare the dataset:
1. Place your original credit card dataset in the `data` folder as `creditcard_compressed.csv`
2. Run the data compression script:
   ```
   python compress_data.py
   ```
3. This will create a reduced version of the dataset at `data/creditcard_reduced.csv`

## Usage

1. Access the web interface
2. Enter 30 comma-separated values representing:
   - V1-V28: PCA-transformed features
   - Amount: Transaction amount
   - Time: Seconds elapsed between this transaction and the first transaction
3. Click "Predict" to see if the transaction is fraudulent or normal

## Deployment

The application is configured for deployment on Render:

1. Push your code to a Git repository
2. Create a new Web Service on Render
3. Connect your repository
4. The `render.yaml` file will automatically configure the deployment

## License

This project is licensed under the terms included in the LICENSE file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
