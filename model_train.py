import joblib
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np

def train_model():
    print("Fetching MNIST data...")
    # X.shape is (70000, 784), y.shape is (70000,)
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    # Normalize the data (0-255 -> 0-1)
    X = X / 255.0
    
    # Use a subset for faster training in this demo environment
    # Full MNIST takes a while. 10k samples is usually enough for decent accuracy.
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=10000, test_size=2000, random_state=42)
    
    print(f"Training MLPClassifier with {len(X_train)} samples...")
    # Simple MLP with one hidden layer of 100 neurons
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=20, alpha=1e-4,
                        solver='sgd', verbose=10, random_state=1,
                        learning_rate_init=.1)
    
    mlp.fit(X_train, y_train)
    
    print(f"Training complete. Score: {mlp.score(X_test, y_test):.4f}")
    
    # Save the model
    joblib.dump(mlp, 'model.pkl')
    print("Model saved to 'model.pkl'")

if __name__ == "__main__":
    train_model()
