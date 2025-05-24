# Quantum Machine Learning Classification with QSVM
# This script generates a synthetic dataset, uses a quantum kernel for feature mapping,
# trains a Support Vector Machine, and plots the decision boundary.

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit.library import ZZFeatureMap
import matplotlib.pyplot as plt

def create_dataset(samples=100, features=2):
    """Generate a simple dataset for binary classification."""
    data, labels = make_classification(
        n_samples=samples,
        n_features=features,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=42
    )
    return data, labels

def build_quantum_kernel(data):
    """Create a quantum kernel matrix using ZZFeatureMap for QSVM."""
    num_samples = len(data)
    num_features = data.shape[1]
    
    # Set up quantum feature map and simulator
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2, entanglement='linear')
    simulator = AerSimulator()
    
    # Initialize kernel matrix
    kernel = np.zeros((num_samples, num_samples))
    
    # Compute kernel values by measuring overlap between quantum states
    for i in range(num_samples):
        for j in range(num_samples):
            circuit = QuantumCircuit(num_features)
            map_i = feature_map.bind_parameters(data[i])
            map_j = feature_map.bind_parameters(data[j])
            circuit.compose(map_i, inplace=True)
            circuit.compose(map_j.inverse(), inplace=True)
            circuit.measure_all()
            
            # Run simulation and calculate kernel value
            job = simulator.run(circuit, shots=1000)
            counts = job.result().get_counts()
            kernel_value = counts.get('0' * num_features, 0) / 1000
            kernel[i, j] = kernel_value
    
    return kernel

def plot_results(data, labels, model, output_file):
    """Visualize the QSVM decision boundary."""
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    test_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Note: Simplified prediction for visualization (recompute kernel for test points in production)
    predictions = model.predict(build_quantum_kernel(test_points))
    predictions = predictions.reshape(xx.shape)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, predictions, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title('Quantum SVM Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig(output_file)
    plt.close()

def main():
    # Create and save dataset
    data, labels = create_dataset(samples=100, features=2)
    dataset = pd.DataFrame(data, columns=['Feature1', 'Feature2'])
    dataset['Label'] = labels
    dataset.to_csv('sample_data.csv', index=False)
    print("Dataset generated and saved as 'sample_data.csv'.")
    
    # Compute quantum kernel
    print("Computing quantum kernel matrix...")
    kernel = build_quantum_kernel(data)
    
    # Train QSVM
    print("Training Quantum SVM...")
    svm = SVC(kernel='precomputed')
    svm.fit(kernel, labels)
    
    # Check accuracy
    predictions = svm.predict(kernel)
    accuracy = accuracy_score(labels, predictions)
    print(f"Classification Accuracy: {accuracy:.2f}")
    
    # Plot decision boundary
    print("Generating decision boundary plot...")
    plot_results(data, labels, svm, 'decision_boundary.png')
    print("Plot saved as 'decision_boundary.png'.")

if __name__ == "__main__":
    main()