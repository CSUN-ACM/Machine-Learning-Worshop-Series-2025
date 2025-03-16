import numpy as np
import matplotlib.pyplot as plt

def distance(x1, x2):
    return np.sqrt(np.sum((np.array(x1) - np.array(x2))**2))

def pred(input_data, labels, point, k):
    classifier = []
    
    for i in range(len(input_data)):
        dist = distance(input_data[i], point)
        classifier.append((dist, labels[i]))
    
    classifier.sort()  # Sort by distance
    
    label_1 = 0
    label_2 = 0
    
    for i in range(k):
        if classifier[i][1] == 1:
            label_1 += 1
        else:
            label_2 += 1
    
    return "Class 1" if label_1 > label_2 else "Class 2"

# Generate test dataset
np.random.seed(0)
class_1 = np.random.randn(10, 2) + np.array([2, 2])
class_2 = np.random.randn(10, 2) + np.array([-2, -2])

X = np.vstack((class_1, class_2))
y = np.array([1] * 10 + [0] * 10)

# Test the function with a sample point
test_point = np.array([0, 0])
k = 3
prediction = pred(X, y, test_point, k)
print("Predicted class:", prediction)

# Plot the dataset
plt.scatter(class_1[:, 0], class_1[:, 1], color='blue', label="Class 1")
plt.scatter(class_2[:, 0], class_2[:, 1], color='red', label="Class 2")
plt.scatter(test_point[0], test_point[1], color='green', marker='x', s=100, label="Test Point")
plt.legend()
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("KNN Classification Example")
plt.show()

        
    
    
