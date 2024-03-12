import subprocess
import matplotlib.pyplot as plt

# Define the file name and the number of runs
file_names = ['baseline.py','supervised_cnn.py']
num_runs = 5  # Adjust the number of runs as needed

# Store test accuracies from each run
all_accuracies = []

for file_name in file_names:
    test_accuracies = []
# Run the script multiple times and accumulate test accuracies
    for _ in range(num_runs):
        output = subprocess.check_output(['python', file_name]).decode('utf-8')
        print("Output from the script:\n", output)  # Print the output of the script

        # Find the line containing the final accuracy information
        accuracy_line = [line for line in output.split('\n') if 'Accuracy on the test data' in line][-1]
        print("Accuracy line:", accuracy_line)  # Print the accuracy line

        # Extract the accuracy value from the line
        accuracy = float(accuracy_line.split('=')[1].strip())
        print("Extracted accuracy:", accuracy)  # Print the extracted accuracy

        # Append the accuracy to the list
        test_accuracies.append(accuracy)
    
    all_accuracies.append(test_accuracies)
# Create a boxplot of the test accuracies
plt.figure(figsize=(10, 6))
plt.boxplot(all_accuracies, whis=1.5)
plt.xticks([1, 2], file_names)  # Set x-axis labels as file names
plt.title('Test Accuracies Comparison')
plt.xlabel('Files')
plt.ylabel('Test Accuracy')
plt.show()
