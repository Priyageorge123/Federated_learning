import subprocess
import matplotlib.pyplot as plt

# Define the file name and the number of runs
file_name = 'supervised_resnet.py'
num_runs = 5  # Adjust the number of runs as needed

# List to store test accuracies from each run

test_accuracies1 = []
test_accuracies2 = []
# Run the script multiple times and accumulate test accuracies
for _ in range(num_runs):
    output = subprocess.check_output(['python', file_name]).decode('utf-8')
    print("Output from the script:\n", output)  # Print the output of the script
    parts = output.split('Finished Training')

    for i, part in enumerate(parts[1:], start=1):  # Start from index 1 to skip the first part before the first "Finished Training"
    # Find the line containing the final accuracy information in this part
        accuracy_line = [line for line in part.split('\n') if 'Accuracy on the test data' in line][-1]
        print("Accuracy line:", accuracy_line)  # Print the accuracy line

        # Extract the accuracy value from the line
        accuracy = float(accuracy_line.split('=')[1].strip())
        #print("Extracted accuracy:", accuracy)  # Print the extracted accuracy

        # Append the accuracy to the list
        if i == 1:
            test_accuracies1.append(accuracy)
        else:
            test_accuracies2.append(accuracy)

test_accuracies3 = [0.3771,0.3733, 0.3765,0.3718, 0.3697, 0.3729, 0.3751, 0.379, 0.3702, 0.3757]
print(test_accuracies1,test_accuracies2)  

# Create a boxplot of the test accuracies
plt.figure(figsize=(10, 6))
plt.boxplot(test_accuracies2,test_accuracies3,test_accuracies1, whis=1.5)
plt.xticks([1, 2], ['One-third-data', 'Federated(two-third)','Full_data'])  # Set x-axis labels as file names
plt.title('Test Accuracies Comparison')
plt.xlabel('Files')
plt.ylabel('Test Accuracy')
#plt.show()
# Save the plot to a file
plt.savefig('test_accuracies_plot.png')

# Close the plot to release resources
plt.close()
