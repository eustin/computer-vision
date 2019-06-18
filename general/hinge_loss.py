import numpy as np

# say we have three training examples with randomly chosen scores.
X = np.array([
    [4.26, 1.33, -1.01],
    [3.76, -1.2, -3.81],
    [-2.37, 1.03, -2.27]
])

# we have scores for three possible classes represented in our images
# y is the column vector containing the true class labels for each of
# the above three examples.
y = np.array([0, 1, 2])

# hinge loss for first example ------------------------------------------------
first_example_scores = X[0]
first_example_label = y[0]
true_label_score = first_example_scores[first_example_label]

# for each class that is not the true class
first_example_loss = 0
for i in range(len(first_example_scores)):
    if i == first_example_label:
        continue
    first_example_loss += max(0,
                              first_example_scores[i] - true_label_score + 1)

print(first_example_loss)

# second example --------------------------------------------------------------
second_example_scores = X[1]
second_example_label = y[1]
true_label_score = second_example_scores[second_example_label]

# for each class that is not the true class
second_example_loss = 0
for i in range(len(second_example_scores)):
    if i == second_example_label:
        continue
    second_example_loss += max(0, second_example_scores[i] - true_label_score + 1)

print(second_example_loss)

# third example ---------------------------------------------------------------
third_example_scores = X[2]
third_example_label = y[2]
true_label_score = third_example_scores[third_example_label]

# for each class that is not the true class
third_example_loss = 0
for i in range(len(third_example_scores)):
    if i == third_example_label:
        continue
    third_example_loss += max(0, third_example_scores[i] - true_label_score + 1)

print(third_example_loss)

# hinge loss for all training examples ----------------------------------------
total_hinge_loss = (first_example_loss + second_example_loss + third_example_loss) / 3.0
print(total_hinge_loss)