
import numpy as np
import seaborn as sns

np.random.seed(123)
sns.set_style('darkgrid')

# cross-entropy intuition.
# when finding cross-entropy loss, we focus on the normalised probability
# assigned to the true class.
# this means that our probabilities will fall in the range [0, 1]
probabilities = np.arange(0, 1, 0.001) + 0.001

# taking the negative log gives us the cross-entropy loss
negative_log = -np.log(probabilities)

# as the probability associated with the true class increases (the x-axis),
# we see a decrease in cross-entropy loss
# as this same probability decreases, we see a sharp increase in
# cross-entropy loss
negative_log_plot = sns.lineplot(x=probabilities, y=negative_log)
negative_log_plot.set_title('Visualising cross-entropy loss')
negative_log_plot.set_xlabel('Probabilities')
negative_log_plot.set_ylabel('Cross-entropy loss')


# worked example --------------------------------------------------------------

# index of score of true class of training example
true_class_idx = 2

# example where class label correctly predicted
# true_class_idx == 2. array idx == 2 contains large
# unnormalised log probability.
#
# results in cross-entropy loss << 0
unnormalised_log_probs = np.array([1, 1, 10])

# exponentiate log probabilities to recover probabilities.
unnormalised_probs = np.exp(unnormalised_log_probs)

# normalise probabilities so that they sum to one. this is the
# softmax classifier
normalised_prob = unnormalised_probs[2] / np.sum(unnormalised_probs)
print(normalised_prob)

# take the negative log of the probability for the true class to
# recover the loss. negative log because a smaller probability
# results in an exponentially greater loss.
cross_entropy_loss = -np.log(normalised_prob)
print(cross_entropy_loss)

# example where class label incorrectly predicted
# cross-entropy loss is far greater at ~9.0
unnormalised_log_probs = np.array([1, 10, 1])
unnormalised_probs = np.exp(unnormalised_log_probs)

normalised_prob = unnormalised_probs[2] / np.sum(unnormalised_probs)
print(normalised_prob)
cross_entropy_loss = -np.log(normalised_prob)
print(cross_entropy_loss)
