import numpy as np


def _phi_stable_batch_epsilon( posterior_probs, labels, epsilon=1e-10):
    posterior_probs = posterior_probs + epsilon
    one_hot_labels = np.zeros_like(posterior_probs)

    one_hot_labels[np.arange(len(labels)), labels] = 1

    # Calculate the log likelihood for the correct labels
    log_likelihood_correct = np.log(posterior_probs[np.arange(len(labels)), labels])

    # Calculate the sum of posterior probabilities for all incorrect labels
    sum_incorrect = np.sum(posterior_probs * (1 - one_hot_labels), axis=1)

    # Replace any zero values with a very small number to prevent division by zero in log
    # Calculate the log likelihood for the incorrect labels
    log_likelihood_incorrect = np.log(sum_incorrect)

    # Calculate phi_stable for each example
    phi_stable = log_likelihood_correct - log_likelihood_incorrect

    return phi_stable