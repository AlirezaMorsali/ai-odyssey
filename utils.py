import matplotlib.pyplot as plt
import numpy as np


def custom_pdf(x):
    mean1, std1 = 2, 0.5
    mean2, std2 = 5, 0.5
    mean3, std3 = 8, 0.5

    gauss1 = np.exp(-0.5 * ((x - mean1) / std1) ** 2) / (std1 * np.sqrt(2 * np.pi))
    gauss2 = np.exp(-0.5 * ((x - mean2) / std2) ** 2) / (std2 * np.sqrt(2 * np.pi))
    gauss3 = np.exp(-0.5 * ((x - mean3) / std3) ** 2) / (std3 * np.sqrt(2 * np.pi))

    sinusoidal_modulation = 0.5 * (1 + np.sin(2 * np.pi * x / 10))

    return (gauss1 + gauss2 + gauss3) * sinusoidal_modulation
    # return gauss1 + gauss2 + gauss3


def normalize_pdf(pdf_func, x_range, num_points=1000):
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    pdf_values = np.array([pdf_func(x) for x in x_values])
    normalization_constant = np.trapezoid(pdf_values, x_values)
    return lambda x: pdf_func(x) / normalization_constant


def generate_random_numbers(pdf_func, x_range, num_samples=1000):
    normalized_pdf = normalize_pdf(pdf_func, x_range)
    max_pdf_value = max(
        [normalized_pdf(x) for x in np.linspace(x_range[0], x_range[1], 1000)]
    )

    samples = []
    while len(samples) < num_samples:
        x = np.random.uniform(x_range[0], x_range[1])
        y = np.random.uniform(0, max_pdf_value)
        if y <= normalized_pdf(x):
            samples.append(x)

    return np.array(samples)


# Step 4: Plot the distribution of generated data
def plot_distribution(samples, pdf_func, x_range, num_bins=1000):
    plt.figure(figsize=(10, 6))

    # Histogram of generated samples
    plt.hist(
        samples,
        bins=num_bins,
        density=True,
        alpha=0.6,
        color="g",
        label="Generated Data",
    )

    # Plot the original PDF
    x_values = np.linspace(x_range[0], x_range[1], 1000)
    pdf_values = [pdf_func(x) for x in x_values]
    normalization_constant = np.trapz(pdf_values, x_values)
    pdf_values = [val / normalization_constant for val in pdf_values]
    plt.plot(x_values, pdf_values, "r-", lw=2, label="PDF")

    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Generated Data Distribution vs PDF")
    plt.show()


if __name__ == "__main__":
    x_range = (0, 10)
    num_samples = 10000
    samples = generate_random_numbers(custom_pdf, x_range, num_samples)
    plot_distribution(samples, custom_pdf, x_range)
