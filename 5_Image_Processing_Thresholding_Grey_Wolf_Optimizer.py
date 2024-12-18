# Lab-5
import numpy as np
import cv2
from matplotlib import pyplot as plt

def grey_wolf_optimizer(obj_func, dim, bounds, num_wolves=10, max_iter=100):
    """
    Grey Wolf Optimizer (GWO) for threshold optimization in image processing.

    Parameters:
    - obj_func: Function to optimize.
    - dim: Number of thresholds (e.g., 1 for single threshold).
    - bounds: Tuple specifying (lower_bound, upper_bound) for thresholds.
    - num_wolves: Number of wolves in the pack.
    - max_iter: Maximum number of iterations.

    Returns:
    - best_position: Best solution (threshold value).
    - best_score: Fitness value of the best solution.
    """
    alpha_score = float("inf")
    beta_score = float("inf")
    delta_score = float("inf")

    alpha_pos = np.zeros(dim)
    beta_pos = np.zeros(dim)
    delta_pos = np.zeros(dim)

    lower_bound, upper_bound = bounds
    wolves = np.random.uniform(lower_bound, upper_bound, (num_wolves, dim))

    for t in range(max_iter):
        a = 2 - t * (2 / max_iter)

        for i in range(num_wolves):
            fitness = obj_func(wolves[i])
            if fitness < alpha_score:
                alpha_score, alpha_pos = fitness, wolves[i].copy()
            elif fitness < beta_score:
                beta_score, beta_pos = fitness, wolves[i].copy()
            elif fitness < delta_score:
                delta_score, delta_pos = fitness, wolves[i].copy()

        for i in range(num_wolves):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = abs(C1 * alpha_pos - wolves[i])
            X1 = alpha_pos - A1 * D_alpha

            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = abs(C2 * beta_pos - wolves[i])
            X2 = beta_pos - A2 * D_beta

            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = abs(C3 * delta_pos - wolves[i])
            X3 = delta_pos - A3 * D_delta

            wolves[i] = (X1 + X2 + X3) / 3
            wolves[i] = np.clip(wolves[i], lower_bound, upper_bound)

    return alpha_pos, alpha_score


# Objective function for image thresholding
def image_thresholding_fitness(threshold, image):
    """
    Fitness function to evaluate the quality of a threshold.

    Parameters:
    - threshold: Threshold value for segmentation.
    - image: Input grayscale image.

    Returns:
    - fitness: Measure of segmentation quality (minimized intra-class variance).
    """
    threshold = int(threshold[0])
    foreground = image[image > threshold]
    background = image[image <= threshold]

    if len(foreground) == 0 or len(background) == 0:
        return float("inf")  # Avoid invalid thresholds

    # Calculate intra-class variance
    foreground_var = np.var(foreground)
    background_var = np.var(background)
    fitness = len(foreground) * foreground_var + len(background) * background_var
    return fitness


if __name__ == "__main__":
    
    print("1BM22CS092\t\t Dipesh Sah")
    print("Lab Experiment-5")
    print("Implementation of Image Processing/Thresholding Using Grey Wolf Optimizer.\n")

    image_path = input("Enter the path to the image file: ")
    num_wolves = int(input("Enter the number of wolves: "))
    max_iter = int(input("Enter the maximum number of iterations: "))

    # Read and preprocess the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not read the image file.")
        exit()

    # Show original image
    plt.imshow(image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")
    plt.show()

    # Define bounds for threshold
    lower_bound, upper_bound = 0, 255

    # Run GWO
    best_threshold, best_score = grey_wolf_optimizer(
        obj_func=lambda t: image_thresholding_fitness(t, image),
        dim=1,
        bounds=(lower_bound, upper_bound),
        num_wolves=num_wolves,
        max_iter=max_iter
    )

    # Apply the best threshold
    best_threshold = int(best_threshold[0])
    _, segmented_image = cv2.threshold(image, best_threshold, 255, cv2.THRESH_BINARY)

    # Display results
    print(f"\nBest Threshold: {best_threshold}")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image, cmap="gray")
    plt.title(f"Segmented Image (Threshold: {best_threshold})")
    plt.axis("off")
    plt.show()
