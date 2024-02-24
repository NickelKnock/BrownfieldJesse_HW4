import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
"""
A fancy, nice little problem statement:

P(x<1|N(0,1)) 
    - the probability that x is less than 1
    - (mean) μ = 0 and (standard deviation) σ = 1
P(x>μ+2σ|N(175, 3))
    - the probability that x is greater than the mean + 2 times σ
    - (mean) μ = 175 and (standard deviation) σ = 3
We are going to plot this using some libraries. Specifically, scipy stats, numpy,
and matplotlib

We should "explore" stats.norm().pdf and stat.norm().cdf 
"""


# Function to calculate the statistics
def statsMcStatFace():
    """
    Calculate the PDF and CDF for our two P()

    I opted to make this modular by making the calculations one function and the plotting functions another. This makes
    it not only more readable but allows me to call the functions in the future if I need to without the need to
    duplicate the function.

    I moved things around from my initial program to make sure I was doing this after realizing I had plots and
    calculations all scattered around when trying to program after an already 16-hour work day. After a solid "oh, duh"
    moment and lots of cleaning up it should make a lot of sense.

    Here we use the CDF/PDF functions to find the probabilities associated with the range of values in the
    distribution. I felt like the function name was awesome and appropriate since we are finding the statistical
    bits here.
    """
    # For the first distribution N(0,1)
    mu0, sigma0 = 0, 1
    # For the second distribution N(175,3)
    mu1, sigma1 = 175, 3

    # Generating points for the second distribution N(1,0)
    # setting the range from -4σ to 4σ since that is very nearly %100 of the distribution
    # and infinity wouldn't fit on the graph, using 1000 steps for resolution
    x0 = np.linspace(mu0 - 4 * sigma0, mu0 + 4 * sigma0, 1000)
    x1 = np.linspace(mu1 - 4 * sigma1, mu1 + 4 * sigma1, 1000)

    # Calculate the PDF for each distribution
    pdf0 = norm.pdf(x0, mu0, sigma0)
    pdf1 = norm.pdf(x1, mu1, sigma1)

    # Calculate the CDF for each distribution
    cdf0 = norm.cdf(x0, mu0, sigma0)
    cdf1 = norm.cdf(x1, mu1, sigma1)

    # Return the calculated values
    return (x0, pdf0, cdf0), (x1, pdf1, cdf1)


def plotMcPlotFace(stats0, stats1):
    """
    When I found this: https://www.tutorialspoint.com/matplotlib/matplotlib_axes_class.htm
    I realized I was taking an overly complicated route to sort out my plotting. The rest of it fell in line after that
    Plot the PDF and CDF using the data in the stats function.

    Using the axes objects from the matplot allows more gradual control over each plot without as much fussing with
    the subplots as much.

    As instructed we don't return any values but just the plots of each.
    """
    # a set of tuples with their respective values
    (x0, pdf0, cdf0), (x1, pdf1, cdf1) = stats0, stats1

    # Create the plots with a uniform size in a 2 x 2 format
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot the PDF for N(0,1) and shade the area for x < 1 in the 0,0 position
    axes[0, 0].plot(x0, pdf0, 'k')
    axes[0, 0].fill_between(x0, pdf0, where=(x0 < 1), color='skyblue', alpha=0.5)
    axes[0, 0].set_title('N(0,1) PDF with shaded area for x < 1')

    # Plot the CDF for N(0,1) and mark the point for x = 1 in the 1,0 position
    axes[1, 0].plot(x0, cdf0, 'b')
    axes[1, 0].scatter([1], norm.cdf(1, 0, 1), marker='H')
    axes[1, 0].set_title('N(0,1) CDF with point marked for x = 1')

    # Plot the PDF for N(175,3) and shade the area for x > mu1 + 2*sigma1 in the 0,1 position
    axes[0, 1].plot(x1, pdf1, 'r')
    axes[0, 1].fill_between(x1, pdf1, where=(x1 > 175 + 2 * 3), color='pink', alpha=0.5)
    axes[0, 1].set_title('N(175,3) PDF with shaded area for x > μ + 2σ')

    # Plot the CDF for N(175,3) and mark the point for x = mu1 + 2*sigma1 in the 1,1 position
    axes[1, 1].plot(x1, cdf1, 'r')
    axes[1, 1].scatter([175 + 2 * 3], norm.cdf(175 + 2 * 3, 175, 3), marker='H')
    axes[1, 1].set_title('N(175,3) CDF with point marked for x = μ + 2σ')

    # Adjusting the layout to avoid overlap
    plt.tight_layout()
    plt.show()


# running the statistic code and grabbing the values
stats0, stats1 = statsMcStatFace()

# Plotting the results
plotMcPlotFace(stats0, stats1)
