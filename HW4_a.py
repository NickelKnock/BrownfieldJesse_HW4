import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

"""
A fancy, nice little problem statement:

P(x<1|N(0,1)) 
    - the probability that x is less than 1
    - (mean) μ = 0 and (standard deviation) σ = 1
P(x>μ+2σ|N(175, 3))
    - the probability that x is greater than the mean + 2 times σ
    - (mean) μ = 175 and (standard deviation) σ = 3
We are going to plot this using some libraries. Specifically, scipy stats
and matplotlib

We should "explore" stats.norm().pdf and stat.norm().cdf 

Not a single line of this code was copied from chatGPT

"""


# Getting out the coffee maker because this is my 13th 16-hour day in a row
def statsMcStatFace():
    """
    Here we use the CDF function to find the probabilities associated with the range of values in the
    distribution. I felt like the function name was awesome and appropriate since we are finding the statistical
    bits here.
    :return:
    prob1- the mean and std deviation for the first problem
    prob2 - the mean and std deviation for the second problem
    """
    mu, sigma = 0, 1  # μ = 0, σ = 1 look fancy unicode characters. I wanted to put them on the same line so I did

    # Going to rock out some probability for x < 1 now
    prob1 = norm.cdf(1, mu, sigma)

    # ...moving on to the other one
    # just updating these should be fine since we are going to rename the output
    mu, sigma = 175, 3

    # using 1-CDF(value)since it should be less than 1 because certainty has no
    # place in statistics...
    prob2 = 1 - norm.cdf(mu + 2 * sigma, mu, sigma)

    return prob1, prob2, mu, sigma


# A function to plot the distributions
def plotMcPlotFace(mu0=0, sigma0=1, mu1=175, sigma1=3, prob1=None, prob2=None):
    """
     Generating some points for the first distribution, setting the range for the graph
     then we go to the actual plotting of the graph, drawing two subplots, so they appear
     at the same time since cycling didn't really display the data in a matter than contrasted them
     very well. We call prob1 and 2 as none, so we don't break anything in testing the graphs
    """
    # Generating points for the second distribution N(1,0)
    # setting the range from -4σ to 4σ since that is very nearly %100 of the distribution
    # and infinity wouldn't fit on the graph, using 1000 steps for resolution
    x0 = np.linspace(mu0 - 4 * sigma0, mu0 + 4 * sigma0, 1000)
    # finding the PDF values for these x guys
    y0 = norm.pdf(x0, mu0, sigma0)

    # Generating points for the second distribution N(175,3)
    x1 = np.linspace(mu1 - 4 * sigma1, mu1 + 4 * sigma1, 1000)
    # now, we find the pdf for these x guys
    y1 = norm.pdf(x1, mu1, sigma1)

    # Plotting the first distribution with shading for x < 1
    # This figure sizes fits nicely on the screen
    # if you have two double wide monitors you can use figsize(100, 10) to really get a good look at it
    plt.figure(figsize=(12,8))
    # First subplot, parameterizing the rows and columns with respect to our desired output
    plt.subplot(1, 2, 1)
    # x0, y0 are the array values for the plot centered on mu0, and sigma0
    # I picked green because it was something other than the standard blue and got fancy with how to label things
    # the value of mu0 and sigma0 instead of 173 and 3 respectively
    plt.plot(x0, y0,'g', label=f'N({mu0},{sigma0})')
    plt.fill_between(x0, 0, y0, where=(x0 < 1), color='lightgreen', alpha=0.5, label=f'P(x < 1) = {prob1:.4f}')
    plt.legend()
    plt.title("N(0,1) Distribution")
    plt.xlabel("x")
    plt.ylabel("Probability Density")

    # Plotting the second distribution with shading for x > mu + 2sigma
    plt.subplot(1, 2, 2)  # Second subplot, parameterizing the rows and columns with respect to our desired output
    plt.plot(x1, y1, 'r', label=f'N({mu1},{sigma1})')
    # This shades the area under the graph for those mu and sigma values in the range we called for in
    # the problem statement. The alpha shouldn't be confused with a confidence level, that's a
    # transparency setting.
    # I got a good laugh trying "lightred" first only to realize that's "pink"
    plt.fill_between(x1, 0, y1, where=(x1 > mu1 + 2 * sigma1), color='pink', alpha=0.5,
                     label=f'P(x > μ + 2σ) = {prob2:.4f}')

    # Producing a legend that displays on the graph
    plt.legend()
    # giving a graph title
    plt.title("N(175,3) Distribution")
    # since we are comparing x, just "x" is a good x-axis title
    plt.xlabel("x")
    # this fella helped fix some overlap
    plt.tight_layout()
    # just waking up the graphs
    plt.show()

    # executing the main function in the code
if __name__ == "__main__":

    prob1, prob2, _, _ = statsMcStatFace()
    # making sure we use everything we went through the trouble of finding using the prob1, and prob2 values for
    # plotting
    plotMcPlotFace(mu0=0, sigma0=1, mu1=175, sigma1=3, prob1=prob1, prob2=prob2)
