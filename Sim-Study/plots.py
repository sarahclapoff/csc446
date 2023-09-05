import numpy as np
import matplotlib.pyplot as plt


TOLERANCE = 5


def calcViewerScore(num_viewers):
    retval = 100/(np.maximum(np.sqrt(num_viewers) * 2/TOLERANCE, 1))

    return retval

def calcQualityScore(quality):
    # Higher this is the lower the slope in the middle
    SLOPE_CONSTANT = 10
    ## SHouldn't edit this, the midpoint of the inverse sigmoid is 0.5 at MIDPOINT_HEIGHT of 1. Instead adjust the QUALITY_CONSTANT
    ## Like you can edit this but you shouldn't need to unless you're doing fancy stuff
    MIDPOINT_HEIGHT = 1

    ## For quality we want to have low quality paintings be very undesirable and high quality very desirable but 25-75 should have less of an effect
    ## We can do this by using an inverse sigmoid function
    retval =  (0.5 - np.log((1-quality)/(quality * MIDPOINT_HEIGHT))/SLOPE_CONSTANT) * 100

    return retval


### Plot for viewer score
# x = np.linspace(0,400,200)
# y = calcViewerScore(x)

# plt.plot(x,y)
# plt.xlabel('Number of Viewers')
# plt.ylabel('Score')
# plt.title("Score from number of viewers, tolerance = 5")

# plt.savefig('viewer_score.png')
# plt.show()




x = np.linspace(0, 1, 100)
y = calcQualityScore(x)
plt.plot(x,y)
plt.xlabel('Painting Quality')
plt.ylabel('Score')
plt.title("Score from painting quality")
plt.ylim(0,100)

# plt.savefig('quality_score.png')
# plt.show()

