# Fiber-Detection
Automatically detects straight fibers in a greyscale image, even with moderate noise and overlaps.

This program uses a filter to find points in the input image which are likely to be centered on a fiber,
and creates a distance matrix of the distance from each pixel to the closest of those points.
The local maximums of this distance matrix typically lie very close to, if not directly on, the center of the fibers in the image.
After some winnowing of the points to decrease the chance of cycles, points near to each other are linked together,
forming a graph structure. The nodes which have only one link are assumed to be the endpoint of a fiber/fiber segment.
Starting at each endpoint, the links of each node are traced in the most straight line possible, until there are no linked points ahead (happens if either the line takes a sudden curve, or when you hit another endpoint). These groups of linked points are saved as fiber segments. The same (roughly) process is then applied to the fiber segments; nearby fibers which are travelling in about the same direction are linked together, and then the links are followed, tracing out full fibers.

It should be O(n), where n is the nunmber of fibers in the image.
While it's (I'm pretty sure) a linear time algorithm, it does have a rather large constant factor.

