# Fiber-Detection
Automatically detects straight fibers in a greyscale image, even with moderate noise and overlaps.

This program uses a filter to find points in the input image which are likely to be centered on a fiber,
and creates a distance matrix of the distance from each pixel to the closest of those points.
The local maximums of this distance matrix typically lie very close to, if not directly along, the centerline of each fiber in the image. Thus the image is reduced to a binary image made up of lines where the fibers used to be.

After some winnowing of the points to decrease the chance of cycles, points near to each other are linked together,
forming a graph structure. The nodes which have only one link are assumed to be the endpoint of a fiber/fiber segment.
Starting at each endpoint, the links of each node are traced in the most straight line possible, until there are no linked points ahead (happens if either the line takes a sudden curve, or when you hit another endpoint). These groups of linked points are saved as fiber segments. A similar process is then applied to the fiber segments; nearby fibers which are travelling in about the same direction are linked together, and then the links are followed in as straight a line as possible, tracing out full fibers.

The filtering process is quite fast, using C libraries.

The first graph-creation is relatively fast. Is O(n).

The second graph-creation is relatively slow, but bearable even with large images. Is O(n*log(n)).


Current issues:

-Output is fragmented and incomplete; the filtering process doesn't detect points of intersection between fibers well.
	Either make up for the inaccuracy by correctly reconnecting segments later on,
	Or change the filter to detect both regular fibers as well as intersections
	Or supplement the existing filter with another one which only detects intersections.

-Memory requirements; processing full images (~20,000x20,000 pixels) takes more than the 12GB of RAM I have available.



