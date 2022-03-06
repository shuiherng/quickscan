import cv2
import numpy as np

class PageExtractor:
	def __init__(self, output_process=False):
		self.output_process = output_process
		self.src = None
		self.dst = None
		self.m = None
		self.max_width = None
		self.max_height = None

	def __call__(self, image, quad):
		warped = image.copy()

		rect = PageExtractor.order_points(quad)
		(tl, tr, br, bl) = rect
		rect = np.array(rect, dtype = "float32")

		# compute the width of the new image, which will be the
		# maximum distance between bottom-right and bottom-left
		# x-coordiates or the top-right and top-left x-coordinates
		widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
		widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
		maxWidth = max(int(widthA), int(widthB))

		# compute the height of the new image, which will be the
		# maximum distance between the top-right and bottom-right
		# y-coordinates or the top-left and bottom-left y-coordinates
		heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
		heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
		maxHeight = max(int(heightA), int(heightB))

		# now that we have the dimensions of the new image, construct
		# the set of destination points to obtain a "birds eye view",
		# (i.e. top-down view) of the image, again specifying points
		# in the top-left, top-right, bottom-right, and bottom-left
		# order
		dst = np.array([
			[0, 0],                         # Top left point
			[maxWidth - 1, 0],              # Top right point
			[maxWidth - 1, maxHeight - 1],  # Bottom right point
			[0, maxHeight - 1]],            # Bottom left point
			dtype = "float32"               # Date type
		)

		# compute the perspective transform matrix and then apply it
		M = cv2.getPerspectiveTransform(rect, dst)
		warped = cv2.warpPerspective(warped, M, (maxWidth, maxHeight))

		if self.output_process:
			cv2.imwrite('output/deskewed.jpg', warped)

		self.src = rect
		self.dst = dst
		self.m = M
		self.max_width = maxWidth
		self.max_height = maxHeight

		# for the entire image, not just the height of the extracted quadrilateral
		self.src_img_dims = image.shape

		return warped

	def reverse_skew(self, image):
		print(self.src_img_dims)
		m_inv = cv2.getPerspectiveTransform(self.dst, self.src)
		unskewed = cv2.warpPerspective(image, m_inv, (self.src_img_dims[1], self.src_img_dims[0]))

		return unskewed

	@classmethod
	def order_points(cls, quad):
		summed = [sum(p[0]) for p in quad]
		topleft = quad[summed.index(min(summed))][0]
		bottomright = quad[summed.index(max(summed))][0]

		remain = []
		for i, x in enumerate(quad):
			if not i in [summed.index(min(summed)), summed.index(max(summed))]:
				remain.append(x[0])

		# bottom left has steeper gradient from topleft
		grads = [abs((topleft[1]-y)/(topleft[0]-x+1e-12)) for x,y in remain]
		bottomleft = remain[grads.index(max(grads))]
		topright = remain[grads.index(min(grads))]

		topleft = [c for c in map(int, topleft)]
		topright = [c for c in map(int, topright)]
		bottomleft = [c for c in map(int, bottomleft)]
		bottomright = [c for c in map(int, bottomright)]

		topleft = tuple(topleft)
		topright = tuple(topright)
		bottomleft = tuple(bottomleft)
		bottomright = tuple(bottomright)

		return topleft, topright, bottomright, bottomleft 

	def set_force_output_process(self):
		self.output_process = True
		
	