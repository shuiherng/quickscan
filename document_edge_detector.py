from math import sin, cos, atan
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from itertools import combinations

class DocumentEdgeDetector:
	def __init__(self, rho=2, theta=360, threshold=100, output_process=False):
		"""
		rho: distance resolution of accumulator
		theta: angle resolution of accumulator
		threshold: accumulator threshold (curve intersections above this 
					will be returned)
		"""
		self.rho = rho
		self.theta = theta
		self.threshold = threshold
		self.output_process = output_process

	def __call__(self, image):
		lines = self.get_hough_lines(image)
		intersections = self.get_intersections(image, lines)
		quadrilaterals = self.find_quadrilaterals(image, lines, intersections)
		return quadrilaterals

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
		return topleft, topright, bottomleft, bottomright

	@classmethod
	def overlay_document_bounding_boxes(cls, image, quad):
		output = image.copy()
		topleft, topright, bottomleft, bottomright = cls.order_points(quad)
		cv2.line(
				output, 
				topleft,
				topright,
				(255, 0, 0), 
				3
			)
		cv2.line(
				output, 
				topright,
				bottomright,
				(255, 0, 0), 
				3
			)
		cv2.line(
				output, 
				topleft,
				bottomleft,
				(255, 0, 0), 
				3
			)
		cv2.line(
				output, 
				bottomleft,
				bottomright,
				(255, 0, 0), 
				3
			)
		return output

	def set_force_output_process(self):
		self.output_process = True

	def get_hough_lines(self, image):
		lines = cv2.HoughLines(image, self.rho, np.pi/self.theta, self.threshold)
		if self.output_process:
			self.draw_hough_lines(image, lines)
		return lines

	def draw_hough_lines(self, image, lines):
		hough_line_output = image
		for line in lines:
			rho, theta = line[0]
			a, b = np.cos(theta), np.sin(theta)
			x0, y0 = a * rho, b * rho
			n = 5000
			x1 = int(x0 + n * (-b))
			y1 = int(y0 + n * (a))
			x2 = int(x0 - n * (-b))
			y2 = int(y0 - n * (a))

			cv2.line(
				hough_line_output, 
				(x1, y1), 
				(x2, y2), 
				(255, 255, 255), 
				2
			)
		cv2.imwrite('output/hough_lines.jpg', hough_line_output)

	def get_intersections(self, image, lines):
		intersections = []
		group_lines = combinations(range(len(lines)), 2)
		x_in_range = lambda x: 0 <= x <= image.shape[1]
		y_in_range = lambda y: 0 <= y <= image.shape[0]

		for i, j in group_lines:
			line_i, line_j = lines[i][0], lines[j][0]
			
			if 80.0 < self.get_angle_between_lines(line_i, line_j) < 100.0:
				int_point = self.intersection(line_i, line_j)
				
				if x_in_range(int_point[0][0]) and y_in_range(int_point[0][1]): 
					intersections.append(int_point)

		if self.output_process:
			self.draw_intersections(image, lines, intersections)

		return intersections

	def find_quadrilaterals(self, image, lines, intersections):
		X = np.array([[point[0][0], point[0][1]] for point in intersections])
		kmeans = KMeans(
			n_clusters = 4, 
			init = 'k-means++', 
			max_iter = 100, 
			n_init = 10, 
			random_state = 0
		).fit(X)

		if self.output_process:
			self.draw_quadrilaterals(image, lines, kmeans)

		return  [[center.tolist()] for center in kmeans.cluster_centers_]

	def get_angle_between_lines(self, line_1, line_2):
		rho1, theta1 = line_1
		rho2, theta2 = line_2
		# x * cos(theta) + y * sin(theta) = rho
		# y * sin(theta) = x * (- cos(theta)) + rho
		# y = x * (-cos(theta) / sin(theta)) + rho
		m1 = -(np.cos(theta1) / np.sin(theta1))
		m2 = -(np.cos(theta2) / np.sin(theta2))
		return abs(atan(abs(m2-m1) / (1 + m2 * m1))) * (180 / np.pi)

	def intersection(self, line1, line2):
		"""Finds the intersection of two lines given in Hesse normal form.
		Returns closest integer pixel locations.
		See https://stackoverflow.com/a/383527/5087436
		"""
		rho1, theta1 = line1
		rho2, theta2 = line2

		A = np.array([
			[np.cos(theta1), np.sin(theta1)],
			[np.cos(theta2), np.sin(theta2)]
		])

		b = np.array([[rho1], [rho2]])
		x0, y0 = np.linalg.solve(A, b)
		x0, y0 = int(np.round(x0)), int(np.round(y0))
		return [[x0, y0]]


	def draw_intersections(self, image, lines, intersections):
		intersection_point_output = image

		for line in lines:
			rho, theta = line[0]
			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a * rho
			y0 = b * rho
			n = 5000
			x1 = int(x0 + n * (-b))
			y1 = int(y0 + n * (a))
			x2 = int(x0 - n * (-b))
			y2 = int(y0 - n * (a))

			cv2.line(
				intersection_point_output, 
				(x1, y1), 
				(x2, y2), 
				(0, 0, 255), 
				2
			)

		for point in intersections:
			x, y = point[0]

			cv2.circle(
				intersection_point_output,
				(x, y),
				5,
				(255, 255, 127),
				5
			)

		cv2.imwrite('output/intersection_point_output.jpg', intersection_point_output)

	def draw_quadrilaterals(self, image, lines, kmeans):
		grouped_output = image

		for idx, line in enumerate(lines):
			rho, theta = line[0]
			a, b = np.cos(theta), np.sin(theta)
			x0, y0 = a * rho, b * rho
			n = 5000
			x1 = int(x0 + n * (-b))
			y1 = int(y0 + n * (a))
			x2 = int(x0 - n * (-b))
			y2 = int(y0 - n * (a))

			cv2.line(
				grouped_output, 
				(x1, y1), 
				(x2, y2), 
				(0, 0, 255), 
				2
			)
		
		for point in kmeans.cluster_centers_:
			x, y = point

			cv2.circle(
				grouped_output,
				(int(x), int(y)),
				5,
				(255, 255, 255),
				5
			)

		cv2.imwrite('output/grouped.jpg', grouped_output)