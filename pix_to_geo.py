from loguru import logger
import math
import numpy as np
import pymap3d
import pymap3d.utils
import pyproj
import pyproj.enums
import rasterio
import rasterio.windows
from scipy.spatial.transform import Rotation


def pixel_to_geocoord(
	dataset: rasterio.DatasetReader,
	data_to_geo: pyproj.pyproj.Transformer,
	pixel: tuple[int, int]
) -> tuple[float, float] | tuple[float, float, float]:
	'''Convert a (row, column) pixel to (longitude, latitude)'''
	return data_to_geo.transform(*dataset.xy(*pixel))


def geocoord_to_pixel(
	dataset: rasterio.DatasetReader,
	data_to_geo: pyproj.pyproj.Transformer,
	coord: tuple[float, float] | tuple[float, float, float]
) -> tuple[int, int]:
	'''Convert a (longitude, latitude) position to a (row, column) pixel'''
	data_crs_pos = data_to_geo.transform(*coord, direction=pyproj.enums.TransformDirection.INVERSE)
	return dataset.index(*data_crs_pos)


def sample_map_geocoord(
	dataset: rasterio.DatasetReader,
	data_to_geo: pyproj.pyproj.Transformer,
	coord: tuple[float, float],
	interpolation = True
) -> tuple[float, tuple[float, float]] | tuple[None, None]:
	'''
	Get the value of a pixel in a map at the specified geographic coordinate, or the interpolated value of the pixels
	surrounding a fractional coordinate.
	'''

	data_crs_pos = data_to_geo.transform(*coord, direction=pyproj.enums.TransformDirection.INVERSE)

	if not interpolation:
		values = list(dataset.sample([data_crs_pos]))

		if len(values) == 0:
			return (None, None)

		return (values[0][0], data_crs_pos)
	else:
		row, col = dataset.index(*data_crs_pos)
		window = rasterio.windows.Window(col, row, 2, 2)

		if row == dataset.height or col == dataset.width:
			return dataset.read(1, window=window, boundless=True)[0][0]

		values = dataset.read(1, window=window)

		if len(values) == 0:
			return (None, None)

		top_left_xy = dataset.xy(row, col)

		x_fraction = (data_crs_pos[0] - top_left_xy[0]) / (data_crs_pos[0] - top_left_xy[0])
		y_fraction = (data_crs_pos[1] - top_left_xy[1]) / (data_crs_pos[1] - top_left_xy[1])

		if len(values) == 1:
			alt = np.interp(x_fraction, [0, 1], values[0])
		elif len(values[0]) == 1:
			alt = np.interp(y_fraction, [0, 1], [values[0][0], values[1][0]])
		else:
			row0 = np.interp(x_fraction, [0, 1], values[0])
			row1 = np.interp(x_fraction, [0, 1], values[1])
			alt = np.interp(y_fraction, [0, 1], [row0, row1])

		return (alt, data_crs_pos)

		#return values[0][0] * (1 - x_fraction) * (1 - y_fraction) + values[0][1] * x_fraction * (1 - y_fraction) \
		#    + values[1][0] * (1 - x_fraction) * y_fraction + values[1][1] * x_fraction * y_fraction


def raycast_elevation_map(
	position: tuple[float, float, float],
	rotation: Rotation,
	elevation_data: rasterio.DatasetReader
) -> tuple[float, float, float] | None:
	'''
	Raycast an elevation map from a given position and rotation, returning the location of the ray after meeting the
	intersection threshold and the elevation at or below the ray.
	'''

	data_crs_to_lonlat = pyproj.Transformer.from_crs(elevation_data.crs, 'EPSG:4326', always_xy=True)
	wgs84_geod = pyproj.Geod(ellps='WGS84')

	euler = rotation.as_euler('ZYX', degrees=True)
	yaw = euler[0] #[-180, 180] range per scipy documentation
	pitch = euler[1] #[-90, 90] range

	# Handle case when facing directly downward
	if math.isclose(pitch, -90.0):
		alt = sample_map_geocoord(elevation_data, data_crs_to_lonlat, position[:2][::-1])[0]

		if alt is None or math.isnan(alt):
			return None

		if position[2] - alt < 0: #below terrain
			logger.error(f'Initial position is below terrain')
			return None

		logger.warning(f'Pitch is exactly -90 deg, using input lat/lon as target location.')
		return ((*position[:2], alt), alt)

	# Determine how much of the travel per unit is actually horizontal
	horiz_scale = math.cos(math.radians(pitch))
	vert_scale = math.sin(math.radians(pitch))

	p00 = pixel_to_geocoord(elevation_data, data_crs_to_lonlat, (0, 0))
	p01 = pixel_to_geocoord(elevation_data, data_crs_to_lonlat, (0, 1))
	p10 = pixel_to_geocoord(elevation_data, data_crs_to_lonlat, (1, 0))
	post_spacing_meters = min(wgs84_geod.inv(*p00, *p01)[2], wgs84_geod.inv(*p00, *p10)[2]) # meters between datapoints, from degrees
	threshold = abs(post_spacing_meters) / 16.0 # meters of acceptable distance between constructed line and datapoint. somewhat arbitrary

	# Meters of increment along ray for each stepwise check (a fraction of the distance between pixels)
	increment = post_spacing_meters / 4.0 #1 #config.increment

	# Initial ray position
	ray_position = [position[1], position[0], position[2]] #(longitude, latitude, altitude)
	prev_ray_position = ray_position

	# If the ray starts above the max altitude of the dataset, then step down to the max altitude.
	if (max_alt := elevation_data.statistics(1).max) < ray_position[2]:
		move_amt = abs((ray_position[2] - max_alt) / vert_scale)
		ray_position[:2] = wgs84_geod.fwd(*ray_position[:2], yaw, move_amt * horiz_scale)[:2]
		ray_position[2] = max_alt

	ground_alt, ground_pos = sample_map_geocoord(elevation_data, data_crs_to_lonlat, ray_position[:2])

	if ground_alt is None:
		logger.error(f'raycast_elevation_map ran out of bounds at {round(ray_position[1],4)}, {round(ray_position[0],4)}, {round(ray_position[2],1)}m. Ensure the source and target location are within the dataset bounds.')
		return None
	elif (ray_position[2] < float(ground_alt)):
		logger.error(f'raycast_elevation_map failed, bad altitude or elevation data. Initial altitude: {round(ray_position[2])}m, terrain altitude: {ground_alt}m')
		return None

	# This algorithm could be improved, and may bias just slightly past the actual target.
	while True:
		ground_alt, ground_pos = sample_map_geocoord(elevation_data, data_crs_to_lonlat, ray_position[:2])

		if ground_alt is None:
			logger.error(f'raycast_elevation_map ran out of bounds at {round(ray_position[1],4)}, {round(ray_position[0],4)}, {round(ray_position[2],1)}m. Ensure the source and target location are within the dataset bounds.')
			return None

		if ray_position[2] - ground_alt <= threshold:
			break

		# If the ray went too far, back up to the last position and increment by half as much
		if ground_alt > ray_position[2]:
			ray_position = prev_ray_position
			increment /= 2.0

		prev_ray_position = ray_position
		ray_position[:2] = wgs84_geod.fwd(*ray_position[:2], yaw, increment * horiz_scale)[:2]
		ray_position[2] += increment * vert_scale

	return (*data_crs_to_lonlat.transform(*ground_pos)[:2][::-1], ground_alt)


def ecef_raycast_elevation_map(
	position: tuple[float, float, float],
	rotation: Rotation,
	elevation_data: rasterio.DatasetReader
) -> tuple[float, float, float] | None:
	lla_to_ecef = pyproj.Transformer.from_crs('EPSG:4979', 'EPSG:4978', always_xy=True)
	data_crs_to_ecef = pyproj.Transformer.from_crs(elevation_data.crs, 'EPSG:4978', always_xy=True)
	data_crs_to_lonlat = pyproj.Transformer.from_crs(elevation_data.crs, 'EPSG:4326', always_xy=True)

	pitch = rotation.as_euler('ZYX')[1] #[-90, 90] range

	# Handle case when facing directly downward
	if math.isclose(pitch, -math.pi / 2):
		alt = sample_map_geocoord(elevation_data, data_crs_to_lonlat, position[:2][::-1])

		if alt is None or math.isnan(alt):
			return None

		if position[2] - alt < 0: #below terrain
			logger.error(f'Initial position is below terrain')
			return None

		logger.warning(f'Pitch is exactly -90 deg, using input lat/lon as target location')
		return ((*position[0:2], alt), alt)

	p00 = np.array(pixel_to_geocoord(elevation_data, data_crs_to_ecef, (0, 0)))
	p01 = np.array(pixel_to_geocoord(elevation_data, data_crs_to_ecef, (0, 1)))
	p10 = np.array(pixel_to_geocoord(elevation_data, data_crs_to_ecef, (1, 0)))
	post_spacing_meters = min(np.linalg.norm(p01 - p00), np.linalg.norm(p10 - p00)) # meters between datapoints, from degrees
	threshold = abs(post_spacing_meters) / 16.0 # meters of acceptable distance between constructed line and datapoint. somewhat arbitrary

	# Meters of increment along ray for each stepwise check (a fraction of the distance between pixels)
	increment = post_spacing_meters / 4.0 #1  #config.increment

	# Initial ray position
	ray_pos = lla_to_ecef.transform(position[1], position[0], position[2])
	ray_direction = np.array(pymap3d.ned.ned2ecef(*rotation.apply([1, 0, 0]), *position)) - ray_pos
	ray_direction /= np.linalg.norm(ray_direction)

	def distance(map_coord, map_alt, ecef_coord) -> float:
		map_ecef = np.array(lla_to_ecef.transform(*data_crs_to_lonlat.transform(*map_coord)[:2], map_alt))
		return np.linalg.norm(map_ecef - ecef_coord)

	# If the ray starts above the max altitude of the dataset, then step down to the max altitude.
	if (max_alt := elevation_data.statistics(1).max) < position[2]:
		vert_scale = math.sin(pitch)
		move_amt = abs((position[2] - max_alt) / vert_scale)
		ray_pos += ray_direction * move_amt

	ground_alt, ground_pos = sample_map_geocoord(elevation_data, data_crs_to_ecef, ray_pos)

	if ground_alt is None:
		logger.error(f'ERROR: raycast_elevation_map ran out of bounds at {round(position[0],4)}, {round(position[1],4)}, {round(position[2],1)}m. Ensure the source and target location are within the dataset bounds.')
		return None
	elif (position[2] < float(ground_alt)):
		logger.error(f'raycast_elevation_map failed, bad altitude or elevation data. Initial altitude: {round(position[2])}m, terrain altitude: {ground_alt}m')
		return None

	# This algorithm could be improved, and may bias just slightly past the actual target.
	while True:
		ground_alt, ground_pos = sample_map_geocoord(elevation_data, data_crs_to_ecef, ray_pos)

		if ground_alt is None:
			logger.error(f'ERROR: raycast_elevation_map ran out of bounds at {round(ray_pos[0],4)}, {round(ray_pos[1],4)}, {round(ray_pos[2],1)}m. Ensure the source and target location are within the dataset bounds.')
			return None

		if distance(ground_pos, ground_alt, ray_pos) <= threshold:
			break

		ray_alt = lla_to_ecef.transform(*ray_pos, direction=pyproj.enums.TransformDirection.INVERSE)[2]

		# If the ray went too far, back up to the last position and increment by half as much
		if ground_alt > ray_alt:
			ray_pos -= ray_direction * increment
			increment /= 2

		ray_pos += ray_direction * increment

	return (*data_crs_to_lonlat.transform(*ground_pos)[:2][::-1], ground_alt)


def raycast_wgs84(position: tuple[float, float, float], orientation: Rotation) -> tuple[float, float, float] | None:
	a = 6378137.0
	b = 6378137.0
	c = 6356752.314245

	lla_to_ecef = pyproj.Transformer.from_crs('EPSG:4979', 'EPSG:4978', always_xy=True)

	ray_position = lla_to_ecef.transform(position[1], position[0], position[2])

	ray_direction = np.array(pymap3d.ned.ned2ecef(*orientation.apply([1, 0, 0]), *position)) - ray_position
	ray_direction /= np.linalg.norm(ray_direction)

	x = ray_position[0]
	y = ray_position[1]
	z = ray_position[2]

	u = ray_direction[0]
	v = ray_direction[1]
	w = ray_direction[2]

	a2 = a**2
	b2 = b**2
	c2 = c**2

	x2 = x**2
	y2 = y**2
	z2 = z**2

	u2 = u**2
	v2 = v**2
	w2 = w**2

	# Quick and unreadable ray-ellipsoid intersection
	# https://stephenhartzell.medium.com/satellite-line-of-sight-intersection-with-earth-d786b4a6a9b6
	value = (-a2 * b2 * w * z) - (a2 * c2 * v * y) - (b2 * c2 * u * x)
	radical = (a2 * b2 * w2) + (a2 * c2 * v2) - (a2 * v2 * z2) + (2 * a2 * v * w * y * z) - (a2 * w2 * y2) + (b2 * c2 * u2) - (b2 * u2 * z2) + (2 * b2 * u * w * x * z) - (b2 * w2 * x2) - (c2 * u2 * y2) + (2 * c2 * u * v * x * y) - (c2 * v2 * x2)
	magnitude = (a2 * b2 * w2) + (a2 * c2 * v2) + (b2 * c2 * u2)

	if radical < 0:
		return None

	d = (value - a * b * c * np.sqrt(radical)) / magnitude

	if d < 0:
		return None

	intersection_pos = lla_to_ecef.transform(
		x + d * u,
		y + d * v,
		z + d * w,
		direction = pyproj.enums.TransformDirection.INVERSE
	)

	return (intersection_pos[1], intersection_pos[0], intersection_pos[2])


def perspective_matrix(fov: float, aspect_ratio: float, near: float, far: float) -> np.ndarray:
	'''
	fov - The camera's vertical field of view (radians)

	aspect_ratio - The camera's aspect_ratio (horizontal_size / vertical_size)

	near - The near plane distance

	far - The far plane distance
	'''

	# This projection matrix follows OpenGL conventions. This means it defaults to looking down the
	# negative direction of the Z-axis of OpenGL's conventional right-handed coordinate frame. These
	# coordinates have an X-axis pointing to the right, a Y-axis pointing up, and a Z-axis pointing
	# backwards. The projection matrix keeps with the conventional OpenGL usage in that it will flip
	# coordinates to a left-handed frame where the Z-axis now points forward.
	#
	# The coordinate frame before perspective transformation is as depicted below. As mentioned
	# above, note that the camera points down the negative Z-axis.
	#
	#              Y(up)
	#              %
	#              %           Camera Direction
	#              %         .
	#              %       .
	#              %     .
	#              %   .
	#              % .
	#              %%%%%%%%%%%%%%%% X(right)
	#            %
	#          %
	#        %
	#      %
	#    %
	#  Z(back)
	#
	# After perspective transformation, the coordinate frame will be as depicted below. The flipped
	# Z-axis means that coordinates which had a negative Z value previously will now be positive.
	#
	#  Y(up)
	#  %
	#  %           Z(forward)
	#  %         %
	#  %       %
	#  %     %
	#  %   %
	#  % %
	#  %%%%%%%%%%%%%%% X(right)

	tan_half_fov = math.tan(fov / 2.0)

	return np.array([
		[1 / (aspect_ratio * tan_half_fov), 0,                0,                                   0],
		[0,                                 1 / tan_half_fov, 0,                                   0],
		[0,                                 0,                (-(far + near)) / (far - near),     -1],
		[0,                                 0,                (-(2 * far * near)) / (far - near),  0]
	])


def normalize_coordinate(coord: tuple[float, float], image_size: tuple[int, int]) -> tuple[float, float]:
	'''
	Translate pixel coordinates (row, column) to normalized [-1, 1] clip space coordinates. Normalized clip space
	have a top-left (x, y) value of (-1, 1), and a bottom-right value of (1, -1). The pixel coordinates are assumed to
	represent the top-left of the pixel. For the center of the pixel, add (0.5, 0.5) to the coordinate.
	'''

	row_x = [0, image_size[0]]
	col_x = [0, image_size[1]]

	row_y = [1, -1]
	col_y = [-1, 1]

	interp_row = np.interp(coord[0], row_x, row_y)
	interp_col = np.interp(coord[1], col_x, col_y)

	return [interp_row, interp_col]


def pix_to_ang(pix: tuple[float, float], inv_projection: np.ndarray) -> tuple[float, float]:
	'''
	This function takes a row [1, -1] and column [-1, 1] in the frame of the camera defined by the
	provided projection matrix, and outputs the azimuth [pi/2, -pi/2] and elevation [pi/2, -pi/2] of
	the pixel relative to the camera's center.
	'''

	# Unproject the 2D image coordinate (in the normalized [-1, 1] range) into camera-relative
	# cartesian space.
	vertex_camera = np.dot([pix[1], pix[0], 0, 1], inv_projection)
	(x, y, z) = vertex_camera[0:3] / vertex_camera[3]

	# Convert the cartesian coordinate to spherical. The range is not needed. The coordinates will
	# be in OpenGL's conventional coordinate frame, so the axis rearranging here is needed to get
	# sensible output from cart2sph.
	return pymap3d.utils.cart2sph(-z, -x, y)[0:2]


def pix_to_geo(
	pix: tuple[float, float],
	camera_resolution: tuple[int, int],
	vertical_fov: float,
	camera_lla: tuple[float, float, float],
	camera_orientation: Rotation,
	elevation_data: rasterio.DatasetReader
) -> tuple[float, float, float] | None:
	'''
	pix - The pixel coordinate (row, column). A top-left image origin is assumed.

	camera_resolution - The resolution of the camera (rows, columns)

	vertical_fov - The camera's vertical field of view (radians)

	camera_lla - The LLA position of the center of the camera's imaging sensor

	camera_orientation - The orientation of the camera's imaging sensor relative to the NED frame

	elevation_data - The elevation raster file
	'''

	# The pixel coordinate needs to be normalized for pix_to_ang
	normalized_pix = normalize_coordinate(pix, camera_resolution)

	# Convert the pixel coordinate into an azimuth and elevation relative to the camera center
	projection = perspective_matrix(vertical_fov, float(camera_resolution[1]) / camera_resolution[0], 1, 1000)
	(az, el) = pix_to_ang(normalized_pix, np.linalg.inv(projection))

	# Calculate the total orientation of the pixel in the NED frame
	pixel_orientation = camera_orientation * Rotation.from_euler('ZY', (-az, -el))

	# Raycast from the pixel to the elevation map
	result = raycast_elevation_map(camera_lla, pixel_orientation, elevation_data)

	# Fall back to WGS84 intersection if the DEM intersection failed
	if result is None:
		logger.warning('The pixel does not intersect with the provided dataset. Attempting to intersect with the WGS84 ellipsoid instead.')
		result = raycast_wgs84(camera_lla, pixel_orientation)

	if result is None:
		logger.error("The ray does not intersect the WGS84 ellipsoid")

	return result


if __name__ == '__main__':
	import argparse
	import time

	parser = argparse.ArgumentParser(description='Resolve a target location from a position and orientation using a GeoTIFF dataset')
	parser.add_argument('-p', '--position', type=float, nargs=3, metavar=('lat', 'lon', 'alt'), required=True, help='Latitude, longitude, and altitude of the source in degrees and meters')
	parser.add_argument('-o', '--orientation', type=float, nargs=2, metavar=('yaw', 'pitch'), required=True, help='Yaw and pitch angle in degrees')
	parser.add_argument('-g', '--geotiff', type=str, required=True, help='Path to a GeoTIFF file for terrain data')
	#parser.add_argument('-v', '--verbose', action='store_true')

	args = parser.parse_args()

	rotation = Rotation.from_euler('ZY', args.orientation, degrees=True)
	elevation_data = rasterio.open(args.geotiff)

	t0 = time.time()

	#result = raycast_elevation_map(args.position, rotation, elevation_data)
	result = pix_to_geo([500, 500], [1000, 1000], math.pi / 2, args.position, rotation, elevation_data)

	logger.info(f'Execution time: {round((time.time() - t0) * 1000, 3)}ms')

	if result is not None:
		logger.opt(colors=True).info(f'<green>Ray intersection: {result}</green>')
	else:
		logger.opt(colors=True).error('<red>Ray tracing failed. Check input parameters and try again.</red>')
