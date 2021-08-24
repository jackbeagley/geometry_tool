############################################################################
#
# AUTHOR:       Jack Beagley, Geophysics Lab, Otago University August 2021
#
# PURPOSE:      Process the navigation files to determine the geometry of
#				a marine seismic survey. Can optionally take into account
#				picked direct arrivals during geometry processing to more
#				accurately bin traces into CDPs
#
#############################################################################

import enum
import math
import matplotlib.pyplot as plt
import numpy as np
import ogr, osr
import pandas as pd
import re

from scipy.interpolate import splprep, splev

class PicReaderState(enum.Enum):
	"""
	State enum to keep track of the .pic file reader
	"""
	not_initialised = 0
	ensemble_information = 1
	header = 2
	data = 3
	blank =  4

def rotation_matrix(theta):
	"""
	Output a numpy array containing a rotation matrix `A`.

	This matrix is to be used on coordinates, rotating them clockwise
	about the origin, such as:

	`p' = A * p`

	Parameters
	----------
	theta: angle (in radians) by which to rotate points

	Returns
	-------
	Numpy rotation matrix corresponding to theta
	"""
	theta_rad = np.radians(theta)
	c, s = np.cos(theta_rad), np.sin(theta_rad)

	return np.array(((c, s), (-s, c)))

def get_hydrophone_location(df, i, channel, channel_spacing = 3.125):
	"""
	Retrace the first hydrophones (channel = 1) locus to determine where a
	specified channel will be

	Parameters
	----------
	df: Dataframe containing streamer, source and navigation locations

	i: Index of `df` to consider

	channel: Hydrophone channel to determine locaiton of

	channel_spacing: Spacing between hydrophone channels on streamer.
		Default for Otago University's Geometrics MicroEel is 3.125m

	Returns
	-------
	Two element numpy array containing Northing and Easting of the hydrophone location
	"""
	# channel starts at 1, so the offset is calculated as below
	streamer_offset = (channel - 1) * channel_spacing
	
	# Create a counter to step back into the dataframe
	j = i

	# To keep track of the streamer offset as we trace back along the streamer
	current_streamer_offset = 0.0

	while (j >= 1):
		# Calculate the spacing between subsequent streamer locations
		streamer_spacing = math.sqrt((df['streamer_e'].iloc[j] - df['streamer_e'].iloc[j - 1]) ** 2 + (df['streamer_n'].iloc[j] - df['streamer_n'].iloc[j - 1]) ** 2)
		
		# Determine if the hydrophone is within this interval or not
		if (current_streamer_offset + streamer_spacing) > streamer_offset:
			# Calculate where in this interval (as a ratio) the hydrophone is
			x = (streamer_offset - current_streamer_offset) / streamer_spacing
			
			hydrophone_location_e = df['streamer_e'].iloc[j] * (1 - x) + df['streamer_e'].iloc[j - 1] * x
			hydrophone_location_n = df['streamer_n'].iloc[j] * (1 - x) + df['streamer_n'].iloc[j - 1] * x

			# Return the location
			return np.array([hydrophone_location_e, hydrophone_location_n])
		
		# Increment the current offset along the streamer locus
		current_streamer_offset = current_streamer_offset + streamer_spacing
		j = j - 1
	
	# If the end of the locus has been reached, determine how much offset is remaining
	remaining_offset = streamer_offset - current_streamer_offset

	# Use the heading at the start of the survey to project back an estimated location of the hydrophone
	hydrophone_location_e = df['streamer_e'].iloc[0] + remaining_offset * math.sin(math.radians(df['heading'].iloc[0] - 180.0))
	hydrophone_location_n = df['streamer_n'].iloc[0] + remaining_offset * math.cos(math.radians(df['heading'].iloc[0] - 180.0))

	# Return the location
	return np.array([hydrophone_location_e, hydrophone_location_n])

def get_line_length(x, y):
	"""
	Get the length of a line specified by x and y coordinates by
	adding together the distance between subsequent points

	Parameters
	----------
	x: Array of x coordinates

	y: Array of y coordinates

	Returns
	-------
	length of line in the same units as the input coordinates
	"""
	n_points = len(x)

	length = 0.0

	# Loop through all of the coordinates, adding the distance between them
	for i in range(n_points - 1):
		length = length + np.sqrt((x[i+1] - x[i])**2 + (y[i+1] - y[i])**2)
	
	# Return the length
	return length

def read_offsets(f, first_shot, n_shots, n_channels, c = 1500.0):
	"""
	Read direct arricals from a .pic file (output by CLARITAS), convert
	to offsets using a specified speed of sound and output to a
	2D numpy array

	Parameters
	----------
	f: File stream object for the .pic file

	first_shot: Shot number for the first expected shot
		This may not be present in the .pic if a direct arrival
		couldn't be picked by the user
	
	n_shots: Number of shots in the line

	n_channels: Number of seismic channels

	c: Speed of sound in water

	Returns
	-------
	2D numpy array with rows corresponding to shots and columns corresponding
	to channels. NaN indicates that a direct arrival wasn't found
	"""
	f_lines = f.readlines()

	# Initialise empty arrays for storing offsets, set to NaN
	offsets_direct = np.empty((n_shots, n_channels))
	offsets_direct[:] = np.nan

	i = first_shot

	# Remember file reader state
	reader_state = PicReaderState.not_initialised

	# Loop through all lines in the file
	# A state machine has been implemented to read the file
	for f_line in f_lines:
		# Strip leading and trailing whitespace
		f_line_stripped = f_line.strip()

		if (reader_state == PicReaderState.header) or (reader_state == PicReaderState.data):
			# Use a REGEX to try extract trace number and offset from the line
			offset_time_match = re.match("(\d+)(\s+)(\d+.\d+)", f_line_stripped)
			
			# If there was a match, read the information
			if offset_time_match:
				channel = int(offset_time_match.group(1))
				offset_time = float(offset_time_match.group(3))

				# Column index is one less than hydrophone channel
				j = channel - 1
				# Calculate the offset
				offset_distance = c * offset_time / 1000.0

				# Populate the offsets array
				offsets_direct[i, j] = offset_distance

		# Check if 'shot' is present in the line
		shot_start_index = f_line_stripped.find('shot ')

		# If 'shot' is present, extract the shot index from the line
		if shot_start_index != -1:
			shot_end_index = f_line_stripped.rfind(')')

			# Get the shot index from the line and parse as an integer
			shot = int(f_line_stripped[shot_start_index + 5:shot_end_index])

			# Map this shot index to the correct row of the array to populate
			i = shot - first_shot

			reader_state = PicReaderState.ensemble_information

		# If 'Trace' exists in the line, it is a header immediately
		# preceding data
		if f_line_stripped.find('Trace') != -1:
			reader_state = PicReaderState.header
		
		# Check if the line is blank
		if len(f_line_stripped) == 0:
			reader_state = PicReaderState.blank

	return offsets_direct

# spatial reference system
input_EPSG = 4326
output_EPSG = 2193

# create coordinate transformation
in_spatial_ref = osr.SpatialReference()
in_spatial_ref.ImportFromEPSG(input_EPSG)

out_spatial_ref = osr.SpatialReference()
out_spatial_ref.ImportFromEPSG(output_EPSG)

coord_transform = osr.CoordinateTransformation(in_spatial_ref, out_spatial_ref)

# Directories
nav_dir = "/errigal_data/jack/claritas/data/21PL0415_Dusky_Sound/from_geode"
offset_dir = "/home/jack/jobs/21PL0415_Dusky/COMMON/STATICS"
header_out_dir = "/home/jack/jobs/21PL0415_Dusky/COMMON/HEADERS"

# Boomer properties
boomer_offset_x = 9.71
boomer_offset_y = -36.64
boomer_offset = np.array([-boomer_offset_x, boomer_offset_y])

# Streamer properties
streamer_offset_x = -0.89
streamer_offset_y = -37.8 # -36.64 - 1.16
streamer_offset = np.array([-streamer_offset_x, streamer_offset_y])
streamer_spacing = 3.125
n_channels = 24

# CDP spacing
cdp_spacing = 2.0

line_number = "0004"

line_df = pd.read_csv((nav_dir + "/line%s.Nav_parsed.txt") % line_number, sep='\s+', names = ['shot', 'lon_deg', 'lon_min', 'lon_ew', 'lat_deg', 'lat_min', 'lat_ns', 'time'], skiprows=1)
line_offsets_f = open((offset_dir + "/line%s.pic") % line_number)

n_shots = len(line_df.index)
n_records = n_shots * n_channels

first_shot = line_df['shot'].iloc[0]
last_shot = line_df['shot'].iloc[-1]

print('1\tReading direct arrivals...')

offsets_direct = read_offsets(line_offsets_f, first_shot, n_shots, n_channels)

line_df['lon_dd'] = line_df['lon_deg'] + line_df['lon_min'] / 60
line_df['lat_dd'] = -(line_df['lat_deg'] + line_df['lat_min'] / 60)

print('2\tReprojecting shotpoint coordinates...')

for new_col in ['easting', 'northing', 'heading', 'boomer_e', 'boomer_n', 'streamer_e', 'streamer_n']:
	line_df[new_col] = pd.Series(dtype=float)

for i in range(n_shots):
	shotpoint = ogr.Geometry(ogr.wkbPoint)
	shotpoint.AddPoint(line_df['lon_dd'].iloc[i], line_df['lat_dd'].iloc[i])
	shotpoint.Transform(coord_transform)

	line_df.ix[i,'easting'] = shotpoint.GetX()
	line_df.ix[i,'northing'] = shotpoint.GetY()

line_df['easting_filt'] = line_df['easting'].rolling(5, center=True, min_periods=1).mean()
line_df['northing_filt'] = line_df['northing'].rolling(5, center=True, min_periods=1).mean()

print('3\tCalculating heading...')

for i in range(1, n_shots):
	diff_x = line_df['easting_filt'].iloc[i] - line_df['easting_filt'].iloc[i - 1]
	diff_y = line_df['northing_filt'].iloc[i] - line_df['northing_filt'].iloc[i - 1]

	line_df.ix[i, 'heading'] = (math.degrees(math.atan2(diff_x, diff_y)) + 360.0) % 360.0

line_df.ix[0, 'heading'] = line_df.ix[1, 'heading']

print('4\tCalculating boomer, streamer locations...')

for i in range(n_shots):
	boomer_offset_rotated = np.matmul(rotation_matrix(line_df['heading'].iloc[i]), boomer_offset)
	streamer_offset_rotated = np.matmul(rotation_matrix(line_df['heading'].iloc[i]), streamer_offset)
	
	line_df.ix[i, 'boomer_e'] = line_df['easting_filt'].iloc[i] + boomer_offset_rotated[0]
	line_df.ix[i, 'boomer_n'] = line_df['northing_filt'].iloc[i] + boomer_offset_rotated[1]

	line_df.ix[i, 'streamer_e'] = line_df['easting_filt'].iloc[i] + streamer_offset_rotated[0]
	line_df.ix[i, 'streamer_n'] = line_df['northing_filt'].iloc[i] + streamer_offset_rotated[1]

# Make arrays for midpoints
mps_e = np.empty((n_shots, n_channels))
mps_n = np.empty_like(mps_e)
offset_estimates = np.empty_like(mps_e)

print('5\tCalculating CMP locations...')

for i in range(n_shots):
	shot_en = np.array([line_df.ix[i, 'boomer_e'], line_df.ix[i, 'boomer_n']])

	for j in range(n_channels):
		record_en = get_hydrophone_location(line_df, i, j + 1)
		mp_en = (record_en + shot_en) / 2.0
		mps_e[i, j] = mp_en[0]
		mps_n[i, j] = mp_en[1]
		
		offset_estimates[i, j] = np.sqrt(sum((shot_en - record_en)**2))
	
print('6\tFitting CDP spline...')

# Fit a spline along the middle channel
pts = np.vstack((mps_e[:, n_channels//2].flatten(), mps_n[:, n_channels//2].flatten()))
s_spl = n_shots / 100.0
(tck, u_eval), fp, ier, msg = splprep(pts, u=None, per=0, k=3, full_output=True, s = s_spl)
print('\tSpline fit score is %.3f' % fp) 
cdp_e_eval, cdp_n_eval = splev(u_eval, tck, der=0)

cdp_length = get_line_length(cdp_e_eval, cdp_n_eval)

print('\tCDP spline is %.0fm long' % cdp_length)

u_cdp = np.arange(0.0, 1.0, cdp_spacing / cdp_length)
n_cdps = len(u_cdp)
cdps = range(1, n_cdps + 1)
cdp_fold = np.empty_like(cdps)
cdp_e, cdp_n = splev(u_cdp, tck, der=0)

print('7\tBinning shots into CDPs...')

record_cdp_indices = np.zeros_like(mps_e, dtype=int)
record_cdps = np.zeros_like(mps_e, dtype=int)
cdp_tracecount = np.zeros_like(cdps, dtype=int)
record_cdptrace = np.zeros_like(mps_e, dtype=int)
offset_final = np.zeros_like(mps_e, dtype=int)

for i in range(n_shots):
	shot_en = np.array([line_df.ix[i, 'boomer_e'], line_df.ix[i, 'boomer_n']])

	for j in range(n_channels):
		# Get the midpoint for the record
		record_en = np.array([mps_e[i, j], mps_n[i, j]])
	
		# If a direct arrival has been picked, update the midpoint location
		if not np.isnan(offsets_direct[i, j]):
			# Move the record along the line between it and the shotpoint to correct for the offset
			record_en = shot_en + (record_en - shot_en) * (offsets_direct[i, j] / offset_estimates[i, j])

			offset_final[i, j] = -int(offsets_direct[i, j] * 10)
		else:
			offset_final[i, j] = -int(offset_estimates[i, j] * 10)

		cdp_distance_e = cdp_e - record_en[0]
		cdp_distance_n = cdp_n - record_en[1]

		cdp_distances = np.sqrt(cdp_distance_e**2 + cdp_distance_n**2)

		record_cdp_indices[i, j] = np.argmin(cdp_distances)
		record_cdps[i, j] = cdps[record_cdp_indices[i, j]]

		cdp_tracecount[record_cdp_indices[i, j]] = cdp_tracecount[record_cdp_indices[i, j]] + 1
		record_cdptrace[i, j] = cdp_tracecount[record_cdp_indices[i, j]]

print('8\tCalculating fold...')

for i in range(n_cdps):
	cdp_fold[i] = np.sum((record_cdps == cdps[i]).flatten())

print('9\tWriting out *.ahl file...')

f_acd = open('%s/line%s.acd' % (header_out_dir, line_number), 'w')

f_acd.write('''\
|Header name  |Key|FirstCol|LastCol|Scalar |Adder  |FillMode|Comment          |
 SHOTID        P   1        6                                                  
 CHANNEL       S   7        10                                                 
 CDP               11       16                                                 
 CDPTRACE          17       20                                                 
 CDP_X             21       30                                                 
 CDP_Y             31       40                                                 
 OFFSET            41       46                                                 ''')

f_acd.close()

f_header = open('%s/line%s.txt' % (header_out_dir, line_number), 'w')
f_ahl = open('%s/line%s.ahl' % (header_out_dir, line_number), 'w')

f_ahl.write('''\
ADDHDR
Primary key : SHOTID
Secondary key : CHANNEL
Interpolation key : 
Add geometry from PYTHON script\n\
''')
f_ahl.write('|Pkey  |Skey  |X1    |X2  |X3        |X4        |X5    |\n')

for i in range(n_shots):
	for j in range(n_channels):
		record_cdp = record_cdps[i, j]
		record_cdp_index = record_cdp_indices[i, j]
		cdp_en = np.array([cdp_e[record_cdp_index], cdp_n[record_cdp_index]])
		cdp_en_dm = np.rint(cdp_en * 10.0).astype(int)

		# SHOT CHANNEL CDP CDPTRACE CDP_X CDP_Y OFFSET
		f_header.write('%6i%4i%6i%4i%10i%10i%6i\n' % (line_df['shot'].iloc[i], j+1, record_cdp, record_cdptrace[i, j], cdp_en_dm[0], cdp_en_dm[1], offset))
		
		f_ahl.write(' %-6i %-6i %-6i %-4i %-10i %-10i %-6i\n' % (line_df['shot'].iloc[i], j+1, record_cdp, record_cdptrace[i, j], cdp_en_dm[0], cdp_en_dm[1], offset))

f_header.close()
f_ahl.close()

plt.figure()

plt.imshow(np.transpose(record_cdps), cmap='hot', interpolation='nearest', aspect = 'auto', extent=(first_shot, last_shot, n_channels, 1))
plt.title('CDP bins')
plt.colorbar()

plt.figure()

plt.imshow(np.transpose(offsets_direct - offset_estimates), cmap='hot', interpolation='nearest', aspect = 'auto', extent=(first_shot, last_shot, n_channels, 1))
plt.title('Offset Discrepancy')
plt.colorbar()

plt.figure()

plt.imshow(np.transpose(offsets_direct), cmap='hot', interpolation='nearest', aspect = 'auto', extent=(first_shot, last_shot, n_channels, 1))
plt.title('Direct Arrival Offsets')
plt.colorbar()

plt.figure()

plt.imshow(np.transpose(offset_estimates), cmap='hot', interpolation='nearest', aspect = 'auto', extent=(first_shot, last_shot, n_channels, 1))
plt.title('Estimated Offsets')
plt.colorbar()

plt.figure()

plt.plot(cdps, cdp_fold)
plt.title('CDP fold')

plt.figure()

plt.plot(mps_e.flatten(), mps_n.flatten(), 'k.', label='MPs')
plt.plot(line_df['easting_filt'], line_df['northing_filt'], 'b.', label='Nav points')
plt.plot(line_df['boomer_e'], line_df['boomer_n'], 'g.', label='Gun points')
plt.plot(line_df['streamer_e'], line_df['streamer_n'], 'r.', label='Streamer points')
plt.plot(cdp_e, cdp_n, 'm-', label='CDP line')
plt.scatter(cdp_e[1:-1], cdp_n[1:-1], c = cdp_fold[1:-1], label='CDPs')
plt.colorbar()

plt.legend()
plt.grid('on')
plt.axis('equal')
plt.show()

plt.figure()

plt.plot(cdp_e, cdp_n, 'm-', label='CDP line')
plt.scatter(cdp_e[1:-1], cdp_n[1:-1], c = cdp_fold[1:-1], label='CDPs')
plt.colorbar()

plt.legend()
plt.grid('on')
plt.axis('equal')
plt.show()

