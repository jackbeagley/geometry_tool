"""
Created on Mon Aug 16 13:23:31 2021

@author: jack
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import ogr, osr
import pandas as pd
import re
import shapely

from scipy.interpolate import splprep, splev

def rotation_matrix(theta):
	theta_rad = np.radians(theta)
	c, s = np.cos(theta_rad), np.sin(theta_rad)
	return np.array(((c, s), (-s, c)))

def get_hydrophone_location(df, i, channel, channel_spacing = 3.125):
	streamer_offset = (channel - 1) * channel_spacing
	
	j = i

	current_streamer_offset = 0.0
	location_found = False

	while (j >= 1):
		streamer_spacing = math.sqrt((df['streamer_e'].iloc[j] - df['streamer_e'].iloc[j - 1]) ** 2 + (df['streamer_n'].iloc[j] - df['streamer_n'].iloc[j - 1]) ** 2)
		
		if (current_streamer_offset + streamer_spacing) > streamer_offset:
			fractional_offset = (streamer_offset - current_streamer_offset) / streamer_spacing
			
			return np.array([df['streamer_e'].iloc[j] * (1 - fractional_offset) + df['streamer_e'].iloc[j - 1] * fractional_offset, df['streamer_n'].iloc[j] * (1 - fractional_offset) + df['streamer_n'].iloc[j - 1] * fractional_offset])
		
		current_streamer_offset = current_streamer_offset + streamer_spacing
		j = j - 1
	
	remaining_offset = streamer_offset - current_streamer_offset

	return np.array([df['streamer_e'].iloc[0] + remaining_offset * math.sin(math.radians(df['heading'].iloc[0] - 180.0)), df['streamer_n'].iloc[0] + remaining_offset * math.cos(math.radians(df['heading'].iloc[0] - 180.0))])

def get_line_length(x, y):
	n_points = len(x)

	length = 0.0

	for i in range(n_points - 1):
		length = length + np.sqrt((x[i+1] - x[i])**2 + (y[i+1] - y[i])**2)
	
	return length

def read_offsets(f, first_shot, n_shots, n_channels, c = 1500.0):
	f_lines = f.readlines()

	offsets_direct = np.empty((n_shots, n_channels))
	offsets_direct[:] = np.nan

	i = first_shot

	# Store flag for previous line
	# -1: not initialised
	#  0: ensemble information
	#  1: header (Trace Time)
	#  2: data
	#  3: blank line
	previous_line = -1

	for f_line in f_lines:
		f_line_stripped = f_line.strip()

		if (previous_line == 1) or (previous_line == 2):
			offset_time_match = re.match("(\d+)(\s+)(\d+.\d+)", f_line_stripped)
			
			if offset_time_match:
				channel = int(offset_time_match.group(1))
				offset_time = float(offset_time_match.group(3))

				j = channel - 1
				offset_distance = c * offset_time / 1000.0

				offsets_direct[i, j] = offset_distance

		shot_start_index = f_line_stripped.find('shot ')

		if shot_start_index != -1:
			shot_end_index = f_line_stripped.rfind(')')

			shot = int(f_line_stripped[shot_start_index + 5:shot_end_index])

			i = shot - first_shot

			previous_line = 0

		if f_line_stripped.find('Trace') != -1:
			previous_line = 1
		
		if len(f_line_stripped) == 0:
			previous_line = 3

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

