# -*- coding: utf-8 -*-
"""
Program: thickness_sim.py
This module contains tools that can be used to simulate thickess variations in
a net.  Then simulate probing the thickness to determine the nature of the
variations.

Angles are in radians.  You must use the same units for all lengths.  Whatever
you put in is what you will get out.

X=0, Y=0 is assumed to be bottom left of sample.
+X is across web with 0 at one edge (by convention I call it right edge)
+Y is in the machine direction with 0 at oldest part of net if measuring inline.

Location is a vector (X, Y)

"""

from math import pi, sin, cos, tan, degrees, sqrt
from copy import copy, deepcopy
import csv

# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.interpolate

import numpy as np
# import pickle
from compress_pickle import dump, load
import pandas as pd


class NetModel:
    '''
    Creates a thickness model of the net by creating all strand intersections
    ("knots") at the correct location given number of slots in the dies and the
    angle.  The knot stores its location thickness and slot number of both the
    inside and outside die that created it.  The thickess can be varied using
    several included methods.
    '''
    def __init__(self):
        # contains list of (x, y) that are to be sampled.
        self.probe_pts = []
        # contains list of samples taken of form ((x, y, thkns, probe_d), ...)
        self.probe_samples = []
        # nominal thickness of net.
        self.net_thickness = 0.
        # width of net sample
        self.net_width = 0
        # Length of net sample in length units.
        self.net_length = 0
        # Number of slots in inside die.
        self.slots_in = 0
        # Number of slots in outside die.
        self.slots_out = 0
        # Calculated strands per in inside die strands.
        self.spi_in = 0
        # Calculated strands per in outside die strands.
        self.spi_out = 0
        # nominal angle of inside die strands.
        self.angle_in = 0
        # nominal angle of outside die strands.
        self.angle_out = 0
        # list of knots of form ((x, y), thkns, slot_in, slot_out),...)
        self.knots = []
        # tolerance used on many location tests.
        self.tol = 0.


    def create_net(self, slots_in, slots_out, angle_in, angle_out, net_width,
                   net_length, net_thickness):
        '''
        Used to create model of net at constant net thickness.
            slots_in = number of slots in inside die.
            slots_out = number of slots in outside die.
            angle_in = angle of inside die strand from center line (radians).
            angle_out = angle of outside die strand from center line (radians).
            net_width = width of net (length units).
            net_length = length of net expressed in rotations of slowest die.
            net_thickness = thickness of net (length units)
        '''
        def newpt_v1_pos(cpt):
            npt = copy(cpt)
            npt['location'] = npt['location'] + v1
            npt['slot_out'] = next_slot(npt['slot_out'], self.slots_out)
            return npt

        def newpt_v1_neg(cpt):
            npt = copy(cpt)
            npt['location'] = npt['location'] - v1
            npt['slot_out'] = next_slot(npt['slot_out'], self.slots_out, True)
            return npt

        def newpt_v2_pos(cpt):
            npt = copy(cpt)
            npt['location'] = npt['location'] + v2
            npt['slot_in'] = next_slot(npt['slot_in'], self.slots_out)
            return npt

        def newpt_v2_neg(cpt):
            npt = deepcopy(cpt)
            npt['location'] = copy(npt['location'] - v2)
            npt['slot_in'] = next_slot(npt['slot_in'],
                                       self.slots_out, True)
            return npt

        def next_slot(cslot, nslots, back=False):
            '''
            Moves to the next slot in the die.
            '''
            if back:
                cslot -= 1
                if cslot < 0:
                    cslot = nslots - 1
            else:
                cslot += 1
                if cslot >= nslots:
                    cslot = 0

            return cslot

        def inbounds(cpt):
            '''
            Returns true if pt falls within bounds of net.
            '''
            return (cpt['location'][0] >= (0 - self.tol) and
                    cpt['location'][0] <= (self.net_width + self.tol) and
                    cpt['location'][1] >= (0 - self.tol) and
                    cpt['location'][1] <= (self.net_length + self.tol))

        def inbound_tr(cpt):
            '''
            returns True if x value is <= width and y value <= length.
            '''
            return ((cpt['location'][0] <= self.net_width + self.tol) and
                    (cpt['location'][1] <= self.net_length + self.tol))

        if angle_in <= 0 or angle_in >= (pi / 2):
            raise ValueError('Must be that 0 < angle_in < pi/2')

        if angle_out < 0 or angle_out >= (pi / 2):
            raise ValueError('Must be that 0 <= angle_out < pi/2')

        if (slots_in < 1) or (slots_out < 1):
            raise ValueError('Slots must be greater than 0')

        self.net_thickness = net_thickness
        self.net_width = net_width

        if angle_out > 0:
            self.net_length = net_length * max(net_width / tan(angle_in),
                                               net_width / tan(angle_out))
        else:
            self.net_length = net_length * net_width / tan(angle_in)

        self.slots_in = slots_in
        self.slots_out = slots_out
        self.angle_in = angle_in
        self.angle_out = angle_out

        self.spi_in = slots_in / (net_width * sin((pi / 2.) + angle_in))
        self.spi_out = slots_out / (net_width * sin((pi / 2.) + angle_out))
        self.tol = 10e-4

        # Calculate pitch of stands
        p_out = 1. / self.spi_out
        p_in = 1. / self.spi_in

        # v1 is vector along inside die strand
        l1 = p_out / cos(pi/2 - angle_in - angle_out)
        v1 = np.array([l1 * sin(angle_out), l1 * cos(angle_out)])

        l2 = p_in / cos(pi/2 - angle_out - angle_in)
        v2 = np.array([-l2 * sin(angle_out), l2 * cos(angle_out)])

        # By definition, knot at location 0, 0 corresponds to slot 0 on both
        # dies.

        # Create knots to right of location 0, 0.

        start_pt = {'location': np.array([0., 0.]),
                    'slot_in': 0,
                    'slot_out': 0}

        cpt = deepcopy(start_pt)

        self.knots = []

        goodrow = True
        while goodrow:
            goodrow = False
            while inbounds(cpt):
                cpt = newpt_v1_neg(cpt)
            while (not (inbounds(cpt)) and inbound_tr(cpt)):
                cpt = newpt_v1_pos(cpt)
            if inbounds(cpt):
                rowstart = cpt
                goodrow = True
                while inbounds(cpt):
                    self.knots.append({'location': cpt['location'],
                                       'thickness': net_thickness,
                                       'slot_in': cpt['slot_in'],
                                       'slot_out': cpt['slot_out']})
                    cpt = newpt_v1_pos(cpt)

            cpt = newpt_v2_neg(rowstart)

        # Create knots to left of location 0, 0.

        cpt = newpt_v2_pos(start_pt)
        goodrow = True
        while goodrow:
            goodrow = False
            while inbounds(cpt):
                cpt = newpt_v1_neg(cpt)
            while (not (inbounds(cpt)) and inbound_tr(cpt)):
                cpt = newpt_v1_pos(cpt)
            if inbounds(cpt):
                rowstart = cpt
                goodrow = True
                while inbounds(cpt):
                    self.knots.append({'location': cpt['location'],
                                       'thickness': net_thickness,
                                       'slot_in': cpt['slot_in'],
                                       'slot_out': cpt['slot_out']})
                    cpt = newpt_v1_pos(cpt)

            cpt = newpt_v2_pos(rowstart)

    def td_variator(self, cycles, magnitude, offset=0):
        '''
        Creates variation across the web.
        magnitude - sets the max amount added or subtracted.
        cycles - the number of sinusoidal cycles across the web
        offset - set in radians, shifts the pattern across the web.
        '''
        for knot in self.knots:
            x = knot['location'][0]
            variation = magnitude * sin((2.0 * pi * cycles * x /
                                         self.net_width) + offset)
            knot['thickness'] += variation

    def md_variator(self, cycles, magnitude, offset=0):
        '''
        Creates variation in machine direction of the web.
        magnitude - sets the max amount added or subtracted.
        cycles - the number of sinusoidal cycles in each die rotation.  The die
                 rotation used is the max, in length of net, of one complete
                 rotation of the die.
        offset - set in radians, shifts the pattern down the web.  For example,
        using offset of pi/4 (45 deg) will shift the pattern 1/8th of a full
        die rotation downstream.
        '''

        Ldr = min(self.net_width / tan(self.angle_in),
                  self.net_width / tan(self.angle_out))

        for knot in self.knots:
            y = knot['location'][1]
            variation = magnitude * sin(2.0 * pi * (y * cycles / Ldr) + offset)
            knot['thickness'] += variation

    def indie_variator(self, cycles, magnitude, offset=0):
        '''
        Applies sinusoidal variation following inside die slots.

        cycles - number of sinusoidal cycles around inside die.

        magnitude - max amount added or subtracted to knots.
        '''
        for knot in self.knots:
            variation = magnitude * sin((2.0 * pi * cycles *
                                         knot['slot_in'] / self.slots_in) +
                                        offset)
            knot['thickness'] += variation

    def outdie_variator(self, cycles, magnitude, offset=0):
        '''
        Applies sinusoidal variation following outside die slots.
        cycles - number of sinusoidal cycles around outside die.
        magnitude - max amount added or subtracted to knots.
        '''
        for knot in self.knots:
            variation = magnitude * sin((2.0 * pi * cycles *
                                         knot['slot_out'] / self.slots_out) +
                                        offset)
            knot['thickness'] += variation

    def set_thickness(self, new_thkns=-1):
        '''
        Sets thickess of all knots to new_thkns.  If no new_thkns is given the
        thickness is reset to initial thickness.  This eliminates all
        variations.
        '''
        if new_thkns == -1:
            new_thkns = self.net_thickness
        else:
            self.net_thickness = new_thkns

        for knot in self.knots:
            knot['thickness'] = new_thkns

    def set_average(self, new_average):
        '''
        Adds same amount to each knot to adjust average thickness from
        net_thickness to new_average.  Also updates net_thickness.
        new_average - target value for thickness average.
        '''
        avg, _, _ = self.calc_thickness_data()
        delta = new_average - avg
        for knot in self.knots:
            knot['thickness'] += delta

        self.net_thickness = new_average

    def set_range(self, new_range):
        '''
        Scales variation of net to new_range.
        '''
        avg, maxt, mint = self.calc_thickness_data()
        scaler = new_range / (maxt - mint)
        for knot in self.knots:
            knot['thickness'] = (knot['thickness'] - avg) * scaler + avg

    def copy(self):
        '''
        Make copy of net object.
        '''
        return deepcopy(self)

    def calc_thickness_data(self):
        '''
        Returns the average, max, and min thickess of all knots.
        '''
        count = 0
        total = 0
        max_thkns = -1e8
        min_thkns = 1e8
        for knot in self.knots:
            tkns = knot["thickness"]
            count += 1
            total += tkns
            if tkns > max_thkns:
                max_thkns = tkns
            if tkns < min_thkns:
                min_thkns = tkns
        return (total / count, max_thkns, min_thkns)

    def print_knots(self):
        '''
        Prints list knots in memory to screen in human readable format.
        '''
        print("\n   X      Y    TKNS   IN_SLOT    OUT_SLOT")
        for knot in self.knots:
            print(" {:9.3f} {:9.3f} {:9.4f} {:4d} {:4d}"
                  .format(knot['location'][0], knot['location'][1],
                          knot['thickness'], knot['slot_in'],
                          knot['slot_out']))

    def print_net_stats(self):
        '''
        Prints statistics of net.
        '''
        print(("Width = {:5.2f} Length = {:5.2f} Target Thickness = {:4.3f}")
              .format(self.net_width, self.net_length, self.net_thickness))
        print("Number of knots: {}".format(len(self.knots)))
        print("Number of slots in inside die: {}".format(self.slots_in))
        print("Number of slots in outside die: {}".format(self.slots_out))
        if abs(self.angle_in - self.angle_out) < .001:
            print("Net is symmetrical, angle = {:.1f} deg."
                  .format(degrees(self.angle_in) + degrees(self.angle_out)))
        else:
            if self.angle_in < .001:
                print("Inside strand is MD, outside strand " +
                      "angle = {:.1f} deg.".format(degrees(self.angle_out)))
            elif self.angle_out < .001:
                print("Outside strand is MD, inside strand " +
                      "angle = {:.1f} deg.".format(degrees(self.angle_in)))
            else:
                print(("Inside strand angle = {:.1f} deg.\n" +
                       "Outside strand angle = {:.1f} deg.")
                      .format(degrees(self.angle_in), degrees(self.angle_out)))

        avg, maxt, mint = self.calc_thickness_data()

        print("Thickness average of all knots = {:.4f}".format(avg))
        print("Thickness max = {:.4f}, min = {:.4f}, range = {:.4f}"
              .format(maxt, mint, maxt-mint))

    def print_probe_data(self):
        '''Prints out data from thickness probe.'''
        print('\n   X        Y       THK')
        for dp in self.probe_samples:
            print('{:6.2f} {:8.2f} {:8.4f}'.format(dp[0][0], dp[0][1], dp[1]))

    def print_probe_stats(self):
        '''Prints stats from current thickness probe data.'''
        print('\nNumber of samples = {:d}'.format(len(self.probe_samples)))
        print('Average thickness = {:0.4f}'.format(self.avg_probe_thkns()))
        print('Max thickness = {:0.4f}'.format(self.max_probe_thkns()))
        print('Min thickness = {:0.4f}'.format(self.min_probe_thkns()))
        print('Thickness range = {:0.4f}\n\n'.format(self.thkns_range_probe()))

    def save_probe_data_csv(self, filename='probe_data.csv'):
        file = open(filename, 'w')
        file.write('"X", "Y", "THK"\n')
        for dp in self.probe_samples:
            file.write('{:0.3f}, {:0.3f}, {:0.4f}\n'
                       .format(dp[0][0], dp[0][1], dp[1]))
        file.close()

    def probe_single(self, location, probe_d=1):
        '''
        returns thickness of thickest knot within probe_d / 2 of location.
        location - iterable with form (x, y) showing position on net to be
                   probed.
        probe_d - diameter of probe
        '''
        tkns = 0
        probed_knots = self.knot_df[((self.knot_df['x'] - location[0])**2 +
                                    (self.knot_df['y'] - location[1])**2) <=
                                    probe_d**2]
        thickest = probed_knots['thkns'].max()
        # for knot in self.knots:
        #     if (self.distance(knot['location'], (location[0],
        #                                          location[1] % self.net_width))
        #             <= (probe_d / 2) and knot['thickness'] > tkns):
        #         tkns = knot['thickness']

        return thickest

    def add_probe_line_across(self, md_start=1, md_end=0, md_spacing=12,
                              n_points=10):
        '''
        Adds probe points at n_points straight across web starting at md_start
        in machine direction.  This will be repeated every md_spacing until
        location exceeds md_end.  If this extends past the length of web, the
        web is assumed to be a repeating pattern so probing of data happens at
        (md mod length) starting at md_start.

        md_start = starting point of probing


        md_end = ending point of probing in machine direction length units.
                 if < 0, measurements will be taken to end of sample.  If = 0,
                 one row will be returned.  If > net_length, probes past the
                 end will use (md mod length) to determine Y position of sample.

        md_spacing = distance in length units between measurement.

        n_points = number of points measured across the web at each MD location.

        probe_d = diameter of the probe.
        '''
        x_pts = [(x + 0.5) * self.net_width / n_points
                  for x in range(n_points)]

        if md_end < 0:
            md_end = self.net_length
        elif md_end == 0:
            md_end = md_start

        y = md_start
        while y <= md_end:
            for x in x_pts:
                self.probe_pts.append((x, y))
            y += md_spacing

    def add_probe_pts_from_list(self, pts, md_start=0, md_end=0):
        '''
        Adds probing points to list from list.

        pts = List of x,y data points of form ((x, y), ...).  0 <= x <= 1.
              Actual probe position is x * net_width.  Y is offset from previous
              md point, or md_start in the case of the 1st point.

        md_start = offset in machine direction from start of data to 1st data
                   point.

        md_end = Stopping point.  If < 0 it is set equal to net_length.  If = 0
                 it is set to pass through list once.  If > 0 it will
                 pass through data repeatedly until locating is greater than
        '''
        one_pass = False
        if md_end < 0:
            md_end = self.net_length
        elif md_end == 0:
            one_pass = True

        y_pos = md_start
        n_passes = 1
        while ((one_pass and n_passes <= 1) or
               (not one_pass and y_pos <= md_end)):
            for pt in pts:
                x_pos = pt[0] * self.net_width
                y_pos += pt[1]
                if one_pass or (y_pos <= md_end):
                    self.probe_pts.append((x_pos, y_pos))
            n_passes += 1


    def add_probe_pts_from_file(self, filename, md_start=0, md_end=0):
        '''
        Adds probing points to list from CSV file.

        filename = name of csv file with x,y data points.  0 <= x <= 1 Actual
                   probe position is x * net_width.  Y is offset from previous
                   md point, or md_start in the case of the 1st point.

        md_start = offset in machine direction from start of data to 1st data
                   point.

        md_end = Stopping point.  If < 0 it is set equal to net_length.  If = 0
                 it is set to pass through datafile once.  If > 0 it will
                 pass through data repeatedly until locating is greater than
        '''
        pts = []
        with open(filename) as csv_data_file:
            csv_reader = csv.reader(csv_data_file)
            for row in csv_reader:
                if float(row[0]) < 0 or float(row[0]) > 1:
                    raise ValueError('X = {}, must be 0 <= X <=1'
                                     .format(row[0]))
                x_pos = float(row[0])
                y_move = float(row[1])
                pts.append((x_pos, y_move))

        self.add_probe_pts_from_list(pts, md_start, md_end)

    def execute_probe(self, probe_d=1):
        '''Probes all points in list.'''
        self.probe_samples.clear()
        for pt in self.probe_pts:
            thkns = self.probe_single(pt)
            self.probe_samples.append((pt, thkns, probe_d))

    def clear_probe_pts(self):
        ''' Clears probe points from list.'''
        self.probe_pts.clear()
        self.probe_samples.clear()

    def max_probe_thkns(self):
        return max([x[1] for x in self.probe_samples])

    def min_probe_thkns(self):
        return min([x[1] for x in self.probe_samples])

    def avg_probe_thkns(self):
        measurements = [x[1] for x in self.probe_samples]
        if len(measurements) > 0:
            return sum(measurements) / len(measurements)
        else:
            return 0

    def thkns_range_probe(self):
        return self.max_probe_thkns() - self.min_probe_thkns()

    def distance(self, pt1, pt2):
        '''
        Returns distance between 2D points.  Points should be of the form
        (x, y).
        '''
        return sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)

    def create_dataframe(self):
        '''Creates Pandas DataFrame from knot list'''
        knot_dict = {}
        x = []
        y = []
        thkns = []
        slot_in = []
        slot_out = []
        for pt in self.knots:
            x.append(pt['location'][0])
            y.append(pt['location'][1])
            thkns.append(pt['thickness'])
            slot_in.append(pt['slot_in'])
            slot_out.append(pt['slot_out'])
        knot_dict['x'] = x
        knot_dict['y'] = y
        knot_dict['thkns'] = thkns
        knot_dict['slot_in'] = slot_in
        knot_dict['slot_out'] = slot_out

        self.knot_df = pd.DataFrame(knot_dict)


    def plot_2d_points(self):
        '''
        Plots 2D points.  Color of point is controlled by thickness.  This plot
        can be used to verify pattern generated.  Best for small number of
        knots.
        '''
        x = []
        y = []
        z = []

        for knot in self.knots:
            x.append(knot['location'][0])
            y.append(knot['location'][1])
            z.append(knot['thickness'])

        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        plt.scatter(x, y, c=z)
        plt.axis('equal')

        plt.colorbar()
        plt.show()

    def plot_3d_points(self):
        '''
        Plots thickness of net in 3D.  It works OK on small number of knots
        but is very slow with larger numbers.  I don't think it is any more
        useful than the 2D contour so I may depreciate this.
        '''
        x = []
        y = []
        z = []

        for knot in self.knots:
            x.append(knot['location'][0])
            y.append(knot['location'][1])
            z.append(knot['thickness'])

        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(x, y, z, c=z, marker='o')
        ax.set_xlabel('Transverse')
        ax.set_ylabel('Machine')
        ax.set_zlabel('thickness')
        plt.axis('equal')

        plt.show()

    def plot_2d_contour(self, include_probe_pts=True):
        '''
        Creates 2D color contour of thickness data.
        '''
        x = []
        y = []
        z = []

        for knot in self.knots:
            x.append(knot['location'][0])
            y.append(knot['location'][1])
            z.append(knot['thickness'])

        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        # Set up a regular grid of interpolation points
        xi = np.linspace(x.min(), x.max(), int((x.max() - x.min()) * 16))
        yi = np.linspace(y.min(), y.max(), int((y.max() - y.min()) * 16))
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate; there's also method='cubic' for 2-D data such as here
        zi = scipy.interpolate.griddata((x, y), z, (xi, yi), method='cubic')

        plt.imshow(zi, vmin=z.min(), vmax=z.max(), origin='lower',
                   extent=[x.min(), x.max(), y.min(), y.max()])
        plt.axis('equal')
        plt.colorbar()
        # if include_probe_pts:
        #     x_list = [x[0] for x in self.probe_pts]
        #     y_list = [x[1] for x in self.probe_pts]
        #     plt.scatter(x_list, y_list, color='b')

        if include_probe_pts and len(self.probe_pts) > 0:
            fig = plt.gcf()
            ax = fig.gca()
            probe_d = 1.
            pts = [(x[0], x[1] % self.net_length) for x in self.probe_pts]
            for pt in pts:
                circle = plt.Circle(pt, probe_d / 2, color='b')
                ax.add_artist(circle)

        plt.show()

    def save_pickle(self, filename):
        '''
        Saves current net model to a compressed file that can be reloaded or
        shared.
        '''
        pickle_tuple = (self.net_thickness,
                        self.net_width,
                        self.net_length,
                        self.slots_in,
                        self.slots_out,
                        self.angle_in,
                        self.angle_out,
                        self.spi_in,
                        self.tol,
                        self.knots,
                        self.probe_pts,
                        self.probe_samples)

        outfile = open(filename, 'wb')
        dump(pickle_tuple, outfile, compression='gzip')
        outfile.close()

    def load_pickle(self, filename):
        '''
        Reloads compressed binary file saved by save_pickle.
        '''
        try:
            infile = open(filename, 'rb')
        except IOError:
            print("File not found: \"{}\"".format(filename))
            return
        indata = load(infile, compression="gzip")

        infile.close()

        self.net_thickness = indata[0]
        self.net_width = indata[1]
        self.net_length = indata[2]
        self.slots_in = indata[3]
        self.slots_out = indata[4]
        self.angle_in = indata[5]
        self.angle_out = indata[6]
        self.spi_in = indata[7]
        self.tol = indata[8]
        self.knots = indata[9]
        self.probe_pts = indata[10]
        self.probe_samples = indata[11]

        self.spi_in = (self.slots_in /
                       (self.net_width * sin((pi / 2.) + self.angle_in)))
        self.spi_out = (self.slots_out /
                        (self.net_width * sin((pi / 2.) + self.angle_out)))

    def save_csv(self, filename='net_data.csv'):
        '''
        Saves net data to csv file.
        '''
        file = open(filename, 'w')

        file.write(("\"Width = {:5.2f} Length = {:5.2f} " +
                    "Target Thickness = {:4.3f}\"\n")
                   .format(self.net_width, self.net_length, self.net_thickness))
        file.write("\"Number of knots: {}\"\n".format(len(self.knots)))
        file.write("\"Number of slots in inside die: {}\"\n"
                   .format(self.slots_in))
        file.write("\"Number of slots in outside die: {}\"\n"
                   .format(self.slots_out))
        if abs(self.angle_in - self.angle_out) < .001:
            file.write("\"Net is symmetrical, angle = {:.1f} deg.\"\n"
                       .format(degrees(self.angle_in) +
                               degrees(self.angle_out)))
        else:
            if self.angle_in < .001:
                file.write(("\"Inside strand is MD, outside strand " +
                            "angle = {:.1f} deg.\"\n")
                           .format(degrees(self.angle_out)))
            elif self.angle_out < .001:
                file.writeprint(("\"Outside strand is MD, inside strand " +
                                 "angle = {:.1f} deg.\"\n")
                                .format(degrees(self.angle_in)))
            else:
                file.writeprint(("\"Inside strand angle = {:.1f} deg.\"\n" +
                                 "\"Outside strand angle = {:.1f} deg.\"\n")
                                .format(degrees(self.angle_in),
                                        degrees(self.angle_out)))

        avg, maxt, mint = self.calc_thickness_data()

        file.write("\"Thickness average of all knots = {:.4f}\"\n".format(avg))
        file.write("\"Thickness max = {:.4f}, min = {:.4f}, range = {:.4f}\"\n"
                   .format(maxt, mint, maxt-mint))

        file.write("\n,X,Y,TKNS,IN_SLOT,OUT_SLOT\n")
        for knot in self.knots:
            file.writelines((",{:f},{:f},{:f},{:d},{:d}\n")
                            .format(knot['location'][0], knot['location'][1],
                                    knot['thickness'], knot['slot_in'],
                                    knot['slot_out']))


if __name__ == '__main__':

    NET = NetModel()
    NET.create_net(100, 100, pi / 4, pi / 4, 40, 2, .03)
    NET.outdie_variator(1, .005, offset=pi)
    NET.indie_variator(1, .005)
    NET.md_variator(1, .005)
    NET.create_dataframe()

    NET.add_probe_line_across(20)
    NET.execute_probe()
    NET.print_probe_data()

    # NET.print_net_stats()
    # NET.add_probe_line_across(24, md_end=120)
    # NET.add_probe_pts_from_file('probe_pts.csv', md_end=120)
    # NET.add_probe_pts_from_list(((.5, 0),))

    # NET.plot_2d_contour(include_probe_pts=True)
    # NET.execute_probe()
    # NET.save_probe_data_csv()

    # NET.plot_2D_contour()
    # NET.normalize_range(.004)
    # NET.plot_2D_contour()
    # NET.print_net_stats()

    # NET = NetModel(10, 10, pi / 4, pi / 4, 40, 80, .03)
    # NET.outdie_variator(1, .005)
    # NET.indie_variator(1, .005)
    # NET.MD_variator(1, .005)

    # NET.plot_2D_contour()

# text_file = open("Output.txt", "w")

# knot_cnt = 0
# for row in net_knots:
#     for knot in row:
#         text_file.write("%.4f %.4f %.5f %d %d\n" % tuple(knot))
#         knot_cnt += 1

# text_file.close()
# print("%s knots printed to Output.txt" % knot_cnt)
