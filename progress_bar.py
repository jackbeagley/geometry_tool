'''
Implementation of python progress bar (or status bar)
without using Progressbar library.

Result:
Completed: [==================================================] 100%

Created by Sibo, 12 Oct.
Modified by jackbeagley 26/08/2021
'''
import sys

def progressbar(i, n, bar_length = 40):
    percent = 100.0 * i / n
    sys.stdout.write('\r')
    sys.stdout.write("Completed: [{:{}}] {:>3}%"
                     .format('=' * int(percent / (100.0 / bar_length)),
                             bar_length, int(percent)))
    sys.stdout.flush()
