#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 10:45:33 2019

@author: 1vn
"""



def myFunc(myloc):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from collections import namedtuple
    

    n_groups = 6
    
    sliding = (42.4,56.2,59.0,60.4,60.6,58.2)
    growing = (57.7,57.7,57.2,55.4,54.0,52.3)
    
    means_men = (20, 35, 30, 35, 27)
    std_men = (2, 3, 4, 1, 2)

    means_women = (25, 32, 34, 20, 25)
    std_women = (3, 5, 2, 3, 3)

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.35

    opacity = 0.7
    error_config = {'ecolor': '0.3'}

    rects1 = ax.bar(index, sliding, bar_width,
                alpha=opacity, color='blue',
                error_kw=error_config,
                label='Sliding')

    rects2 = ax.bar(index + bar_width, growing, bar_width,
                alpha=opacity, color='orange',
                error_kw=error_config,
                label='Growing')

    ax.set_xlabel('Training Window Size')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Training Window Size and Method')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(('8 Cycles', '16 Cycles', '32 Cycles', '64 Cycles', '96 Cycles', '128 Cycles'))
    ax.legend(loc=myloc)

    fig.tight_layout()
    plt.show()
    myloc += 1
    
myData = 'a'
print(myData)

for item in range(11):
    myFunc(item)
#    letter = myData + '--> b'
#    letter = letter + '--> c'
#    print(letter)
#    
#print('Final Letter',letter)

#allow files to be passed as pre