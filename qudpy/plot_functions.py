import numpy as np
import matplotlib.pyplot as plt


def multiplot(data=None, scan_range=None, labels=None, title_list=None, scale='linear', color_map='PuOr',
              interpolation='spline36', center_scale=True):
    """
    Plot multiple dataset for spectral and evolution data
    :param data: List of spectra or dipole expectation values or any other variable of interest. Must be float data type
    :param scan_range: The min and max of both axis in the format [xmin, xmax, ymin, ymax]
    :param labels: List of label for each axis
    :param title_list: List of titles for each plot
    :param scale: Scaling of the data points, two choices are 'linear' and 'log'
    :param color_map: Choice of colormap
    :param interpolation: Interpolation for points in plot.
    :param center_scale: Shift individual datasets to sent center value to zero.
    :return: Does not return anything
    """
    if data is None:
        print('Nothing to plot, kindly provide the data')
        return
    if scan_range is None:
        print('Scan range not given')
        scan_range = [0, 1, 0, 1]
    if title_list is None:
        print('titles not given')
        title_list = [str(x + 1) for x in range(len(data))]

    num_plots = len(data)  # number of plots (depends on the length of data list)
    if num_plots <= 3:
        rows = 1
        cols = num_plots
    else:
        rows = int(np.ceil(num_plots / 3))
        cols = 3

    if center_scale:
        print('centering data around zero')
        data = [d-(np.min(d) + np.max(d))/2 for d in data]

    if scale == 'log':
        data = np.array([log_scale(s) for s in data])

    axes = []
    fig = plt.figure(figsize=(16, 4))
    for k in range(num_plots):
        axes.append(fig.add_subplot(rows, cols, k + 1))
        subplot_title = (title_list[k])
        axes[-1].set_title(subplot_title)
        plt.plot([scan_range[0], scan_range[1]], [scan_range[3], scan_range[2]], '--', color="black", linewidth=0.5)
        plt.plot([scan_range[0], scan_range[1]], [scan_range[2], scan_range[3]], '--', color="black", linewidth=0.5)
        im = plt.imshow(data[k], cmap=color_map, origin='lower', interpolation=interpolation, extent=scan_range, aspect=1)
        if labels:
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])
        plt.colorbar(im, ax=axes[-1])

    fig.tight_layout()
    plt.show()

    return


def plot(data=None, scan_range=None, labels=None, title=None, scale='linear', color_map='viridis', interpolation='spline36'):
    """
    Plot singe dataset for spectral and evolution data
    :param data: Single dataset. Must be float
    :param scan_range: The min and max of both axis in the format [xmin, xmax, ymin, ymax]
    :param labels: List of label for each axis
    :param title: Title the plot
    :param scale: Scaling of the data points, two choices are 'linear' and 'log'
    :param color_map: Choice of colormap
    :param interpolation: Interpolation for points in plot.
    :return: Does not return anything
    """
    if data is None:
        print('Nothing to plot, kindly provide the data')
        return
    if scan_range is None:
        print('Scan range not given')
        scan_range = [0, 1, 0, 1]

    plt.figure()
    if scale == 'log':
        data = log_scale(data)

    plt.plot([scan_range[0], scan_range[1]], [scan_range[3], scan_range[2]], '--', color="black", linewidth=0.5)
    plt.plot([scan_range[0], scan_range[1]], [scan_range[2], scan_range[3]], '--', color="black", linewidth=0.5)
    plt.imshow(data, cmap=color_map, origin='lower', interpolation=interpolation, extent=scan_range, aspect='auto')
    plt.colorbar()
    if title:
        plt.title(title)
    if labels:
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
    plt.show()

    return


def log_scale(z):
    """
    Simple function for rescaling the 2D input matrix to log scale.
    Note: the negative numbers are downshifted by 1 and the positive numbers are upshifted by 1 to remove numbers
    between -1 and 1.
    """
    x, y = np.shape(z)
    for n in range(x):
        for m in range(y):
            if z[n, m] >= 0:
                z[n, m] = np.log(z[n, m]+1)
            else:
                z[n, m] = -np.log(-z[n, m]+1)
    return z


def pop_plot(data=None, scan_range=None, labels=None, title_list=None, scale='linear', color_map='PuOr', interpolation='spline36'):
    """
    Plot multiple dataset for spectral and evolution data
    :param data: List of spectra or dipole expectation values or any other variable of interest
    :param scan_range: The min and max of both axis in the format [xmin, xmax, ymin, ymax]
    :param labels: List of label for each axis
    :param title_list: List of titles for each plot
    :param scale: Scaling of the data points, two choices are 'linear' and 'log'
    :param color_map: Choice of colormap
    :param interpolation: Interpolation for points in plot.
    :return: Does not return anything
    """
    if data is None:
        print('Nothing to plot, kindly provide the data')
        return
    if scan_range is None:
        print('Scan range not given')
        scan_range = [0, 1, 0, 1]
    if title_list is None:
        print('titles not given')
        title_list = [str(x + 1) for x in range(len(data))]

    num_plots = len(data)  # number of plots (depends on the length of data list)
    if num_plots <= 3:
        rows = 1
        cols = num_plots
    else:
        rows = int(np.ceil(num_plots / 3))
        cols = 3

    if scale == 'log':
        data = np.array([log_scale(s.real) for s in data])

    axes = []
    fig = plt.figure(figsize=(16, 4))
    for k in range(num_plots):
        axes.append(fig.add_subplot(rows, cols, k + 1))
        subplot_title = (title_list[k])
        axes[-1].set_title(subplot_title)
        im = plt.imshow(data[k], cmap=color_map, origin='lower', interpolation=interpolation, extent=scan_range, aspect=1)
        if labels:
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])
        plt.colorbar(im, ax=axes[-1])

    fig.tight_layout()
    plt.show()

    return


def coor(data_x,data_y,z):
    index_1= []
    index_2= []
    index_3= []
    index_4= []
    for a in range(len(data_x[0])):
        
        if data_x[0][a]>=z[0]:
           index_1.append(a)
           # print(a)
        if data_x[0][a]<=z[1]:
           index_2.append(a)

        if data_y[a][0]>=z[2]:
              index_3.append(a)

        if data_y[a][0]<=z[3]:
              index_4.append(a)

    index = [index_1[0],index_2[len(index_2)-1],index_3[0],index_4[len(index_4)-1]]
    return index
        
    

def log_scale(z):
    """
    Simple function for rescaling the 2D input matrix to log scale.
    Note: the negative numbers are downshifted by 1 and the positive numbers are upshifted by 1 to remove numbers
    between -1 and 1.
    """
    x, y = np.shape(z)
    for n in range(x):
        for m in range(y):
            if z[n, m] >= 0:
                z[n, m] = np.log(z[n, m]+1)
            else:
                z[n, m] = -np.log(-z[n, m]+1)
    return z


def silva_plot(spectra_list=None,x_val=None,y_val=None, labels=None, title_list=None, scale='linear', color_map='PuOr',
               interpolation='spline36', center_scale=True, plot_sum=True, plot_quadrant='All', invert_y=True,
               diagonals=[True, True],Zoom_coor=None):
    """
    Plot multiple spectra with real, imaginary and abs values
    :param data: List of spectra or dipole expectation values or any other variable of interest. Must be float data type
    :param x_val: scan range of data in the x position
    :param y_val: scan range of data in the y position
    :param labels: List of label for each axis
    :param title_list: List of titles for each plot
    :param scale: Scaling of the data points, two choices are 'linear' and 'log'
    :param color_map: Choice of colormap
    :param interpolation: Interpolation for points in plot.
    :param center_scale: Shift individual datasets to sent center value to zero.
    :param plot_sum: plots the total sum of the input data sets with separate graphs for real, imag and abs values
    :param plot_quadrant: only plots the selected quadrant(s) for the graphs
    :param plot_quadrant = 'Zoom' to make a zoom on the coordinates Zoom_coor
    :param invert_y: flips the y-axis by converting -ve values to +ve
    :param Zoom_coor: allows to Zoom in a specific tuple of coordinates in the format [xmin,xmax,ymin,ymax]
    :return: Does not return anything
    
    """
    if spectra_list is None:
        print('Nothing to plot, kindly provide the data')
        return
    if x_val is None:
        print('Scan range not given. Using default range of 0 to 1')
        x_val = [0, 1]
    if y_val is None:
        y_val = [0,1]
    if title_list is None:
        print('titles not given. Using default titles: simple numbers')
        title_list = [str(x + 1) for x in range(len(spectra_list)*3)]
    x_i = int(np.where(x_val==0)[1][0])
    y_i = int(np.where(x_val==0)[1][1])
    if plot_quadrant == '1':
        spectra_list = [x[x_i:, y_i:] for x in spectra_list]
        scan_range = [0, np.max(x_val), 0, np.max(y_val)]
    elif plot_quadrant == '2':

        spectra_list = [x[x_i:, :y_i] for x in spectra_list]
        scan_range = [np.min(x_val), 0, 0, np.max(y_val)]
        
        
    elif plot_quadrant == '3':
        spectra_list = [x[:x_i, :y_i] for x in spectra_list]
        scan_range = [np.min(x_val), 0, np.min(y_val), 0]
    elif plot_quadrant == '4':
        spectra_list = [x[:x_i, y_i:] for x in spectra_list]
        scan_range = [0, np.max(x_val), np.min(y_val), 0]
    #print(np.shape(spectra_list[0]))
    elif plot_quadrant == 'Zoom':
        index = coor(x_val,y_val,Zoom_coor)
        # print(index)
        spectra_list = [x[index[2]:index[3],index[0]:index[1]] for x in spectra_list]
        scan_range = [x_val[0][index[0]],x_val[0][index[1]],y_val[index[2]][0],y_val[index[3]][0]]
    elif plot_quadrant == 'All':
         scan_range = [np.min(x_val),np.max(x_val),np.min(y_val),np.max(y_val)] 
    if invert_y:
        spectra_list = [np.flip(x, 1) for x in spectra_list]
        scan_range[2], scan_range[3] = -scan_range[3], -scan_range[2]

    # separating the real, imaginary and absolute values of each spectrum
    data_real = np.real(spectra_list)
    data_imag = np.imag(spectra_list)
    data_abs = np.abs(spectra_list)
    data = []
    for k in range(len(spectra_list)):
        data.append(data_real[k])
        data.append(data_imag[k])
        data.append(data_abs[k])

    if plot_sum:
        data_sum = np.sum(spectra_list, 0)
        data.append(data_sum.real)
        data.append(data_sum.imag)
        data.append(np.abs(data_sum))
        title_list.append('Total')



    num_plots = len(data) # number of plots (depends on the length of data list)
    if num_plots <= 3:
        rows = 1
        cols = num_plots
    else:
        rows = int(np.ceil(num_plots / 3))
        cols = 3

    if center_scale:
        print('centering data around zero')
        data = [d-(np.min(d) + np.max(d))/2 for d in data]

    if scale == 'log':
        print('using log scale')
        data = np.array([log_scale(s) for s in data])

    axes = []
    titles = ['real', 'imag', 'abs']
    fig = plt.figure(figsize=(5*cols, 5*rows))
    for k in range(num_plots):
        axes.append(fig.add_subplot(rows, cols, k + 1))
        if k % 3 == 0:
            title = title_list[k//3]


        subplot_title = (title + ' ' + titles[k % 3])
        axes[-1].set_title(subplot_title)
        # drawing diagonal lines
        if diagonals[0]:
            plt.plot([scan_range[0], scan_range[1]], [scan_range[3], scan_range[2]], '--', color="black", linewidth=0.5)
        if diagonals[1]:
            plt.plot([scan_range[0], scan_range[1]], [scan_range[2], scan_range[3]], '--', color="black", linewidth=0.5)
        im = plt.imshow(data[k], cmap=color_map, origin='lower', interpolation=interpolation, extent=scan_range,
                        aspect=1)


        if labels:
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])
        plt.colorbar(im, ax=axes[-1], shrink=0.7)

    fig.tight_layout()
    plt.show()

    return