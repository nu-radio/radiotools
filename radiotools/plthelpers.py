import inspect, re
import math
import sys
import os
from matplotlib import colors as mcolors

from scipy import optimize

import matplotlib.pyplot as plt
import numpy as np
import radiotools.stats


def get_discrete_cmap(N, base_cmap='viridis'):
    cmap = plt.get_cmap(base_cmap, N)
    colors = []
    N = int(N)
    for i in range(N):
        if(i % 2 == 0):
            colors.append(cmap.colors[i // 2])
        else:
            colors.append(cmap.colors[-i // 2])
    return mcolors.ListedColormap(colors)

#     base = plt.cm.get_cmap(base_cmap)
#     color_list = base(np.linspace(0, 1, N))
#     cmap_name = base.name + str(N)
#     return base.from_list(cmap_name, color_list, N)


def fit_chi2(func, datax, datay, datayerror, p_init=None):
    datayerror[datayerror == 0] = 1
    popt, cov = optimize.curve_fit(func, datax, datay, sigma=datayerror,
                                     p0=p_init)
#     print optimize.curve_fit(func, datax, datay, sigma=datayerror,
#                                p0=p_init, full_output=True)
    y_fit = func(datax, *popt)
    chi2 = np.sum((y_fit - datay) ** 2 / datayerror ** 2)
    if(len(datax) != len(popt)):
        chi2ndf = 1. * chi2 / (len(datax) - len(popt))
    else:
        chi2ndf = np.NAN
    func2 = lambda x: func(x, *popt)
    print("result of chi2 fit:")
    print("\tchi2/ndf = %0.2g/%i = %0.2g" % (chi2, len(datax) - len(popt), chi2ndf))
    print("\tpopt = ", popt)
    print("\tcov = ", cov)
    return popt, cov, chi2ndf, func2


def fit_line(datax, datay, datayerror=None, p_init=None):
    if datayerror is None:
        datayerror = np.ones_like(datay)
    func = lambda x, p0, p1: x * p1 + p0
    return fit_chi2(func, datax, datay, datayerror, p_init=p_init)


def fit_pol2(datax, datay, datayerror=None, p_init=None):
    if datayerror is None:
        datayerror = np.ones_like(datay)
    func = lambda x, p0, p1, p2: x ** 2 * p2 + x * p1 + p0
    return fit_chi2(func, datax, datay, datayerror, p_init=p_init)

#     popt, cov = optimize.curve_fit(lambda x, p0, p1: x * p1 + p0, datax, datay,
#                                    sigma=datayerror)
#     y_fit = popt[1] * datax + popt[0]
#     chi2 = np.sum((y_fit - datay) ** 2 / datayerror ** 2)
#     chi2ndf = 1. * chi2 / (len(datax) - 2)
#     func = lambda x: popt[1] * x + popt[0]
#     return popt, cov, chi2ndf, func


def fit_gaus(datax, datay, datayerror, p_init=None):
    func = lambda x, p0, p1, p2: p0 * (2 * np.pi * p2) ** -0.5 * np.exp(-0.5 * (x - p1) ** 2 * p2 ** -2)
#     if(p_init == None):
#         p_init = [datay.max()*(2 * np.pi * datay.std())**0.5, datay.mean(), datay.std()]
    return fit_chi2(func, datax, datay, datayerror, p_init=p_init)


def plot_fit_stats(ax, popt, cov, chi2ndf, posx=0.95, posy=0.95, ha='right',
                   significant_figure=False, color='k', funcstring='', parnames=None):
    textstr = ''
    if(funcstring != ''):
        textstr += funcstring + "\n"
    textstr += "$\chi^2/ndf=%.2g$" % chi2ndf

    if(len(popt) == 1):
        parname = "p0"
        if(parnames):
            parname = parnames[0]
        textstr += "\n$%s = %.2g \pm %.2g$" % (parname, popt[0], np.squeeze(cov) ** 0.5)
    elif(len(cov) == 1):
        for i, p in enumerate(popt):
            parname = "p%i" % i
            if(parnames):
                parname = parnames[i]
            textstr += "\n$%s = %.2g \pm %s$" % (parname, p, 'nan')
    else:
        for i, p in enumerate(popt):
            parname = "p%i" % i
            if(parnames):
                parname = parnames[i]
            if(significant_figure):
                import SignificantFigures as serror
                textstr += "\n$%s = %s \pm %s$" % ((parname,) + (serror.formatError(p, cov[i, i] ** 0.5)))
            else:
                textstr += "\n$%s = %.2g \pm %.2g$" % (parname, p, cov[i, i] ** 0.5)
    props = dict(boxstyle='square', facecolor='wheat', alpha=0.5)
    ax.text(posx, posy, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', horizontalalignment=ha,
                multialignment='left', bbox=props, color=color)


def plot_fit_stats2(ax, popt, cov, chi2ndf, posx=0.95, posy=0.95):
    from MatplotlibTools import Figures
    plt.rc('text', usetex=True)
    table = Figures.Table()

    table.addValue("$\chi^2/ndf$", chi2ndf)
    if(len(cov) == 1):
        for i, p in enumerate(popt):
            table.addValue("p%i" % i, p)
    else:
        for i, p in enumerate(popt):
            table.addValue("p%i" % i, p, cov[i, i] ** 0.5)
    props = dict(boxstyle='square', facecolor='wheat', alpha=0.5)
#     print(table.getTable())
    ax.text(posx, posy, r'$%s$' % table.getTable(), transform=ax.transAxes, fontsize=14,
                verticalalignment='top', horizontalalignment='right',
                multialignment='left', bbox=props)


def plot_hist_stats(ax, data, weights=None, posx=0.05, posy=0.95, overflow=None,
                    underflow=None, rel=False,
                    additional_text="", additional_text_pre="",
                    fontsize=12, color="k", va="top", ha="left",
                    median=True, quantiles=True, mean=True, std=True, N=True,
                    single_sided=False):
    data = np.array(data)
    textstr = additional_text_pre
    if (textstr != ""):
        textstr += "\n"
    if N:
        textstr += "$N=%i$\n" % data.size
    if not single_sided:
        tmean = data.mean()
        tstd = data.std()
        if weights is not None:

            def weighted_avg_and_std(values, weights):
                """
                Return the weighted average and standard deviation.

                values, weights -- Numpy ndarrays with the same shape.
                """
                average = np.average(values, weights=weights)
                variance = np.average((values - average) ** 2, weights=weights)  # Fast and numerically precise
                return (average, variance ** 0.5)

            tmean, tstd = weighted_avg_and_std(data, weights)

    #     import SignificantFigures as serror
        if mean:
            if weights is None:
    #             textstr += "$\mu = %s \pm %s$\n" % serror.formatError(tmean,
    #                                                 tstd / math.sqrt(data.size))
                textstr += "$\mu = {:.3g}$\n".format(tmean)
            else:
                textstr += "$\mu = {:.3g}$\n".format(tmean)
        if median:
            tweights = np.ones_like(data)
            if weights is not None:
                tweights = weights
            if quantiles:
                q1 = radiotools.stats.quantile_1d(data, tweights, 0.16)
                q2 = radiotools.stats.quantile_1d(data, tweights, 0.84)
                median = radiotools.stats.median(data, tweights)
    #             median_str = serror.formatError(median, 0.05 * (np.abs(median - q2) + np.abs(median - q1)))[0]
                textstr += "$\mathrm{median} = %.3g^{+%.2g}_{-%.2g}$\n" % (median, np.abs(median - q2),
                                                                           np.abs(median - q1))
            else:
                textstr += "$\mathrm{median} = %.3g $\n" % radiotools.stats.median(data, tweights)
        if std:
            if rel:
                textstr += "$\sigma = %.2g$ (%.1f\%%)\n" % (tstd, tstd / tmean * 100.)
            else:
                textstr += "$\sigma = %.2g$\n" % (tstd)
    else:
        if(weights is None):
            w = np.ones_like(data)
        else:
            w = weights
        q68 = radiotools.stats.quantile_1d(data, weights=w, quant=.68)
        q95 = radiotools.stats.quantile_1d(data, weights=w, quant=.95)
        textstr += "$\sigma_\mathrm{{68}}$ = {:.1f}$^\circ$\n".format(q68)
        textstr += "$\sigma_\mathrm{{95}}$ = {:.1f}$^\circ$\n".format(q95)

    if(overflow):
        textstr += "$\mathrm{overflows} = %i$\n" % overflow
    if(underflow):
        textstr += "$\mathrm{underflows} = %i$\n" % underflow

    textstr += additional_text
    textstr = textstr[:-1]

    props = dict(boxstyle='square', facecolor='w', alpha=0.5)
    ax.text(posx, posy, textstr, transform=ax.transAxes, fontsize=fontsize,
            verticalalignment=va, ha=ha, multialignment='left',
            bbox=props, color=color)


def get_graph(x_data, y_data, filename=None, show=False, xerr=None, yerr=None, funcs=None,
               xlabel="", ylabel="", title="", fmt='bo', fit_stats=None,
               xmin=None, xmax=None, ymin=None, ymax=None, hlines=None,
               **kwargs):
    fig, ax1 = plt.subplots(1, 1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    ax1.errorbar(x_data, y_data, xerr=xerr, yerr=yerr, fmt='bo', **kwargs)
    if(not xmin is None):
        ax1.set_xlim(left=xmin)
    if(not xmax is None):
        ax1.set_xlim(right=xmax)
    if(not ymin is None):
        ax1.set_ylim(bottom=ymin)
    if(not ymax is None):
        ax1.set_ylim(top=ymax)

    if(funcs):
        for func in funcs:
            xlim = np.array(ax1.get_xlim())
            xx = np.linspace(xlim[0], xlim[1], 100)
            if('args' in func):
                ax1.plot(xx, func['func'](xx), *func['args'])
                if('kwargs' in func):
                    ax1.plot(xx, func['func'](xx), *func['args'], **func['kwargs'])
            else:
                ax1.plot(xx, func['func'](xx))
    if(hlines):
        for hline in hlines:
            xlim = np.array(ax1.get_xlim())
            ax1.hlines(hline['y'], *xlim, **hline['kwargs'])
    if(fit_stats):
        plot_fit_stats(ax1, *fit_stats)
    plt.tight_layout()
    if(show):
        plt.show()
    return fig, ax1


def save_graph(filename=None, **kwargs):
    fig, ax1 = get_graph(**kwargs)
    if(filename and filename != ""):
        fig.savefig(filename)
    plt.close(fig)


def get_histograms(histograms, bins=None, xlabels=None, ylabels=None, stats=True,
                   fig=None, axes=None, histtype=u'bar', titles=None,
                   weights=None, figsize=4,
                   stat_kwargs=None,
                   kwargs={'facecolor': '0.7', 'alpha': 1, 'edgecolor': "k"}):
    N = len(histograms)
    if((fig is None) or (axes is None)):
        if(N == 1):
            fig, axes = get_histogram(histograms[0], bins=bins, xlabel=xlabels, ylabel=ylabels, title=titles,
                                      stats=stats, weights=weights)
            return fig, axes
        elif(N <= 3):
            fig, axes = plt.subplots(1, N, figsize=(figsize * N, figsize))
        elif(N == 4):
            fig, axes = plt.subplots(2, 2, figsize=(figsize * 2, figsize * 2))
        elif(N <= 6):
            fig, axes = plt.subplots(2, 3, figsize=(figsize * 3, figsize * 2))
        elif(N <= 8):
            fig, axes = plt.subplots(2, 4, figsize=(figsize * 4, figsize * 2))
        elif(N <= 9):
            fig, axes = plt.subplots(3, 3, figsize=(figsize * 3, figsize * 3))
        elif(N <= 12):
            fig, axes = plt.subplots(3, 4, figsize=(figsize * 4, figsize * 3))
        elif(N <= 16):
            fig, axes = plt.subplots(4, 4, figsize=(figsize * 4, figsize * 4))
        elif(N <= 20):
            fig, axes = plt.subplots(4, 5, figsize=(figsize * 5, figsize * 4))
        elif(N <= 25):
            fig, axes = plt.subplots(5, 5, figsize=(figsize * 5, figsize * 5))
        elif(N <= 30):
            fig, axes = plt.subplots(5, 5, figsize=(figsize * 6, figsize * 5))
        elif(N <= 35):
            fig, axes = plt.subplots(5, 7, figsize=(figsize * 7, figsize * 5))
        else:
            print("WARNING: more than 35 pads are not implemented")
            raise "WARNING: more than 35 pads are not implemented"
    shape = np.array(np.array(axes).shape)
    n1 = shape[0]
    n2 = 1
    if(len(shape) == 2):
        n2 = shape[1]
    axes = np.reshape(axes, n1 * n2)

    for i in range(N):
        xlabel = ""
        if xlabels:
            if(type(xlabels) != str):
                xlabel = xlabels[i]
            else:
                xlabel = xlabels
        ylabel = "entries"
        if ylabels:
            if(len(ylabels) > 1):
                ylabel = ylabels[i]
            else:
                ylabel = ylabels
        tbin = 10
        title = ""
        if titles:
            title = titles[i]
        if bins is not None:
            if(type(bins) == int):
                tbin = bins
            else:
                if (isinstance(bins[0], float) or isinstance(bins[0], int)):
                    tbin = bins
                else:
                    tbin = bins[i]
        if(len(histograms[i])):
            tweights = None
            if weights is not None:
                if (isinstance(weights[0], float) or isinstance(weights[0], int)):
                    tweights = weights
                else:
                    tweights = weights[i]
            n, binst, patches = axes[i].hist(histograms[i], bins=tbin,
                                             histtype=histtype,
                                             weights=tweights, **kwargs)
            axes[i].set_xlabel(xlabel)
            axes[i].set_ylabel(ylabel)
            ymax = max(axes[i].get_ylim()[1], 1.2 * n.max())
            axes[i].set_ylim(0, ymax)
            axes[i].set_xlim(binst[0], binst[-1])
            axes[i].set_title(title)
            underflow = np.sum(histograms[i] < binst[0])
            overflow = np.sum(histograms[i] > binst[-1])
            if isinstance(stats, bool):
                if stats:
                    if(stat_kwargs is None):
                        plot_hist_stats(axes[i], histograms[i], weights=tweights,
                                        overflow=overflow, underflow=underflow)
                    else:
                        plot_hist_stats(axes[i], histograms[i], weights=tweights,
                                        overflow=overflow, underflow=underflow, **stat_kwargs)
            else:
                if(stats[i]):
                    if(stat_kwargs is None):
                        plot_hist_stats(axes[i], histograms[i], weights=tweights,
                                        overflow=overflow, underflow=underflow)
                    else:
                        plot_hist_stats(axes[i], histograms[i], weights=tweights,
                                        overflow=overflow, underflow=underflow, **stat_kwargs)
    fig.tight_layout()
    return fig, axes


def get_histogram(data, bins=10, xlabel="", ylabel="entries", weights=None,
                  title="", stats=True, show=False, stat_kwargs=None, funcs=None, overflow=True,
                  ax=None, kwargs={'facecolor':'0.7', 'alpha':1, 'edgecolor':"k"},
                  figsize=None):
    """ creates a histogram using matplotlib from array """
    if(ax is None):
        if figsize is None:
            fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)
    else:
        ax1 = ax

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    n, bins, patches = ax1.hist(data, bins, weights=weights, **kwargs)
    if(funcs):
        for func in funcs:
            xlim = np.array(ax1.get_xlim())
            xx = np.linspace(xlim[0], xlim[1], 100)
            if('args' in func):
                ax1.plot(xx, func['func'](xx), *func['args'])
                if('kwargs' in func):
                    ax1.plot(xx, func['func'](xx), *func['args'], **func['kwargs'])
            else:
                ax1.plot(xx, func['func'](xx))
    ax1.set_ylim(0, n.max() * 1.2)
    ax1.set_xlim(bins[0], bins[-1])

    if stats:
        if overflow:
            underflow = np.sum(data < bins[0])
            overflow = np.sum(data > bins[-1])
        else:
            underflow = None
            underflow = None
        if(stat_kwargs is None):
            plot_hist_stats(ax1, data, overflow=overflow, underflow=underflow, weights=weights)
        else:
            plot_hist_stats(ax1, data, overflow=overflow, underflow=underflow, weights=weights, **stat_kwargs)
    if(show):
        plt.show()
    if(ax is None):
        return fig, ax1


def get_2dhist_normalized_columns(X, Y, fig, ax, binsx, binsy, shading='flat', clim=(None, None), norm=None, cmap=None):
    """
    creates a 2d histogram where the number of entries are normalized to 1 per column

    Parameters
    ----------
    X: array
        x values
    Y: array
        y values
    fig: figure instance
        the figure to plot in
    ax: axis instance
        the axis to plot in
    binsx: array
        the x bins
    binsy: array
        the y bins
    shading: string
        fill style {'flat', 'gouraud'}, see matplotlib documentation (default flat)
    clim: tuple, list
        limits for the color axis (default (None, None))
    norm: None or Normalize instance (e.g. matplotlib.colors.LogNorm()) (default None)
        normalization of the color scale
    cmap: string or None
        the name of the colormap

    Returns
    --------
    pcolormesh object, colorbar object
    """
    H, xedges, yedges = np.histogram2d(X, Y, bins=[binsx, binsy])
    np.nan_to_num(H)
#         Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
    Hmasked = H
    H_norm_rows = Hmasked / np.outer(Hmasked.sum(axis=1, keepdims=True), np.ones(H.shape[1]))

    if(cmap is not None):
        cmap = plt.get_cmap(cmap)

    vmin, vmax = clim
    pc = ax.pcolormesh(xedges, yedges, H.T, shading=shading, vmin=vmin, vmax=vmax , norm=norm, cmap=cmap)
    cb = fig.colorbar(pc, ax=ax, orientation='vertical')

    return pc, cb


def get_histogram2d(x=None, y=None, z=None,
                bins=10, range=None,
                xscale="linear", yscale="linear", cscale="linear",
                normed=False, cmap=None, clim=(None, None),
                ax1=None, grid=True, shading='flat', colorbar={},
                cbi_kwargs={'orientation': 'vertical'},
                xlabel="", ylabel="", clabel="", title="",
                fname="hist2d.png"):
    """
    creates a 2d histogram

    Parameters
    ----------
    x, y, z :
        x and y coordinaten for z value, if z is None the 2d histogram of x and z is calculated

    numpy.histogram2d parameters:
        range : array_like, shape(2,2), optional
        bins : int or array_like or [int, int] or [array, array], optional

    ax1: mplt.axes
        if None (default) a olt.figure is created and histogram is stored
        if axis is give, the axis and a pcolormesh object is returned

    colorbar : dict

    plt.pcolormesh parameters:
        clim=(vmin, vmax) : scalar, optional, default: clim=(None, None)
        shading : {'flat', 'gouraud'}, optional

    normed: string
        colum, row, colum1, row1 (default: None)

    {x,y,c}scale: string
        'linear', 'log' (default: 'linear')


    """

    if z is None and (x is None or y is None):
        sys.exit("z and (x or y) are all None")

    if ax1 is None:
        fig, ax = plt.subplots(1)
        fig.subplots_adjust(wspace=0.3, left=0.05, right=0.95)
    else:
        ax = ax1

    if z is None:
        z, xedges, yedges = np.histogram2d(x, y, bins=bins, range=range)
        z = z.T
    else:
        xedges, yedges = x, y

    if normed:
        if normed == "colum":
            z = z / np.sum(z, axis=0)
        elif normed == "row":
            z = z / np.sum(z, axis=1)[:, None]
        elif normed == "colum1":
            z = z / np.amax(z, axis=0)
        elif normed == "row1":
            z = z / np.amax(z, axis=1)[:, None]
        else:
            sys.exit("Normalisation %s is not known.")

    color_norm = mcolors.LogNorm() if cscale == "log" else None
    vmin, vmax = clim
    im = ax.pcolormesh(xedges, yedges, z, shading=shading, vmin=vmin, vmax=vmax, norm=color_norm, cmap=cmap)

    if colorbar is not None:
        cbi = plt.colorbar(im, **cbi_kwargs)
        cbi.ax.tick_params(axis='both', **{"labelsize": 30})
        cbi.set_label(clabel)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    ax.set_title(title)

    if ax is None:
        save_histogram(fig, fname)
    else:
        return ax, im


def save_histogram(filename, *args, **kwargs):
    fig, ax = get_histogram(*args, **kwargs)
    fig.savefig(filename)
    plt.close(fig)


def get_subplot(figsize=None):
        fig, ax = plt.subplots(1, figsize=figsize)
        fig.subplots_adjust(wspace=0.3, left=0.05, right=0.95)
        return fig, ax


def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)


def make_dir(path):
    if(not os.path.isdir(path)):
        os.makedirs(path)


def get_marker(i):
    colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    markers = ["o", "D", "^", "s", ">"]
    return colors[i % len(colors)] + markers[i // len(colors)]


def get_marker_only(i):
    markers = ["o", "D", "^", "s", ">"]
    return markers[i % len(markers)]


def get_marker2(i):
    colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    markers = ["o", "D", "^", "s", ">"]
    return colors[i % len(colors)] + markers[i % len(markers)]


def get_color(i):
    colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    return colors[i % len(colors)]


def get_color_linestyle(i):
    colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    markers = ["-", "--", ":", "-."]
    return markers[i % len(markers)] + colors[i % len(colors)]


def get_color_linestyle2(i):
    colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    markers = ["-", "--", ":", "-."]
    return markers[i % len(markers)] + colors[i // len(markers)]
