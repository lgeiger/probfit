# -*- coding: utf-8 -*-
# Plotting is on python since this will make it much easier to debug and adjust
# no need to recompile everytime i change graph color....

# needs a serious refactor
from matplotlib import pyplot as plt
import numpy as np
from .nputil import mid, minmax, vector_apply
from util import parse_arg, describe
from math import sqrt, ceil, floor
from warnings import warn


def draw_simultaneous(self, minuit=None, args=None, errors=None, **kwds):
    numf = len(self.allf)
    ret = []
    numraw = sqrt(numf)
    numcol = ceil(numraw)
    numrow = floor(numraw) if floor(numraw) * numcol >= numf else ceil(numraw)

    for i in range(numf):
        plt.subplot(numrow, numcol, i + 1)
        part_args, part_errors = self.args_and_error_for(i, minuit, args, errors)
        ret.append(self.allf[i].draw(args=part_args, errors=part_errors, **kwds))

    return ret


def _get_args_and_errors(self, minuit=None, args=None, errors=None):
    """
    consistent algorithm to get argument and errors
    1) get it from minuit if minuit is available
    2) if not get it from args and errors
    2.1) if args is dict parse it.
    3) if all else fail get it from self.last_arg
    """
    ret_arg = None
    ret_error = None
    if minuit is not None:  # case 1
        ret_arg = minuit.args
        ret_error = minuit.errors
        return ret_arg, ret_error

    # no minuit specified use args and errors
    if args is not None:
        if isinstance(args, dict):
            ret_arg = parse_arg(self, args)
        else:
            ret_arg = args
    else:  # case 3
        ret_arg = self.last_arg

    if errors is not None:
        ret_error = errors

    return ret_arg, ret_error


def _param_text(parameters, arg, error):
    txt = u''
    for (k, v) in zip(parameters, arg):
        txt += u'%s = %5.4g' % (k, v)
        if error is not None:
            txt += u'±%5.4g' % error[k]
        txt += u'\n'
    return txt


# from UML
def draw_ulh(self, minuit=None, bins=100, ax=None, bound=None,
             parmloc=(0.05, 0.95), nfbins=200, print_par=True, grid=True,
             args=None, errors=None, parts=False, show_errbars=True,
             errbar_algo='normal', no_plot=False):

    part_ret = []
    arg, error = _get_args_and_errors(self, minuit, args, errors)

    n, e = np.histogram(self.data, bins=bins, range=bound, weights=self.weights)
    data_ret = (e, n)
    ef = np.linspace(e[0], e[-1], nfbins + 1)

    if self.extended:
        scale = nfbins / float(bins)
    else:
        scale = (n * np.diff(e)).sum()

    if errbar_algo == 'normal':
        w2 = n
    elif errbar_algo == 'sumw2':
        weights = None
        if self.weights is not None:
            weights = self.weights**2
        w2, e = np.histogram(self.data, bins=e, weights=weights)
    else:
        raise ValueError('errbar_algo must be \'normal\' or \'sumw2\'')
    data_err = np.sqrt(w2)

    # Draw data
    draw_arg = [('lw', 2)]
    if not parts:
        draw_arg.append(('color', 'r'))

    if not no_plot:
        ax = plt.gca() if ax is None else ax
        if show_errbars:
            ax.errorbar(mid(e), n, data_err, fmt='b.', capsize=0)
        else:
            ax.hist(mid(e), bins=e, weights=n, histtype='step')
        ax.grid(grid)
        ax.set_axisbelow(True)
        if print_par:
            txt = _param_text(describe(self), arg, error)
            ax.text(parmloc[0], parmloc[1], txt, ha='left', va='top',
                    transform=ax.transAxes)

    # Draw pdf with finer bins
    total_ret = draw_pdf_with_edges(self.f, arg, ef, ax=ax,
                                    density=not self.extended, scale=scale,
                                    no_plot=no_plot, **dict(draw_arg))

    if parts:
        f_parts = getattr(self.f, 'parts', None)
        if f_parts is not None:
            for p in f_parts():
                ret = draw_pdf_with_edges(p, arg, ef, ax=ax, scale=scale,
                                          density=not self.extended,
                                          no_plot=no_plot)
                part_ret.append(ret)

    return (data_ret, data_err, total_ret, part_ret)


def draw_residual_ulh(self, minuit=None, bins=100, ax=None, bound=None,
                      parmloc=(0.05, 0.95), print_par=False, grid=True,
                      args=None, errors=None, show_errbars=True,
                      errbar_algo='normal', norm=False):

    arg, error = _get_args_and_errors(self, minuit, args, errors)

    n, e = np.histogram(self.data, bins=bins, range=bound, weights=self.weights)

    if self.extended:
        scale = np.diff(e)
    else:
        scale = (n * np.diff(e)).sum()

    if errbar_algo == 'normal':
        w2 = n
    elif errbar_algo == 'sumw2':
        weights = None
        if self.weights is not None:
            weights = self.weights**2
        w2, e = np.histogram(self.data, bins=e, weights=weights)
    else:
        raise ValueError('errbar_algo must be \'normal\' or \'sumw2\'')
    yerr = np.sqrt(w2)

    yf = vector_apply(self.f, mid(e), *arg) * scale
    n = n - yf
    if norm:
        sel = yerr > 0
        n[sel] /= yerr[sel]
        yerr = np.ones(len(yerr))
    data_ret = e, n

    ax = plt.gca() if ax is None else ax
    if show_errbars:
        ax.errorbar(mid(e), n, yerr, fmt='b.', capsize=0)
    else:
        ax.bar(e[:-1], n, width=np.diff(e))
    ax.plot([e[0], e[-1]], [0., 0.], 'r-')
    ax.grid(grid)
    ax.set_axisbelow(True)
    if print_par:
        txt = _param_text(describe(self), arg, error)
        ax.text(parmloc[0], parmloc[1], txt, ha='left', va='top',
                transform=ax.transAxes)

    return (data_ret, yerr)


# from chi2 regression
def draw_x2(self, minuit=None, ax=None, parmloc=(0.05, 0.95), print_par=True,
            args=None, errors=None, grid=True, parts=False, no_plot=False):

    part_ret = []
    arg, error = _get_args_and_errors(self, minuit, args, errors)

    data_err = self.error if self.error is not None else np.ones(len(self.x))
    data_ret = self.x, self.y

    draw_arg = [('lw', 2)]
    if not parts:
        draw_arg.append(('color', 'r'))

    if not no_plot:
        ax = plt.gca() if ax is None else ax
        ax.errorbar(self.x, self.y, data_err, fmt='.')
        ax.grid(grid)
        ax.set_axisbelow(True)
        if print_par:
            txt = _param_text(describe(self), arg, error)
            chi2 = self(*arg)
            if self.ndof > 0:
                txt += u'chi2/ndof = %5.4g(%5.4g/%d)' % (chi2 / self.ndof, chi2, self.ndof)
            else:
                txt += u'chi2/ndof = (%5.4g/%d)' % (chi2, self.ndof)

            ax.text(parmloc[0], parmloc[1], txt, ha='left', va='top',
                    transform=ax.transAxes)

    total_ret = draw_pdf_with_midpoints(self.f, arg, self.x, ax=ax,
                                        no_plot=no_plot, **dict(draw_arg))

    if parts:
        f_parts = getattr(self.f, 'parts', None)
        if f_parts is not None:
            for p in f_parts():
                tmp = draw_pdf_with_midpoints(p, arg, self.x, ax=ax,
                                              no_plot=no_plot, **dict(draw_arg))
                part_ret.append(tmp)

    return (data_ret, data_err, total_ret, part_ret)


def draw_x2_residual(self, minuit=None, ax=None, args=None, errors=None,
                     grid=True, norm=False):

    arg, _ = _get_args_and_errors(self, minuit, args, errors)

    data_err = self.error if self.error is not None else np.ones(len(self.x))

    yf = vector_apply(self.f, self.x, *arg)
    yplot = self.y - yf
    data_ret = self.x, yplot
    if norm:
        yplot = yplot / data_err
        data_err = np.ones(len(data_err))

    # Make plot
    ax = plt.gca() if ax is None else ax
    ax.errorbar(self.x, yplot, data_err, fmt='.')
    ax.grid(grid)
    ax.set_axisbelow(True)

    return (data_ret, data_err)


# from binned chi2
def draw_bx2(self, minuit=None, parmloc=(0.05, 0.95), nfbins=500, ax=None,
             print_par=True, args=None, errors=None, parts=False, grid=True,
             no_plot=False):

    part_ret = []
    arg, error = _get_args_and_errors(self, minuit, args, errors)

    m = mid(self.edges)
    data_ret = (self.edges, self.h)

    bound = (self.edges[0], self.edges[-1])
    scale = nfbins / float(self.bins)  # scale back to bins

    draw_arg = [('lw', 2)]
    if not parts:
        draw_arg.append(('color', 'r'))

    if not no_plot:
        ax = plt.gca() if ax is None else ax
        ax.errorbar(m, self.h, self.err, fmt='.')
        ax.grid(grid)
        ax.set_axisbelow(True)
        if print_par:
            txt = _param_text(describe(self), arg, error)
            chi2 = self(*arg)
            if self.ndof > 0:
                txt += u'chi2/ndof = %5.4g(%5.4g/%d)' % (chi2 / self.ndof, chi2, self.ndof)
            else:
                txt += u'chi2/ndof = (%5.4g/%d)' % (chi2, self.ndof)

            ax.text(parmloc[0], parmloc[1], txt, ha='left', va='top',
                    transform=ax.transAxes)

    total_ret = draw_pdf(self.f, arg, bins=nfbins, bound=bound, ax=ax,
                         density=False, scale=scale, no_plot=no_plot,
                         **dict(draw_arg))

    if parts:
        f_parts = getattr(self.f, 'parts', None)
        if f_parts is not None:
            for p in f_parts():
                tmp = draw_pdf(p, arg, bound=bound, bins=nfbins, ax=ax,
                               density=False, scale=scale, no_plot=no_plot)
                part_ret.append(tmp)

    return (data_ret, self.err, total_ret, part_ret)


# from binnedLH
def draw_blh(self, minuit=None, parmloc=(0.05, 0.95),
             nfbins=1000, ax=None, print_par=True, grid=True,
             args=None, errors=None, parts=False, no_plot=False):

    part_ret = []
    arg, error = _get_args_and_errors(self, minuit, args, errors)

    m = mid(self.edges)
    n = np.copy(self.h)

    bound = (self.edges[0], self.edges[-1])
    data_ret = (self.edges, n)

    if self.use_w2:
        data_err = np.sqrt(self.w2)
    else:
        data_err = np.sqrt(self.h)

    if self.extended:
        scale = nfbins / float(self.bins)
    else:
        scale = (n * np.diff(self.edges)).sum()

    draw_arg = [('lw', 2)]
    if not parts:
        draw_arg.append(('color', 'r'))

    if not no_plot:
        ax = plt.gca() if ax is None else ax
        ax.errorbar(m, n, data_err, fmt='.')
        ax.grid(grid)
        ax.set_axisbelow(True)
        if print_par:
            txt = _param_text(describe(self), arg, error)
            ax.text(parmloc[0], parmloc[1], txt, ha='left', va='top',
                    transform=ax.transAxes)

    total_ret = draw_pdf(self.f, arg, bins=nfbins, bound=bound, ax=ax,
                         density=not self.extended, scale=scale,
                         no_plot=no_plot, **dict(draw_arg))

    if parts:
        f_parts = getattr(self.f, 'parts', None)
        if f_parts is not None:
            for p in f_parts():
                tmp = draw_pdf(p, arg, bins=nfbins, bound=bound, ax=ax,
                               density=not self.extended, scale=scale,
                               no_plot=no_plot)
                part_ret.append(tmp)

    return (data_ret, data_err, total_ret, part_ret)


def draw_residual_blh(self, minuit=None, parmloc=(0.05, 0.95),
                      ax=None, print_par=False, args=None, errors=None,
                      norm=False, grid=True):

    arg, error = _get_args_and_errors(self, minuit, args, errors)

    m = mid(self.edges)
    n = np.copy(self.h)

    if self.use_w2:
        data_err = np.sqrt(self.w2)
    else:
        data_err = np.sqrt(self.h)

    if self.extended:
        scale = np.diff(self.edges)
    else:
        scale = (n * np.diff(self.edges)).sum()

    yf = vector_apply(self.f, m, *arg) * scale
    n = n - yf
    if norm:
        sel = data_err > 0
        n[sel] /= data_err[sel]
        data_err = np.ones(len(data_err))
    data_ret = self.edges, n

    # make plot
    ax = plt.gca() if ax is None else ax
    ax.errorbar(m, n, data_err, fmt='.')
    ax.plot([self.edges[0], self.edges[-1]], [0., 0.], 'r-')
    ax.grid(grid)
    ax.set_axisbelow(True)
    if print_par:
        txt = _param_text(describe(self), arg, error)
        ax.text(parmloc[0], parmloc[1], txt, ha='left', va='top',
                transform=ax.transAxes)

    return (data_ret, data_err)


def draw_compare(f, arg, edges, data, errors=None, ax=None, grid=True,
                 normed=False, parts=False):
    """
    TODO: this needs to be rewritten
    """
    # arg is either map or tuple
    ax = plt.gca() if ax is None else ax
    arg = parse_arg(f, arg, 1) if isinstance(arg, dict) else arg
    x = (edges[:-1] + edges[1:]) / 2.0
    bw = np.diff(edges)
    yf = vector_apply(f, x, *arg)
    total = np.sum(data)
    if normed:
        ax.errorbar(x, data / bw / total, errors / bw / total, fmt='.b')
        ax.plot(x, yf, 'r', lw=2)
    else:
        ax.errorbar(x, data, errors, fmt='.b')
        ax.plot(x, yf * bw, 'r', lw=2)

    # now draw the parts
    if parts:
        if not hasattr(f, 'eval_parts'):
            warn(RuntimeWarning('parts is set to True but function does '
                                'not have eval_parts method'))
        else:
            scale = bw if not normed else 1.
            parts_val = list()
            for tx in x:
                val = f.eval_parts(tx, *arg)
                parts_val.append(val)
            py = zip(*parts_val)
            for y in py:
                tmpy = np.array(y)
                ax.plot(x, tmpy * scale, lw=2, alpha=0.5)
    ax.grid(grid)
    ax.set_axisbelow(True)

    return x, yf, data


def draw_normed_pdf(f, arg, bound, bins=100, scale=1.0, density=True, ax=None,
                    no_plot=False, **kwds):
    return draw_pdf(f, arg, bound, bins=100, scale=1.0, density=True,
                    normed_pdf=True, ax=ax, no_plot=no_plot, **kwds)


def draw_pdf(f, arg, bound, bins=100, scale=1.0, density=True,
             normed_pdf=False, ax=None, no_plot=False, **kwds):
    """
    draw pdf with given argument and bounds.

    **Arguments**

        * **f** your pdf. The first argument is assumed to be independent
          variable

        * **arg** argument can be tuple or list

        * **bound** tuple(xmin,xmax)

        * **bins** number of bins to plot pdf. Default 100.

        * **scale** multiply pdf by given number. Default 1.0.

        * **density** plot density instead of expected count in each bin
          (pdf*bin width). Default True.

        * **normed_pdf** Normalize pdf in given bound. Default False

        * The rest of keyword argument will be pass to pyplot.plot

    **Returns**

        x, y of what's being plot
    """
    edges = np.linspace(bound[0], bound[1], bins)
    return draw_pdf_with_edges(f, arg, edges, ax=ax, scale=scale,
                               density=density, normed_pdf=normed_pdf,
                               no_plot=no_plot, **kwds)


def draw_pdf_with_edges(f, arg, edges, ax=None, scale=1.0, density=True,
                        normed_pdf=False, no_plot=False, **kwds):
    x = mid(edges)
    bw = np.diff(edges)
    scale *= bw if not density else 1.

    return draw_pdf_with_midpoints(f, arg, x, ax=ax, scale=scale,
                                   normed_pdf=normed_pdf, no_plot=no_plot,
                                   **kwds)


def draw_pdf_with_midpoints(f, arg, x, ax=None, scale=1.0, normed_pdf=False,
                            no_plot=False, **kwds):
    arg = parse_arg(f, arg, 1) if isinstance(arg, dict) else arg
    yf = vector_apply(f, x, *arg)

    if normed_pdf:
        normed_factor = sum(yf)  # assume equal binwidth
        yf /= normed_factor
    yf *= scale

    if not no_plot:
        ax = plt.gca() if ax is None else ax
        ax.plot(x, yf, **kwds)
    return x, yf


# draw comparison between function given args and data
def draw_compare_hist(f, arg, data, bins=100, bound=None, ax=None, weights=None,
                      normed=False, use_w2=False, parts=False, grid=True):
    """
    draw histogram of data with poisson error bar and f(x,*arg).

    ::

        data = np.random.rand(10000)
        f = gaussian
        draw_compare_hist(f, {'mean':0,'sigma':1}, data, normed=True)

    **Arguments**

        - **f**
        - **arg** argument pass to f. Can be dictionary or list.
        - **data** data array
        - **bins** number of bins. Default 100.
        - **bound** optional boundary of plot in tuple form. If `None` is
          given, the bound is determined from min and max of the data. Default
          `None`
        - **weights** weights array. Default None.
        - **normed** optional normalized data flag. Default False.
        - **use_w2** scaled error down to the original statistics instead of
          weighted statistics.
        - **parts** draw parts of pdf. (Works with AddPdf and Add2PdfNorm).
          Default False.
    """
    ax = plt.gca() if ax is None else ax
    bound = minmax(data) if bound is None else bound
    h, e = np.histogram(data, bins=bins, range=bound, weights=weights)
    err = None
    if weights is not None and use_w2:
        err, _ = np.histogram(data, bins=bins, range=bound, weights=weights**2)
        err = np.sqrt(err)
    else:
        err = np.sqrt(h)
    return draw_compare(f, arg, e, h, err, ax=ax, grid=grid, normed=normed,
                        parts=parts)
