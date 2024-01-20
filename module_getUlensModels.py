import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import corner

import urllib, base64
import io
import os

import mechanize
import requests

import json

import time
from datetime import datetime

from astropy.time import Time

from collections import OrderedDict

import MulensModel as mm

try:
    from pymultinest.solve import solve
    from pymultinest.analyse import Analyzer
except ImportError as err:
    print(err)
    print("\nPyMultiNest could not be imported.")
    print("Get it from: https://github.com/JohannesBuchner/PyMultiNest")
    print("and re-run the script")
    sys.exit(1)


def fit_pspl(name, psplStart, datasets, zeroBlendingDict):
	# Starting parameters:
	params = dict()
	params['t_0'] = psplStart[0]
	params['u_0'] = psplStart[1]
	params['t_E'] = psplStart[2]
	my_model = mm.Model(params)
	my_event = mm.Event(datasets=datasets, model=my_model, fix_blend_flux=zeroBlendingDict)
	# Which parameters we want to fit?
	parameters_to_fit = ["t_0", "u_0", "t_E"]
	# Min and max values
	min_values = np.array([2454000., -1.5, 0.])
	max_values = np.array([2460000., 1.5, 1000.])
	
	# Setting up minimizer
	minimizer = Minimizer(my_event, parameters_to_fit)
	minimizer.set_cube(min_values, max_values)
	minimizer.set_chi2_0()
	
	dir_out = "chains/"
	if not os.path.exists(dir_out):
	    os.mkdir(dir_out)
	file_prefix = "%s_pspl"%(name)
	
	# we save the results to a file
	start_time = time.time()
	
	# Run MultiNest:
	run_kwargs = {
	'LogLikelihood': minimizer.ln_likelihood,
	'Prior': minimizer.transform_cube,
	'n_dims': len(parameters_to_fit),
	'resume': False,
	'importance_nested_sampling': False,
	'outputfiles_basename': os.path.join(dir_out, file_prefix+"_"),
	'multimodal': True,
	'n_live_points': 400}
	result = solve(**run_kwargs)
	
	# Analyze results - we print each mode separately and give log-evidence:
	analyzer = Analyzer(n_params=run_kwargs['n_dims'], outputfiles_basename=run_kwargs['outputfiles_basename'])
					
	log = "multinest_log.dat"
	logOutput = open(log, "a")
	print("--- Multinest took %s seconds ---" % (time.time() - start_time))
	logOutput.write("%s : PSPL, sourceid: %s "%(datetime.now(), str(name)))
	logOutput.write("--- Multinest took %s seconds --- \n" % (time.time() - start_time))
	logOutput.close()
	
	modes = analyzer.get_mode_stats()['modes']
	chi2 = 10e10
	best = []
	print("Mode params")
	for mode in modes:
		chi2Local =  mode["local log-evidence"]*(-2)
		print(mode["mean"])
		if( chi2 > chi2Local):
			best = mode["mean"]
			chi2 = chi2Local

	print('creating marginal plot ...')
	data = analyzer.get_data()[:,2:]
	weights = analyzer.get_data()[:,0]
	
	#mask = weights.cumsum() > 1e-5
	mask = weights > 1e-4
	
	corner.corner(data[mask,:], weights=weights[mask], 
		labels=parameters_to_fit, show_titles=True)
	# plt.savefig('corner_plots/' + file_prefix + 'corner.pdf')
	plt.savefig('corner_plots/' + file_prefix + 'corner.png')
	plt.close()

	return best, chi2, modes

def fit_parallax(name, parStart, datasets, zeroBlendingDict, coords):
	# Starting parameters:
	params = dict()
	params['t_0'] = parStart[0]
	params['t_0_par'] = int(parStart[0])
	params['u_0'] = parStart[1]
	params['t_E'] = parStart[2]
	params['pi_E_N'] = parStart[3]
	params['pi_E_E'] = parStart[4]
	t_0_par = int(parStart[0])
	my_model = mm.Model(params, coords=coords)
	my_event = mm.Event(datasets=datasets, model=my_model, fix_blend_flux=zeroBlendingDict)
	
	# print("Coords check:", my_model.coords)
	
	# Which parameters we want to fit?
	parameters_to_fit = ["t_0", "u_0", "t_E", "pi_E_N", "pi_E_E"]
	# Min and max values
	min_values = np.array([2454000., -1.5, 0., -1.0, -1.0])
	max_values = np.array([2460000., 1.5, 1000., 1.0, 1.0])
	
	# Setting up minimizer
	minimizer = Minimizer(my_event, parameters_to_fit)
	minimizer.set_cube(min_values, max_values)
	minimizer.set_chi2_0()
	
	dir_out = "chains/"
	if not os.path.exists(dir_out):
	    os.mkdir(dir_out)
	file_prefix = "%s_psplpie"%(name)
	
	# we will not save the results to a file
	start_time = time.time()
	
	# Run MultiNest:
	run_kwargs = {
	'LogLikelihood': minimizer.ln_likelihood,
	'Prior': minimizer.transform_cube,
	'n_dims': len(parameters_to_fit),
	'resume': False,
	'outputfiles_basename': os.path.join(dir_out, file_prefix+"_"),
	'importance_nested_sampling': False,
	'multimodal': True,
	'n_live_points': 450}
	result = solve(**run_kwargs)
	
	# Analyze results - we print each mode separately and give log-evidence:
	analyzer = Analyzer(n_params=run_kwargs['n_dims'], outputfiles_basename=run_kwargs['outputfiles_basename'])
					
	log = "multinest_log.dat"
	logOutput = open(log, "a")
	logOutput.write("%s : PSPL+piE, sourceid: %s "%(datetime.now(), str(name)))
	print("--- Multinest took %s seconds ---" % (time.time() - start_time))
	logOutput.write("--- Multinest took %s seconds ---\n" % (time.time() - start_time))
	logOutput.close()
	
	modes = analyzer.get_mode_stats()['modes']
	chi2 = 10e10
	best = []
	# print("Mode params")
	for mode in modes:
		chi2Local =  mode["local log-evidence"]*(-2)
		#print(mode["mean"])
		if( chi2 > chi2Local):
			best = mode["mean"]
			chi2 = chi2Local
			
	print('creating marginal plot ...')
	data = analyzer.get_data()[:,2:]
	weights = analyzer.get_data()[:,0]
	
	#mask = weights.cumsum() > 1e-5
	mask = weights > 1e-4
	
	corner.corner(data[mask,:], weights=weights[mask], 
		labels=parameters_to_fit, show_titles=True)
	# plt.savefig('corner_plots/' + file_prefix + 'corner.pdf')
	plt.savefig('corner_plots/' + file_prefix + 'corner.png')
	plt.close()

	return best, chi2, modes

def image_data_lightcurve(name, datasets, tmin, tmax, magmin, magmax, params):
	my_model = mm.Model({'t_0': params[0], 'u_0': params[1], 't_E': params[2]})
	my_event = mm.Event(datasets=datasets, model=my_model) #, fix_blend_flux={data: 0.0})
	
	fig = plt.figure()
	my_event.plot_data(subtract_2450000=True)
	plt.xlim(tmin-2450000., tmax-2450000.)
	plt.ylim(magmin, magmax)
	plt.legend(loc='best')
	plt.title(name)
	plt.grid(True)
	
	img = io.BytesIO()
	fig.savefig(img, format='png',
				bbox_inches='tight')
	img.seek(0)
	encoded = base64.b64encode(img.getvalue())
	html_code = '<img src="data:image/png;base64, {}">'.format(encoded.decode('utf-8'))
	img.close()
	plt.close(fig)
	return  html_code

def image_model_lightcurve(name, datasets, tmin, tmax, magmin, magmax, parPspl, parPar, psplModes, parModes, t0par, coords):
	# Now let's plot 2 models
	model_0 = mm.Model({'t_0': parPspl[0], 'u_0': parPspl[1], 't_E': parPspl[2]})
	model_1 = mm.Model(
	{'t_0': parPar[0], 'u_0': parPar[1], 't_E': parPar[2],
	 'pi_E_N': parPar[3], 'pi_E_E': parPar[4], 't_0_par': t0par},
	coords=coords)
	
	event_pspl = mm.Event(model=model_0, datasets=datasets)#, fix_blend_flux={data: 0.0})
	event_par = mm.Event(model=model_1, datasets=datasets)#, fix_blend_flux={data: 0.0})
	    
	fig = plt.figure()
	grid = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
	axes = plt.subplot(grid[0])
	plot_params = {'lw': 2.5, 'alpha': 1.0, 'subtract_2450000': True,
	               't_start': tmin, 't_stop': tmax}
	
	event_pspl.plot_data(subtract_2450000=True)
	event_pspl.plot_model(label='no pi_E', color='black', ls='--', zorder=8, **plot_params)
	event_par.plot_model(label='w/ pi_E', color='red', ls='-', zorder=9, **plot_params)
	
	plot_params = {'lw': 2.5, 'alpha': 0.3, 'subtract_2450000': True,
	               't_start': tmin, 't_stop': tmax}
	
	# for mode in psplModes:
		# modePar = mode["mean"]
		# model = mm.Model(
		# {'t_0': modePar[0], 'u_0': modePar[1], 't_E': modePar[2]})
		# event = mm.Event(model=model, datasets=datasets)#, fix_blend_flux={data: 0.0})
		# event.plot_model(color='gray', **plot_params)
	
	# for mode in parModes:
		# modePar = mode["mean"]
		# model = mm.Model(
		# {'t_0': modePar[0], 'u_0': modePar[1], 't_E': modePar[2], 
		# 'pi_E_N': modePar[3], 'pi_E_E': modePar[4], 't_0_par': t0par},
		# coords=coords)
		# event = mm.Event(model=model, datasets=datasets)#, fix_blend_flux={data: 0.0})
		# event.plot_model(color='coral', **plot_params)
	
	plt.xlim(tmin-2450000., tmax-2450000.)
	plt.ylim(magmin, magmax)
	plt.grid(True)
	plt.legend(loc='best')
	plt.title(name)
	
	axes = plt.subplot(grid[1])
	event_par.plot_residuals(subtract_2450000=True)
	plotTimes = np.linspace(tmin,tmax,10000)
	
	data_ref = event_par.data_ref
	(f_source_0, f_blend_0) = event_par.get_flux_for_dataset(data_ref)
	I0Par = mm.utils.Utils.get_mag_from_flux(f_source_0+f_blend_0)
	fsPar = f_source_0/(f_source_0+f_blend_0)
	magnification = model_1.get_magnification(plotTimes)
	parModelFlux = f_source_0 * magnification + f_blend_0
	parModelMag = mm.Utils.get_mag_from_flux(parModelFlux)
	
	data_ref = event_pspl.data_ref
	(f_source_0, f_blend_0) = event_pspl.get_flux_for_dataset(data_ref)
	I0PSPL = mm.utils.Utils.get_mag_from_flux(f_source_0+f_blend_0)
	fsPSPL = f_source_0/(f_source_0+f_blend_0)
	magnification = model_0.get_magnification(plotTimes)
	psplModelFlux = f_source_0 * magnification + f_blend_0
	psplModelMag = mm.Utils.get_mag_from_flux(psplModelFlux)
	
	plt.plot(plotTimes - 2450000., psplModelMag - parModelMag, color='black', ls='--', zorder=9)
	
	# for mode in psplModes:
		# modePar = mode["mean"]
		# model = mm.Model(
		# {'t_0': modePar[0], 'u_0': modePar[1], 't_E': modePar[2]})
		# event = mm.Event(model=model, datasets=datasets)#, fix_blend_flux={data: 0.0})
		# data_ref = event.data_ref
		# (f_source_0, f_blend_0) = event.get_flux_for_dataset(data_ref)
		# magnification = model.get_magnification(plotTimes)
		# ModelFlux = f_source_0 * magnification + f_blend_0
		# ModelMag = mm.Utils.get_mag_from_flux(ModelFlux)
	
		# plt.plot(plotTimes - 2450000., ModelMag - parModelMag, color='gray', ls='-', lw=2.5, alpha=0.3)
			
	# for mode in parModes:
		# modePar = mode["mean"]
		# model = mm.Model(
		# {'t_0': modePar[0], 'u_0': modePar[1], 't_E': modePar[2], 
		# 'pi_E_N': modePar[3], 'pi_E_E': modePar[4], 't_0_par': t0par},
		# coords=coords)
		# event = mm.Event(model=model, datasets=datasets)#, fix_blend_flux={data: 0.0})
		# data_ref = event.data_ref
		# (f_source_0, f_blend_0) = event.get_flux_for_dataset(data_ref)
		# magnification = model.get_magnification(plotTimes)
		# ModelFlux = f_source_0 * magnification + f_blend_0
		# ModelMag = mm.Utils.get_mag_from_flux(ModelFlux)
		
		# plt.plot(plotTimes - 2450000., ModelMag - parModelMag, color='coral', ls='-', lw=2.5, alpha=0.3)
	
	plt.xlim(tmin-2450000., tmax-2450000.)
	plt.axhline(y=0, ls='-', color='red')
	plt.grid()
		
	img = io.BytesIO()
	fig.savefig(img, format='png',
				bbox_inches='tight')
	img.seek(0)
	encoded = base64.b64encode(img.getvalue())
	html_code = '<img src="data:image/png;base64, {}">'.format(encoded.decode('utf-8'))
	img.close()
	plt.close(fig)
	
	
	return html_code, I0PSPL, fsPSPL, I0Par, fsPar
	
def image_prediction_lightcurve(name, datasets, tmin, tmax, magmin, magmax, parPspl, parPar, psplModes, parModes, t0par, coords):
	# Now let's plot 2 models
	model_0 = mm.Model({'t_0': parPspl[0], 'u_0': parPspl[1], 't_E': parPspl[2]})
	model_1 = mm.Model(
	{'t_0': parPar[0], 'u_0': parPar[1], 't_E': parPar[2],
	 'pi_E_N': parPar[3], 'pi_E_E': parPar[4], 't_0_par': t0par},
	coords=coords)
	
	event_pspl = mm.Event(model=model_0, datasets=datasets[0])#, fix_blend_flux={data: 0.0})
	event_par = mm.Event(model=model_1, datasets=datasets[0])#, fix_blend_flux={data: 0.0})
	    
	fig = plt.figure()
	grid = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
	axes = plt.subplot(grid[0])
	plot_params = {'lw': 2.5, 'alpha': 1., 'subtract_2450000': True,
	               't_start': tmin, 't_stop': tmax}
	
	event_pspl.plot_data(subtract_2450000=True)
	event_pspl.plot_model(label='no pi_E', color='black', ls='--', **plot_params)
	event_par.plot_model(label='w/ pi_E', color='red', ls='-', **plot_params)
	
	plot_params = {'lw': 2.5, 'alpha': 0.5, 'subtract_2450000': True,
	               't_start': tmin, 't_stop': tmax}
	
	for mode in psplModes:
		modePar = mode["mean"]
		model = mm.Model(
		{'t_0': modePar[0], 'u_0': modePar[1], 't_E': modePar[2]})
		event = mm.Event(model=model, datasets=datasets)#, fix_blend_flux={data: 0.0})
		event.plot_model(color='gray', **plot_params)
	
	for mode in parModes:
		modePar = mode["mean"]
		model = mm.Model(
		{'t_0': modePar[0], 'u_0': modePar[1], 't_E': modePar[2], 
		'pi_E_N': modePar[3], 'pi_E_E': modePar[4], 't_0_par': t0par},
		coords=coords)
		event = mm.Event(model=model, datasets=datasets)#, fix_blend_flux={data: 0.0})
		event.plot_model(color='coral', **plot_params)
	
	
	plt.xlim(tmin-2450000., tmax-2450000.)
	plt.ylim(magmin, magmax)
	plt.grid(True)
	plt.legend(loc='best')
	plt.title(name)
		
	axes = plt.subplot(grid[1])
	event_par.plot_residuals(subtract_2450000=True)
	plotTimes = np.linspace(tmin,tmax,10000)
	
	data_ref = event_par.data_ref
	(f_source_0, f_blend_0) = event_par.get_flux_for_dataset(data_ref)
	magnification = model_1.get_magnification(plotTimes)
	parModelFlux = f_source_0 * magnification + f_blend_0
	parModelMag = mm.Utils.get_mag_from_flux(parModelFlux)
	
	data_ref = event_pspl.data_ref
	(f_source_0, f_blend_0) = event_pspl.get_flux_for_dataset(data_ref)
	magnification = model_0.get_magnification(plotTimes)
	psplModelFlux = f_source_0 * magnification + f_blend_0
	psplModelMag = mm.Utils.get_mag_from_flux(psplModelFlux)
	
	plt.plot(plotTimes - 2450000., psplModelMag-parModelMag, color='black', ls='--', zorder=9)
	
	for mode in psplModes:
		modePar = mode["mean"]
		model = mm.Model(
		{'t_0': modePar[0], 'u_0': modePar[1], 't_E': modePar[2]})
		event = mm.Event(model=model, datasets=datasets)#, fix_blend_flux={data: 0.0})
		data_ref = event.data_ref
		(f_source_0, f_blend_0) = event.get_flux_for_dataset(data_ref)
		magnification = model.get_magnification(plotTimes)
		ModelFlux = f_source_0 * magnification + f_blend_0
		ModelMag = mm.Utils.get_mag_from_flux(ModelFlux)
	
		plt.plot(plotTimes - 2450000., ModelMag - parModelMag, color='gray', ls='-', lw=2.5, alpha=0.3)
			
	for mode in parModes:
		modePar = mode["mean"]
		model = mm.Model(
		{'t_0': modePar[0], 'u_0': modePar[1], 't_E': modePar[2], 
		'pi_E_N': modePar[3], 'pi_E_E': modePar[4], 't_0_par': t0par},
		coords=coords)
		event = mm.Event(model=model, datasets=datasets)#, fix_blend_flux={data: 0.0})
		data_ref = event.data_ref
		(f_source_0, f_blend_0) = event.get_flux_for_dataset(data_ref)
		magnification = model.get_magnification(plotTimes)
		ModelFlux = f_source_0 * magnification + f_blend_0
		ModelMag = mm.Utils.get_mag_from_flux(ModelFlux)
		
		plt.plot(plotTimes - 2450000., ModelMag - parModelMag, color='coral', ls='-', lw=2.5, alpha=0.3)
	
	plt.xlim(tmin-2450000., tmax-2450000.)
	plt.axhline(y=0, ls='-', color='red')
	plt.grid()
	
	img = io.BytesIO()
	fig.savefig(img, format='png',
				bbox_inches='tight')
	img.seek(0)
	encoded = base64.b64encode(img.getvalue())
	html_code = '<img src="data:image/png;base64, {}">'.format(encoded.decode('utf-8'))
	img.close()
	plt.close(fig)
	
	
	return html_code
 

# Define Minimizer class used by MultiNest
class Minimizer(object):
    """
    A class used to store settings and functions that are used and
    called by MultiNest.
    Parameters :
        event: *MulensModel.event*
            Event for which chi^2 will be calculated.
        parameters_to_fit: *list*
            Names of parameters to be fitted, e.g.,  ["t_0", "u_0", "t_E"]
    """
    def __init__(self, event, parameters_to_fit):
        self.event = event
        self.parameters_to_fit = parameters_to_fit
        self._chi2_0 = None

    def set_chi2_0(self, chi2_0=None):
        """set reference value of chi2 (can help with numerical stability"""
        if chi2_0 is None:
            chi2_0 = np.sum([d.n_epochs for d in self.event.datasets])
        self._chi2_0 = chi2_0

    def set_cube(self, min_values, max_values):
        """
        remembers minimum and maximum values of model parameters so that later
        transform_cube() can be called
        """
        self._zero_points = min_values
        self._differences = max_values - min_values

    def transform_cube(self, cube):
        """transforms n-dimensional cube into space of physical quantities"""
        return self._zero_points + self._differences * cube

    def chi2(self, theta):
        """return chi^2 for a parameters theta"""
        for (i, param) in enumerate(self.parameters_to_fit):
            setattr(self.event.model.parameters, param, theta[i])
        chi2 = self.event.get_chi2()
        return chi2

    def ln_likelihood(self, theta):
        """logarithm of likelihood
        modified to be "safer,
        like in example16 in MulensModel tutorials by R. Poleski"""
        ln_like = -0.5 * (self.chi2(theta) - self._chi2_0)
        ln_max = -1.e300
        if not np.isfinite(ln_like) or ln_like < ln_max:
            if not np.isfinite(ln_like):
                msg = "problematic likelihood: {:}\nfor parameters: {:}"
                print(msg.format(ln_like, theta))
            ln_like = ln_max
            
        return	ln_like
