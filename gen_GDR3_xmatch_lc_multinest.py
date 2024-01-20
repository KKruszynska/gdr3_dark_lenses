import numpy as np
import pandas as pd

from operator import itemgetter, attrgetter

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import urllib, base64
import io
import os

import mechanize
import requests

import json

import time

from astropy.time import Time
import astropy.coordinates as coord
import astropy.units as u

import MulensModel as mm

from module_getAuxData import *
from module_getUlensModels import *

from collections import OrderedDict

try:
    from pymultinest.solve import solve
    from pymultinest.analyse import Analyzer
except ImportError as err:
    print(err)
    print("\nPyMultiNest could not be imported.")
    print("Get it from: https://github.com/JohannesBuchner/PyMultiNest")
    print("and re-run the script")
    sys.exit(1)

def loadGaiaLc(name):
	fin = "/home/kasia/Documents/PhD/Gaia/cu7/SOS/MicrolensingAstro/scripts/lighcurves/G/%s_G.csv"%str(name)
	lcG = pd.read_csv(fin, header=0)
	fin = "/home/kasia/Documents/PhD/Gaia/cu7/SOS/MicrolensingAstro/scripts/lighcurves/BP/%s_BP.csv"%str(name)
	lcBP = pd.read_csv(fin, header=0)
	fin = "/home/kasia/Documents/PhD/Gaia/cu7/SOS/MicrolensingAstro/scripts/lighcurves/RP/%s_RP.csv"%str(name)
	lcRP = pd.read_csv(fin, header=0)
	return lcG, lcBP, lcRP

def modifyErrorbars(mags, errs):
	if(mags<13.5):
		errExp = np.sqrt(30.)*10**((0.17*13.5)-5.1)
	else:
		errExp = np.sqrt(30.)*10**((0.17*mags)-5.1)
	errMod = np.sqrt(errs**2 + errExp**2)
	return errMod

fin = "/home/kasia/Documents/PhD/Papers/DR3_dark_lens_candidates/DR3_sample_te50.csv"
dr3Data = pd.read_csv(fin, header=0)
gal = coord.SkyCoord(ra=dr3Data["ra_deg"]*u.degree, dec=dr3Data["dec_deg"]*u.degree)

fin = "/home/kasia/Documents/PhD/Gaia/cu7/ForPaper/GDR3_OGLEEWS_xmatch.csv"
ewsKnown = pd.read_csv(fin, header=0)
fin="/home/kasia/Documents/PhD/Gaia/cu7/ForPaper/GDR3_OGLEMroz_xmatch.csv"
ogleKnown = pd.read_csv(fin, header=0)
fin = "/home/kasia/Documents/PhD/Gaia/cu7/ForPaper/GDR3_MOA_xmatch.csv"
moaKnown = pd.read_csv(fin, header=0)
fin = "/home/kasia/Documents/PhD/Gaia/cu7/ForPaper/GDR3_KMTNet_xmatch.csv"
kmtnetKnown = pd.read_csv(fin, header=0)
fin = "/home/kasia/Documents/PhD/Gaia/cu7/ForPaper/GDR3_ASASSN_xmatch.csv"
asassnKnown = pd.read_csv(fin, header=0)
fin = "/home/kasia/Documents/PhD/Gaia/cu7/ForPaper/GDR3_GSA_xmatch.csv"
gsaKnown = pd.read_csv(fin, header=0)
	
part = -1
# for i in range(90, 100):
for i in range(50): #, len(dr3Data["sourceid"])):
	if((i%50)==0):
		print("Fitted %d/%d sources!"%(i, len(dr3Data["sourceid"])))
		part +=1	
		fout = "GDR3_aux_models_multinest_te50_part_%d.html"%(part)
		csv_outname = "GDR3_aux_models_multinest_te50_part_%d.csv"%(part)
		
		output = open(fout, "w")
		csvOutput = open(csv_outname, "w")
		
		output.write("<HTML><HEAD><TITLE>GDR3 candidate list x Auxiliary Data</TITLE></HEAD> \n <BODY>")
		#currentdate = datetime.utcnow()
		#output.write("%s UTC \n <br>\n"%currentdate)
		output.write("<center>Microlensing Candidates from Gaia Data Release 3 + other surveys")
		
		
		output.write('<table border="1">')
		output.write('<tr><th>#</th> <th>Names</th> <th>Gaia + foll LC</th>  <th>Models</th> <th>Predicted models</th> </tr>\n')
		
		csvOutput.write('#sourceid, RA, Dec, l, b, amplP2P, t0PSPL, u0PSPL, tEPSPL, G0PSPL, chi2PSPL, chidofPSPL, t0Par, u0Par, tEPar, piENPar, piEEPar, G0Par, chi2Par,  chidofPar,')
		for k in range(4):
			csvOutput.write('t0PSPL_%d, u0PSPL_%d, tEPSPL_%d, chi2PSPL_%d, chidofPSPL_%d'%(k,k,k,k,k))
			csvOutput.write(',')
		for k in range(4):
			csvOutput.write('t0Par_%d, u0Par_%d, tEPar_%d, piENpar_%d, piEEPar_%d, chi2Par_%d, chidofPar_%d'%(k,k,k,k,k,k,k))
			if(k<3):
				csvOutput.write(',')
			else:
				csvOutput.write('\n')
				
	# print("i", i)
	name = dr3Data["sourceid"].values[i]
	plotName = "GaiaDR3-ULENS-%03d"%(dr3Data["name"].values[i])
	
	ra, dec = dr3Data["ra_deg"].values[i], dr3Data["dec_deg"].values[i]
	coords = coord.SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
	# print("Coords check ra, dec:", ra, dec)
	
	gData, bpData, rpData = loadGaiaLc(name)
	gTime, gMags, gErrs1 = gData["times"], gData["mags"], gData["magerr"]
	gErrs = []
	for j in range(len(gErrs1)):
		mag = gMags[j]
		err = gErrs1[j]
		gErrs.append(modifyErrorbars(mag, err))
	gErrs = np.asarray(gErrs)
	amplpeak2peak = max(gMags) - min(gMags)
	
	mulens_datas = OrderedDict()
		
	mulens_datas['G_Gaia'] = mm.MulensData(
			data_list = (gTime, gMags, gErrs),
			phot_fmt = 'mag',
			add_2450000 = True,
			# ephemerides_file='/home/kasia/Documents/PhD/Gaia/cu7/ForPaper/Gaia_ephemeris.txt',
			plot_properties={'color': 'green', 'label': 'Gaia G ('+str(len(gTime))+')', 'zorder': 15, 'marker' : 'o'})
	lDatasets = len(gTime)
	parLen = 2
	# print(gTime)
	
	mulens_datas['BP_Gaia'] = mm.MulensData(
			data_list = (bpData["times"], bpData["mags"], 10.*bpData["magerr"]),
			phot_fmt = 'mag',
			add_2450000 = True,
			# ephemerides_file='/home/kasia/Documents/PhD/Gaia/cu7/ForPaper/Gaia_ephemeris.txt',
			plot_properties={'color': 'blue', 'label': 'Gaia BP ('+str(len(bpData["times"]))+')', 'zorder': 14, 'marker' : 'o'})
	lDatasets += len(bpData["times"])
	parLen += 2
	
	# print(bpData["times"])
	
	mulens_datas['RP_Gaia'] = mm.MulensData(
			data_list = (rpData["times"], rpData["mags"], 10.*rpData["magerr"]),
			phot_fmt = 'mag',
			add_2450000 = True,
			# ephemerides_file='/home/kasia/Documents/PhD/Gaia/cu7/ForPaper/Gaia_ephemeris.txt',
			plot_properties={'color': 'darkorange', 'label': 'Gaia RP ('+str(len(rpData["times"]))+')', 'zorder': 13, 'marker' : 'o'})
	# print(name)
	lDatasets += len(rpData["times"])
	parLen += 2


	string = "%s <br>sourceid: %d"%(plotName, name)
	othersName = ""

	# xmatch res
	resOGLE = ewsKnown[ewsKnown["#sourceid"] == name].index
	resOGLEMroz = ogleKnown[ogleKnown["#sourceid"] == name].index
	resMOA = moaKnown[moaKnown["#sourceid"] == name].index
	resKMTN = kmtnetKnown[kmtnetKnown["#sourceid"] == name].index
	resASASSN = asassnKnown[asassnKnown["sourceid"] == name].index
	resGSA = gsaKnown[gsaKnown["#sourceid"] == name].index
	
	if(len(resOGLE)>0 or len(resOGLEMroz)>0):
		if(len(resOGLEMroz)>0):
			ogleName = ogleKnown["name"].values[resOGLEMroz[0]]
			file_name = "/home/kasia/Documents/PhD/Thesis/Xmatch/ogle_data/"+ogleName+".dat"
			oTime, oMags, oErrs = np.genfromtxt(file_name, dtype=np.float, usecols=(0,1,2), unpack=True)
			mulens_datas['I_OGLE'] = mm.MulensData(
				data_list = (oTime, oMags, oErrs),
				phot_fmt = 'mag',
				add_2450000 = False,
				plot_properties={'color': 'cornflowerblue', 'label': 'OGLE EWS'+"("+str(len(oTime))+")", 'zorder': 12, 'marker' : '.'})
			lDatasets += len(oTime)
			parLen += 2
			string += "<br> in Mroz+ 2019 or Mroz+ 2020: %s\n"%(ogleName)
			othersName+= " "+ogleName
			# print(data.mag)
		elif(len(resOGLE)>0):
			ogleName = ewsKnown["name"].values[resOGLE[0]]
			#print(ogleName)
			oTime, oMags, oErrs = getOgleEwsLc(ogleName)
			#print(oTime[-1])
			mulens_datas['I_OGLEEWS'] = mm.MulensData(
				data_list = (oTime, oMags, oErrs),
				phot_fmt = 'mag',
				add_2450000 = False,
				plot_properties={'color': 'cornflowerblue', 'label': 'OGLE EWS'+"("+str(len(oTime))+")", 'zorder': 12, 'marker' : '.'})
			# datasets.append(data)
			# plotDatasets.append(data)
			lDatasets += len(oTime)
			parLen += 2
			text = ogleName.split('-')
			year = text[1]
			num = text[3]
			string += "<br><a href='http://ogle.astrouw.edu.pl/ogle4/ews/%s/blg-%s.html'>%s</a>\n"%(year, num, ogleName)
			othersName+= " "+ogleName
			dataToSave = np.array([oTime, oMags, oErrs])
			dataToSave = dataToSave.T
			np.savetxt("auxData/OGLE_EWS/%s_%s.dat"%(plotName, ogleName), dataToSave, delimiter="\t")
			# print(data.mag)
		if(len(resOGLE)>0 and len(resOGLEMroz)>0):
			ogleName = ewsKnown["name"].values[resOGLE[0]]
			text = ogleName.split('-')
			year = text[1]
			num = text[3]
			string += "<br><a href='http://ogle.astrouw.edu.pl/ogle4/ews/%s/blg-%s.html'>%s</a>\n"%(year, num, ogleName)
			othersName+= " "+ogleName
		
	if(len(resMOA)>0):
		moaName = moaKnown["name"].values[resMOA[0]]
		moaField = moaKnown["Field"].values[resMOA[0]]
		mTime, mMags, mErrs = getMoaLc(moaName, moaField)
		mulens_datas['MOA'] = mm.MulensData(
			data_list = (mTime, mMags, mErrs),
			phot_fmt = 'flux',
			add_2450000 = False,
			plot_properties={'color': 'darkseagreen', 'label': 'MOA'+"("+str(len(mTime))+")", 'zorder': 1, 'marker' : '.'})
		# datasets.append(data)
		# plotDatasets.append(data)
		lDatasets += len(mTime)
		parLen += 2
		text = moaName.split('-')
		year = text[1]
		string += "<br><a href='http://www.massey.ac.nz/~iabond/moa/alert%s/display.php?id=%s'>%s</a>\n"%(year, moaField, moaName)
		othersName+= " "+moaName
		dataToSave = np.array([mTime, mMags, mErrs])
		dataToSave = dataToSave.T
		np.savetxt("auxData/MOA/%s_%s.dat"%(plotName, moaName), dataToSave, delimiter="\t")
		# print(data.mag)
	
	if(len(resKMTN)>0):
		kmtnName = kmtnetKnown["name"].values[resKMTN[0]]
		kmtnField = kmtnetKnown["Field"].values[resKMTN[0]]
		kmtnStarNr = kmtnetKnown["StarID"].values[resKMTN[0]]
		obs = ["KMTA", "KMTC", "KMTS"]
		clist = ['indianred', 'plum', 'aquamarine']
		for site, color in zip(obs, clist):
			# print(site, color)
			kData = getKmtnetLc(kmtnName, kmtnField, kmtnStarNr, site)
			# print(kData)
			if(len(kData)>0):
				kTime, kMags, kErrs = kData[:, 0], kData[:, 1], kData[:, 2]
				mulens_datas[site] = mm.MulensData(
					data_list = (kTime, kMags, kErrs),
					add_2450000 = False,
					phot_fmt = 'flux',
					plot_properties={'color': color, 'label': 'KMTNet_'+site+"_I("+str(len(kTime))+")", 'zorder': 10, 'marker' : '.'})
				# if(plotName != "GaiaDR3-ULENS-240" and plotName != "GaiaDR3-ULENS-326"):
					# datasets.append(data)
				lDatasets += len(kTime)
				parLen += 2
				dataToSave = np.array([kTime, kMags, kErrs])
				dataToSave = dataToSave.T
				np.savetxt("auxData/KMTNet/%s_%s_%s.dat"%(plotName, kmtnName, site), dataToSave, delimiter="\t")
				# print(data.mag)
		text = kmtnName.split('-')
		year = text[1] 
		string += "<br><a href='https://kmtnet.kasi.re.kr/~ulens/event/%s/view.php?event=%s'>%s</a>\n"%(year, kmtnName, kmtnName)
		othersName+= " "+kmtnName
		
	if(len(resASASSN)>0):
		asassnName = asassnKnown["name"].values[resASASSN[0]]
		photName = asassnKnown["phot_file"].values[resASASSN[0]]
		band = ["g", "V"]
		colors = ['mediumpurple', 'mediumvioletred']
		for k in range(len(band)):
			fileName = "/home/kasia/Documents/PhD/Thesis/Xmatch/asassn_data/%s_%s.dat"%(photName, band[k])
			if os.path.isfile(fileName):
				aTime, aMags, aErrs = np.genfromtxt(fileName, dtype=np.float, usecols=(0,1,2), unpack=True)
				mulens_datas['ASASSN_'+band[k]] = mm.MulensData(
					data_list = (aTime, aMags, aErrs),
					phot_fmt = 'mag',
					add_2450000 = False,
					plot_properties={'color': colors[k], 'label': 'ASASSN '+band[k]+" ("+str(len(aTime))+")", 'zorder': 9, 'marker' : '.'})
				# if(plotName != "GaiaDR3-ULENS-023"):
					# datasets.append(data)
				# plotDatasets.append(data)
				lDatasets += len(aTime)
				parLen += 2
					# print(data.mag)
		string += "<br>%s</a>"%(asassnName)
		othersName+= " "+asassnName

	if(len(resGSA)>0):
		gsaName = gsaKnown["#Name"].values[resGSA[0]]
		gsaTime, gsaMags = getLightCurveGaia(gsaName)
		gsaErrs = getGaiaErrors(gsaMags)
			
		mulens_datas['G_GSA'] = mm.MulensData(
			data_list = (gsaTime, gsaMags, gsaErrs),
			phot_fmt = 'mag',
			add_2450000 = False,
			ephemerides_file='/home/kasia/Documents/PhD/Gaia/cu7/ForPaper/Gaia_ephemeris.txt',
			plot_properties={'color': 'darkred', 'label': 'Gaia (GSA)'+"("+str(len(gsaTime))+")", 'zorder': 11, 'marker' : 'o'})
		# datasets.append(data)
		# plotDatasets.append(data)
		lDatasets += len(gsaTime)
		parLen += 2
		string += "<br><a href='https://gsaweb.ast.cam.ac.uk/alerts/alert/%s/'>%s</a>"%(gsaName, gsaName)
		othersName+= " "+gsaName
		dataToSave = np.array([gsaTime, gsaMags, gsaErrs])
		dataToSave = dataToSave.T
		np.savetxt("auxData/GSA/%s_%s.dat"%(plotName, gsaName), dataToSave, delimiter="\t")
		# print(data.mag)
	
	# check for follow-up
	#filter_list = ['u','B','g','V','B2pg','r','R','R1pg','i','I','Ipg','z', 'B1pg', 'G', 'H', 'K', 'J']
	filter_list = ['u','B','g','B2pg','r','R','R1pg','i','I','Ipg','z', 'B1pg', 'G', 'H', 'K', 'J']
	# color_list = ['dodgerblue', 'skyblue', 'green', 'orange',  'slategrey', \
			 # 'darkred', 'salmon', 'lightcoral', 'dimgrey', 'peru', \
			 # 'fuchsia', 'hotpink', 'lightseagreen', 'navy', 'dimgrey', \
			 # 'tan', 'indigo', 'teal', 'crimson', 'blueviolet', \
			 # 'darkslategrey', 'palevioletred']
	color_list = ['dodgerblue', 'skyblue', 'green',  'slategrey', \
			 'darkred', 'salmon', 'lightcoral', 'dimgrey', 'peru', \
			 'fuchsia', 'hotpink', 'lightseagreen', 'navy', 'dimgrey', \
			 'tan', 'indigo', 'teal', 'crimson', 'blueviolet', \
			 'darkslategrey', 'palevioletred']
	datajson, status = get_followup(name)
	k = 0
	if (status != 0):	
		mjd0=np.array(datajson['mjd'])
		mag0=np.array(datajson['mag'])
		magerr0=np.array(datajson['magerr'])
		filter0=np.array(datajson['filter'])
		caliberr0=np.array(datajson['caliberr'])
		obs0 = np.array(datajson['observatory'])
		
		for band in filter_list:
			indexes = np.where(filter0 == band)
			#print(len(indexes))
			#print(indexes)
			if(len(mjd0[indexes])>0):
				ftime = mjd0[indexes] + 2400000.5
				fmags = mag0[indexes]
				ferr = magerr0[indexes]
				clr = color_list[k]
				indexes2 = np.where(ferr != -1.0)
				ftime = ftime[indexes2]
				fmags = fmags[indexes2]
				ferr = ferr[indexes2]
				#times = float(ftime)
				#mags = float(fmags)
				#errs = float(ferr)
				if(len(ftime)>1):
					mulens_datas['followup_'+band] = mm.MulensData(
						data_list = (ftime, fmags, ferr),
						phot_fmt = 'mag',
						plot_properties={'color': clr, 'label': 'followup_'+band+"("+str(len(ftime))+")", 'marker': '.'})
					# datasets.append(data)
					# plotDatasets.append(data)
					lDatasets += len(ftime)
					parLen += 2
					othersName+= " "+"followup_"+band
					dataToSave = np.array([ftime, fmags, ferr])
					dataToSave = dataToSave.T
					np.savetxt("auxData/followup/%s_followup_%s.dat"%(plotName, band), dataToSave, delimiter="\t")
					# print(data.mag)
			k += 1
		
		# ztfDrData, ztfBand = getZTFdrLc(gsaRA[i], gsaDec[i])
		# color = ['mediumturquoise', 'tomato', 'deeppink']
		# if(len(ztfDrData)>0):
			# b = 0
			# for zdata in ztfDrData: 
				# band = ztfBand[b]
				# #print(zdata[:,0], zdata[:,1], zdata[:,2])
				# mulens_datas['ZTF_'+band] = mm.MulensData(
				# data_list = (zdata[:,0], zdata[:,1], zdata[:,2]),
				# add_2450000 = False,
				# phot_fmt = 'mag',
				# plot_properties={'color': color[b], 'label': 'ZTF_'+band+" ("+str(len(zdata[:,0]))+")", 'marker' : '.'})
				# plotDatasets.append(data)
				# b += 1
				# dataToSave = np.array([zdata[:,0], zdata[:,1], zdata[:,2]])
				# dataToSave = dataToSave.T
				# np.savetxt("auxData/ZTF/%s_ZTF_%s.dat"%(plotName, band), dataToSave, delimiter="\t")
				# print(data.mag)
			# string += "<br>in ZTF DR"
	
	# for d in datasets:
		# print(d.time)
	
	output.write("<tr><td><b>%d</b></td>\n"%(i+1))
	# print("i", i)
	output.write("<td>%s</td>"%string)
	print(string)
	
	zeroBlendingDict= dict()
	zeroBlendingDict = { i : 0 for i in tuple(mulens_datas.values()) }
	
	# for d in datasets:
		# print(d.time)

	t0 = gTime[np.argmin(np.array(gMags))]+2450000.
	psplStart = [t0, 0.3, 100.]
	parStart = [t0, 0.3, 100., 0., 0.]

	psplParams, psplChi2, psplModes =  fit_pspl(plotName, psplStart, (tuple(mulens_datas.values())), zeroBlendingDict)
	parParams, parChi2, parModes = fit_parallax(plotName, parStart, (tuple(mulens_datas.values())), zeroBlendingDict, coords)

	parameters_to_fit = ["t_0", "u_0", "t_E"]
	my_model = mm.Model({'t_0': psplParams[0], 'u_0': parParams[1], 't_E': psplParams[2]})
	my_event = mm.Event(datasets=(tuple(mulens_datas.values())), model=my_model, fix_blend_flux=zeroBlendingDict)
	minimizer1 = Minimizer(my_event, parameters_to_fit)

	psplChi2 = minimizer1.chi2(psplParams)

	parameters_to_fit = ["t_0", "u_0", "t_E", "pi_E_N", "pi_E_E"]
	my_model = mm.Model({'t_0': parParams[0], 'u_0': parParams[1], 't_E': parParams[2], 'pi_E_N': parParams[3], 'pi_E_E': parParams[4], 't_0_par': int(t0)}, coords=coords)
	my_event = mm.Event(datasets=(tuple(mulens_datas.values())), model=my_model, fix_blend_flux=zeroBlendingDict)
	minimizer2 = Minimizer(my_event, parameters_to_fit)

	parChi2 = minimizer2.chi2(parParams)
	
	psplChiDof = psplChi2/(lDatasets-(3+parLen))
	parChiDof = parChi2/(lDatasets-(5+parLen))
	
	#print(psplParams)
	#print(parParams)
	
	tmin = gTime.values[0]-150.+2450000.
	tmax = gTime.values[-1]+150.+2450000.
	magmin = max(gMags)+0.25
	magmax = min(gMags)-0.25
	
	allImgUrl = image_data_lightcurve(plotName, (tuple(mulens_datas.values())), tmin, tmax, magmin, magmax, psplParams)
	
	folImgUrl, I0PSPL, fsPSPL, I0Par, fsPar = image_model_lightcurve(plotName, (tuple(mulens_datas.values())), tmin, tmax, magmin, magmax, psplParams, parParams, psplModes, parModes, int(t0), coords)
	
	predImgUrl = image_prediction_lightcurve(plotName, (tuple(mulens_datas.values())), tmin, tmax, magmin, magmax, psplParams, parParams, psplModes, parModes, int(t0), coords)
	
	output.write("<td>")
	output.write(allImgUrl)
	output.write('<br>Coordinates, \
				<br>l, b= %f, %f,\
				<br>RA, dec= %f, %f'%(gal.galactic.l.degree[i], gal.galactic.b.degree[i], ra, dec))
	# output.write('<br><a href="http://photometry-classification.herokuapp.com/classification_results/%f%%26%f">Gezer et al. PhotClass</a>'%(gsaRA[i], gsaDec[i]))
	output.write("</td>")
		
	output.write("<td>")
	output.write(folImgUrl)
	output.write('<br>amplPeak2Peak=%f, \
				<br> PSPL:\
				<br>t0=%.2f, u0=%.3f, tE=%.2f, I0=%.2f, fs=%.3f, chi2dof=%.2f, \
				<br> Par: \
				<br>t0=%.2f, u0=%.3f, tE=%.2f, piEN=%.3f, piEE=%.3f, I0=%.2f, fs=%.3f, chi2dof=%.2f\n'%(amplpeak2peak, \
				psplParams[0], psplParams[1], psplParams[2], I0PSPL, fsPSPL, psplChiDof,\
				parParams[0], parParams[1], parParams[2], parParams[3], parParams[4], I0Par, fsPar, parChiDof))
	output.write("</td>")
	#parameters_to_fit = ["t_0", "u_0", "t_E", "pi_E_N", "pi_E_E"]
	csvOutput.write('%s, %f, %f, %f, %f,'\
				'%f,' \
				'%f, %f, %f, %f, %f, %f,' \
				'%f, %f, %f, %f, %f, %f, %f, %f,'%(str(name), dr3Data["ra_deg"].values[i], dr3Data["dec_deg"].values[i], gal.galactic.l.degree[i], gal.galactic.b.degree[i], \
				amplpeak2peak, \
				psplParams[0], psplParams[1], psplParams[2], I0PSPL, fsPSPL, psplChiDof,\
				parParams[0], parParams[1], parParams[2], parParams[3], parParams[4], I0Par, fsPar, parChiDof))
	output.write("<td>")
	output.write(predImgUrl)
	string = "<br>PSPL predicted models: "
	for mode in psplModes:
		#print(mode)
		params = [mode["mean"][0], mode["mean"][1], mode["mean"][2]]
		modeChi2 = minimizer1.chi2(params)
		string += "<br>t0=%.2f, u0=%.3f, tE=%.2f, chi2= %.2f, chi2dof=%.2f"%(mode["mean"][0], mode["mean"][1], mode["mean"][2], modeChi2, modeChi2/(lDatasets-(3+parLen)))
	string += "<br> PSPL + piE predicted models:"
	for mode in parModes:
		params = [mode["mean"][0], mode["mean"][1], mode["mean"][2], mode["mean"][3], mode["mean"][4]]
		modeChi2 = minimizer2.chi2(params)
		string += "<br>t0=%.2f, u0=%.3f, tE=%.2f, piEN=%.3f, piEE=%.3f,  chi2= %.2f, chi2dof=%.2f"%(mode["mean"][0], mode["mean"][1], mode["mean"][2], mode["mean"][3], mode["mean"][4], modeChi2, modeChi2/(lDatasets-(5+parLen)))
	output.write(string)
	output.write("</td>")
	
	psplModesToSort = []
	psplModeLen = 0
	for mode in psplModes:
		psplModeLen += 1
		params = [mode["mean"][0], mode["mean"][1], mode["mean"][2]]
		modeChi2 = minimizer1.chi2(params)
		psplModesToSort.append((mode["mean"][0], mode["mean"][1], mode["mean"][2], modeChi2, modeChi2/(lDatasets-(3+parLen))))
		
	parModesToSort = []
	parModeLen = 0
	for mode in parModes:
		parModeLen += 1
		params = [mode["mean"][0], mode["mean"][1], mode["mean"][2], mode["mean"][3], mode["mean"][4]]
		modeChi2 = minimizer2.chi2(params)
		parModesToSort.append((mode["mean"][0], mode["mean"][1], mode["mean"][2], mode["mean"][3], mode["mean"][4],modeChi2, modeChi2/(lDatasets-(5+parLen))))
		
	# print(len(psplModesToSort), len(parModesToSort))
	# print(psplModesToSort)
	# print(parModesToSort)
	if(psplModeLen == 1):
		psplModesSorted = psplModesToSort
	else:
		psplModesSorted = sorted(psplModesToSort, key = lambda mode: mode[3])
	# print(psplModesSorted)
		
	if(parModeLen == 1):
		parModesSorted = parModesToSort
	else:
		parModesSorted = sorted(parModesToSort, key = lambda mode: mode[5])
	# print(parModesSorted)
		
	if(psplModeLen<4):
		if(psplModeLen == 1):
			csvOutput.write('%.2f, %.3f, %.2f, %.2f, %.2f'%(psplModesSorted[0][0], psplModesSorted[0][1], psplModesSorted[0][2], psplModesSorted[0][3], psplModesSorted[0][4]))
			csvOutput.write(',')
		else:
			for j in range(psplModeLen):
				csvOutput.write('%.2f, %.3f, %.2f, %.2f, %.2f'%(psplModesSorted[j][0], psplModesSorted[j][1], psplModesSorted[j][2], psplModesSorted[j][3], psplModesSorted[j][4]))
				csvOutput.write(',')
		for j in range(psplModeLen, 4):
			csvOutput.write('%.2f, %.3f, %.2f, %.2f, %.2f'%(0., 0., 0., 0., 0.))
			csvOutput.write(',')
	else:
		for j in range(4):
			csvOutput.write('%.2f, %.3f, %.2f, %.2f, %.2f'%(psplModesSorted[j][0], psplModesSorted[j][1], psplModesSorted[j][2], psplModesSorted[j][3], psplModesSorted[j][4]))
			csvOutput.write(',')
			
	if(parModeLen<4):
		if(parModeLen == 1):
			csvOutput.write('%.2f, %.3f, %.2f, %.3f, %.3f, %.2f, %.2f'%(parModesSorted[0][0], parModesSorted[0][1], parModesSorted[0][2], parModesSorted[0][3], parModesSorted[0][4], parModesSorted[0][5], parModesSorted[0][6]))
			csvOutput.write(',')
		else:
			for j in range(parModeLen):
				csvOutput.write('%.2f, %.3f, %.2f, %.3f, %.3f, %.2f, %.2f'%(parModesSorted[j][0], parModesSorted[j][1], parModesSorted[j][2], parModesSorted[j][3], parModesSorted[j][4], parModesSorted[j][5], parModesSorted[j][6]))
				csvOutput.write(',')
		for j in range(parModeLen, 4):
			csvOutput.write('%.2f, %.3f, %.2f, %.3f, %.3f, %.2f, %.2f'%(0., 0., 0., 0., 0., 0., 0.))
			if(j<3):
				csvOutput.write(',')
			else:
				csvOutput.write('\n')
	else:
		for j in range(4):
			csvOutput.write('%.2f, %.3f, %.2f, %.3f, %.3f, %.2f, %.2f'%(parModesSorted[j][0], parModesSorted[j][1], parModesSorted[j][2], parModesSorted[j][3], parModesSorted[j][4], parModesSorted[j][5], parModesSorted[j][6]))
			if(j<3):
				csvOutput.write(',')
			else:
				csvOutput.write('\n')
				
	if((i%50)==49):
		output.write("</table>")
		output.write('FINISHED<br><br>')
		output.write("</BODY>")
		output.close()
		csvOutput.close()
	
output.write("</table>")
output.write('FINISHED<br><br>')
output.write("</BODY>")
output.close()
csvOutput.close()
