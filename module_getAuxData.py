import numpy as np
import pandas as pd

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

import MulensModel as mm

def getZTFdrLc(ra, dec):
	rad = 0.5/3600.
	bands = ['g', 'r', 'i']
			
	datasets = []
	b = []
	try:
		for band in bands:
			data = []
			url = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?POS=CIRCLE+%f+%f+%f&BANDNAME=%s&NOBS_MIN=3&BAD_CATFLAGS_MASK=32768&FORMAT=ipac_table"%(ra, dec, rad, band)
			print(url)
			req = requests.get(url)
			text = req.text
			lines = text.split('\n')
		
			i = 0
			for line in lines:
				col = line.split()
				#print(len(col), col)
				if(len(col)>1):
					if(col[0][0] != "|" and col[0][0] != "\\" and col[0][0] != "<"):
						#print(i, col[2], col[4], col[5], col[7])
						data.append((float(col[2]), float(col[4]), float(col[5])))
				i += 1
			data = np.asarray(data)
			if(len(data)>0):
				datasets.append(data)
				b.append(band)
			
			
	except requests.HTTPError as exception:
		print(exception)

	return datasets, b


def getOgleEwsLc(name):
	#print(name)
	text = name.split('-')
	#print(text)
	year = text[1]
	num = text[3]
	
	url = ("http://www.astrouw.edu.pl/ogle/ogle4/ews/%s/blg-%s/phot.dat")%(year, num)
	
	req = requests.get(url).content
	ogleLc = pd.read_csv(io.StringIO(req.decode('utf-8')), delimiter=" ", header=None)
	#print(ogleLc)
	data = ogleLc.to_numpy()
	return data[:,0], data[:,1], data[:,2]
	
def getMoaLc(name, field):
	text = name.split('-')
	# print(text)
	year = text[1]
	
	if(int(year)<=2015):
		url = "http://www.massey.ac.nz/~iabond/moa/alerts/view_txt.php?url=http://it047333.massey.ac.nz/moa/ephot/phot-%s.dat"%(field)
	else:
		url = ("https://www.massey.ac.nz/~iabond/moa/alert%s/fetchtxt.php?path=moa/ephot/phot-%s.dat")%(year, field)
	# print(url)
	times = []
	mags = []
	errs = []
	
	try:
		req = requests.get(url)
		text = req.text
		lines = text.split('\n')
		#print(len(lines))
		
		i = 0
		if(len(lines)>10):
			for l in lines:
				#print(i)
				#print(l)
				if(len(l)>0):
					col = l.split()
					if(i>10 and i<len(lines)-2 and float(col[1])>0.):
						#print(l)
						m, er = mm.utils.Utils.get_mag_and_err_from_flux(float(col[1]), float(col[2]))
						if(not np.isnan(m)):
						#print(i, col)
						#print(col[0], col[3], col[4])
							times.append(float(col[0]))
							mags.append(float(col[1]))
							errs.append(float(col[2]))
				i += 1
	except requests.HTTPError as exception:
		print(exception)
	
	
	return np.array(times), np.array(mags), np.array(errs)
	
def getKmtnetLc(name, field, starNr, obsName):
	# print(name)
	text = name.split('-')
	year = text[1]
	num = text[3]
	fNum = field[3:5]
	# print(field, fNum)
	
	eventNr = "KB"+year[-2:]+num
	data = []

	# moved on to use DIA phot, since its reccomended by KMTNet people -> Kim et al. 2016
	# url: https://kmtnet.kasi.re.kr/~ulens/event/2019/data/KB191015/diapl/I04_I.diapl
	# url: https://kmtnet.kasi.re.kr/~ulens/event/2019/data/KB191015/diapl/KMTA04_I.diapl
	# url: https://kmtnet.kasi.re.kr/~ulens/event/2015/fig/ctio/dia.BLG01K0126.0793.txt
	# url: https://kmtnet.kasi.re.kr/~ulens/event/2015/fig/saao/dia.BLG01K0126.0793.txt
	# url: https://kmtnet.kasi.re.kr/~ulens/event/2015/fig/sso/dia.BLG01K0126.0793.txt
	if(year == "2015"):
		site = ""
		if(obsName == "KMTA"):
			site = "saao"
		elif(obsName == "KMTC"):
			site = "ctio"
		elif(obsName == "KMTS"):
			site = "sso"
		url = "https://kmtnet.kasi.re.kr/~ulens/event/%s/fig/%s/dia.%s.%04d.txt"%(year, site, field, int(starNr))
	else:
		url = "https://kmtnet.kasi.re.kr/~ulens/event/%s/data/%s/diapl/%s%s_I.diapl"%(year, eventNr, obsName, fNum)
	print(url)
	try:
		req = requests.get(url)
		text = req.text
		lines = text.split('\n')
		#count = 0
		#print(len(lines))
		if(len(lines)>1):
			for l in lines:
				#count += 1
				if(l[0:1]!="<"):
					if(len(l)>0):
						# print(l)
						col = l.split()
						if(col[0] != "#" and abs(float(col[1])-0.)>1e-8 and float(col[4])>0. and abs(-99.0 - float(col[1]))>1e-8):
							m, er = mm.utils.Utils.get_mag_and_err_from_flux(float(col[1]), float(col[2]))
							if(not np.isnan(m)):
							#print(col[0], col[3], col[4])
								data.append((float(col[0])+2450000., float(col[1]), float(col[2])))
								#data.append((float(col[0])+2450000., m, er))
		#print(len(lines))
	except requests.HTTPError as exception:
		print(exception)
	data = np.asarray(data)
	#print(data)
			
	return data

def getLightCurveGaia(name):
	url = ("http://gsaweb.ast.cam.ac.uk/alerts/alert/%s/lightcurve.csv")%(name)
	req = requests.get(url)
	text = req.text
	lines = text.split('\n')
	times = []
	mags = []
	
	for l in lines:
		col = l.split(',')
		#print(col)
		if (len(col)>1):
			if (len(col)==3 and (col[1] != 'JD(TCB)')):
				if (col[2] != 'null' and  col[2] != 'untrusted'):
					times.append(float(col[1]))
					mags.append(float(col[2]))
	return times, mags
	
def getGaiaErrors(mag):
	a1 = 0.2
	b1 = -5.2
	a2 = 0.26
	b2 = -6.26
	
	err = []
	for i in range(0,len(mag)):
		if mag[i] <= 13.5:
			 err_corr=a1*13.5+b1
		elif mag[i] > 13.5 and mag[i] <= 17.:
			err_corr=a1*mag[i]+b1
		elif mag[i] > 17.:
			err_corr=a2*mag[i]+b2	
		err.append(10.**err_corr)
	return err

# using ZKR's code getfollowup.py as a base here
def get_followup(name):
	followup=1
	data1 = 0.
	try:
		# print("Opening followup page")
		br = mechanize.Browser()
		followuppage=br.open('http://gsaweb.ast.cam.ac.uk/followup/')
		req=br.click_link(text='Login')
		br.open(req)
		br.select_form(nr=0)
		br.form['hashtag']='ZKR_ceb5e70b2c4e9b8866d7d62b9c60f811'
		br.submit()
		#print "Logged in!"
		#print "Requesting followup!"
		r=br.open('http://gsaweb.ast.cam.ac.uk/followup/get_alert_lc_data?alert_name=ivo:%%2F%%2F%s'%name)
		page=br.open('http://gsaweb.ast.cam.ac.uk/followup/get_alert_lc_data?alert_name=ivo:%%2F%%2F%s'%name)
		pagetext=page.read()
		data1=json.loads(pagetext)
		#print "Followup downloaded. Proceeding further"
		filter_list = ['u','B','g','B2pg','r','R','R1pg','i','I','Ipg','z', 'B1pg', 'G', 'H', 'K', 'J']
		#filter_list = ['u','B','g','V','B2pg','r','R','R1pg','i','I','Ipg','z', 'B1pg', 'G', 'H', 'K', 'J']
		#if len(set(data1["filter"]) & set(['u','B','g','V','B2pg','r','R','R1pg','i','I','Ipg','z', 'B1pg', 'G', 'H', 'K', 'J']))>0:
		if len(set(data1["filter"]) & set(['u','B','g','B2pg','r','R','R1pg','i','I','Ipg','z', 'B1pg', 'G', 'H', 'K', 'J']))>0:
			fup=[data1["mjd"],data1["mag"],data1["magerr"],data1["filter"],data1["observatory"]] 
			# print("Followup data downloaded!")
		else:
			followup = 0.
			# print("No followup available.")
	except mechanize.HTTPError as e:
		followup = 0.
		print(e)
	return data1, followup	
