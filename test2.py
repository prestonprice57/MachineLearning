import json
import pprint

with open('data.json') as infile:
	d = infile.read()
	pp = pprint.PrettyPrinter(indent=4)
	print type(d)