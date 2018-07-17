# Load CSV (using python)
import csv,math,copy
import numpy as np
import mmit
from mmit import MaxMarginIntervalTree
from mmit.pruning import min_cost_complexity_pruning
from mmit.model_selection import GridSearchCV
from mmit.metrics import mean_squared_error

o = []
filename = ['mmit_data/auto93/features.csv', 'mmit_data/auto93/targets.csv', 
			'mmit_data/autohorse/features.csv', 'mmit_data/autohorse/targets.csv',
			'mmit_data/autompg/features.csv', 'mmit_data/autompg/targets.csv',
			'mmit_data/autoprice/features.csv', 'mmit_data/autoprice/targets.csv',
			'mmit_data/baskball/features.csv', 'mmit_data/baskball/targets.csv',
			'mmit_data/bodyfat/features.csv', 'mmit_data/bodyfat/targets.csv',
			'mmit_data/cloud/features.csv', 'mmit_data/cloud/targets.csv',
			'mmit_data/cpu/features.csv', 'mmit_data/cpu/targets.csv',
			'mmit_data/meta/features.csv', 'mmit_data/meta/targets.csv' ,
			'mmit_data/sleep/features.csv', 'mmit_data/sleep/targets.csv']


for i in xrange(20):
	raw_data = open(filename[i], 'rt')
	reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
	x = list(reader)
	o.append(np.array(x[1:]).astype('float'))

for i in xrange(0,20,2):
	x = o[i]
	y = o[i + 1]

	trainx = x[:4*len(x)/5,]
	trainy = y[:4*len(y)/5,]
	testx = x[len(x)/2:,]
	testy = y[len(y)/2:,]

	estimator = MaxMarginIntervalTree(margin=1.0, max_depth=4)
	clf = estimator.fit(trainx, trainy)
	fit = estimator.predict(testx)

	print mean_squared_error(testy, fit)
	estimator = None
	"""
	alphas, pruned_trees = min_cost_complexity_pruning(estimator)
	print alphas

	for pt in pruned_trees:
	    print sorted(pt.tree_.rules)
	 

	param_grid =  {"margin": [0.0, 2.0], "loss":["linear_hinge"], "max_depth":[np.infty], "min_samples_split":[0]}
	search = GridSearchCV(estimator, param_grid)
	search.fit(x,y)

	print search.cv_results_

'''(0, 0.004674249800000002, 0.006961569799999998, 0.008737611699999995, 0.009817563300000026, 0.016639071099999986, 0.018635427300000007, 
,0.02920825290000001, 0.03934498460000001, 0.052938155499999986, 0.06204415179999999, 0.09427285350000003, 0.13414432539999993, 
,0.20163216990000005, 0.2684263044000001, 1.1128983988)'''"""

