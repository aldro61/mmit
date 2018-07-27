# Load CSV (using python)
import csv,math,copy
import numpy as np
import mmit,time
from mmit import MaxMarginIntervalTree
from mmit.pruning import min_cost_complexity_pruning
from mmit.model_selection import GridSearchCV
from mmit.metrics import mean_squared_error
from mmit.model import _latex_export
from os import *
from os.path import *
from mmit_predictions import Dataset

def find_datasets(path):
    for d in listdir(path):
    	print d
        if exists(join(path, d, "features.csv")) and \
                exists(join(path, d, "targets.csv")) and \
                exists(join(path, d, "folds.csv")):
            yield Dataset(abspath(join(path, d)))

for d in find_datasets("/home/parismita/mmit_data"):
	x = d.X
	y = d.y

	trainx = x[:len(x)/2,]
	trainy = y[:len(y)/2,]
	testx = x[len(x)/2:,]
	testy = y[len(y)/2:,]

	start_time = time.time()
	estimator = MaxMarginIntervalTree(margin=1.0, max_depth=4, loss = "linear_hinge", min_samples_split = 0)
	clf = estimator.fit(trainx, trainy)
	fit = estimator.predict(testx)

	#print time.time() - start_time 
	#print len(x)
	print "|  ", mean_squared_error(testy, fit)
	#file = open(str(i)+".tex", 'w')
	#file.write( _latex_export(estimator))
	#file.close()
	

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



