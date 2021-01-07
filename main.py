from Fairness_In_HAR.DataLoader import dataLoader as DL
from Fairness_In_HAR.Activity-Classifier import activityClassifier as AC
from Fairness_In_HAR.Sensitive-Attr-Classifier import sensitiveAttrClassifier as SAC
from Fairness_In_HAR.FairnessProcessor import fairnessProcessor as FP


activity-results = AC.predict('hmdb', 1, 'test')
sa-results = SAC.predict('hmdb', 1, 'test')
ground-truth = DL.load('hmdb', 1, 'test')

fp-results = FP.evaluate(activity-results, sa-results, ground-truth)








