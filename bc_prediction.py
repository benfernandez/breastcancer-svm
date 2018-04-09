from sklearn.datasets import load_breast_cancer
from sklearn import svm

bc = load_breast_cancer();

clf = svm.SVC(gamma =0.001, C=100)

print(bc.keys())
print("Size: ",len(bc.data))
print("Features: ",list(bc.feature_names))
print("Target names: ",list(bc.target_names))

bc_half = int(len(bc.data) / 2)


datax,targety = bc.data[:bc_half],bc.target[:bc_half]

print(datax)
print(targety)

clf.fit(datax,targety)

for x in range(1,11):
    print("Prediction: ",clf.predict(bc.data[[-x]]))
    print("Target:     ",bc.target[[-x]],"\n")


print("Accuracy: ",clf.score(bc.data,bc.target))
