from sklearn.datasets import load_breast_cancer
from sklearn import svm

bc = load_breast_cancer();

clf = svm.SVC(gamma =0.001, C=100)

print(bc.keys())
print(len(bc.data))
print(type(bc.data))

bc_half = int(len(bc.data) / 2)
print(bc_half)

datax,targety = bc.data[:bc_half],bc.target[:bc_half]
clf.fit(datax,targety)

for x in range(1,11):
    print("Prediction:    ",clf.predict(bc.data[[-x]]))
    print("Actual target: ",bc.target[[-x]],"\n")


print(list(bc.target_names))
print("Accuracy: ",clf.score(bc.data,bc.target))
