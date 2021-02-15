import numpy as np
import pandas as pd

class Tree:
    def leaf(data):
        return Tree(data=data)
    def __repr__(self):
        if self.is_leaf():
            return "Leaf(%r)" % self.data
        else:
            return "Tree(%r) { left = %r, right = %r }" % (self.data, self.left, self.right) 
            
    def __init__(self, *, data = None, left = None, right = None):
        self.data = data
        self.left = left
        self.right = right
    
    def is_leaf(self):
        return self.left == None and self.right == None

    def children(self):
        return [x for x in [self.left, self.right] if x]

    def depth(self):
        return max([x.depth() for x in self.children()], default=0) + 1

tree = Tree(data='isSystems?')
tree.left = Tree.leaf(data='like')
tree.right = Tree.leaf(data='TakenOtherSys?')
l1 = Tree.leaf('like')
l2 = Tree.leaf('like')
l3 = Tree.leaf('nah')
l4 = Tree.leaf('nah')
l5 = Tree.leaf('like')
tree4 = Tree(data='likeOtherSys',left=l3,right=l2)
tree3 = Tree(data='morning',left=l5,right=l4)
tree2 = Tree(data='takenOtherSys?', left=tree3, right=tree4)
tree1 = Tree(data='isSystems?', left=l1, right=tree2)



dataset = pd.read_csv(r'dataset.csv')
values = []
for i in dataset['rating']:
    if i>=0:
        values.append(True)
    else:
        values.append(False)
dataset['ok'] = values

columns = list(dataset)
features = []
for i in columns:
    if i != 'ok' and i != 'rating':
        features.append(i)

def single_feature_score(data, goal, feature):
    YES = []
    NO = [] 
    for i in range(len(feature)):
        if goal[i] == True:
            YES.append(feature[i])
        if goal[i] == False:
            NO.append(feature[i])
    NO_true_count = NO.count(True)
    NO_false_count = NO.count(False)
    NO = max(NO_true_count, NO_false_count)
    YES_true_count = YES.count(True)
    YES_false_count = YES.count(False)
    YES = max(YES_true_count, YES_false_count)
    score = YES + NO
    return score

def best_feature(data, goal, features):
    score_list = []
    feature_dict = {}
    for feature in features:
        feature_score = single_feature_score(dataset, goal, dataset[feature])
        feature_dict[feature] = feature_score
        score_list.append(feature_score)
        best = max(score_list)
        worst = min(score_list)
    for k, item in feature_dict.items():
        if item == best:
            best_feature = k
        if item == worst:
            worst_feature = k
    #return best_feature, worst_feature
    return best_feature
best_feature(dataset, dataset['ok'], features) # 'systems'\n",
# best_feature(dataset, dataset['ok'])[1] # 'easy


def DecisionTreeTrain(dataset,goal,features):
    guess = goal.value_counts().idxmax()
    if np.all(goal == goal.iloc[0]):
        return Tree.leaf(guess)
    elif not features:
        return Tree.leaf(guess)
    else:
        current_best_feature = best_feature(dataset, goal, features)
        NO = dataset[dataset[current_best_feature] == False]
        YES = dataset[dataset[current_best_feature] == True]
        features.remove(current_best_feature)
        left = DecisionTreeTrain(NO, goal, features)
        right = DecisionTreeTrain(YES, goal, features)
        return Tree(data = current_best_feature, left = left, right = right)

'''DecisionTreeTrain(dataset,dataset['ok'],features)
Tree('systems') { left = Tree('ai') { left = Tree('theory') { left = Tree('morning') { left = Tree('easy') { left = Leaf(True), right = Leaf(True) }, right = Leaf(True) }, right = Leaf(True) }, right = Leaf(True) }, right = Leaf(True) }'''

def DecisionTreeTrain(dataset,goal,features, maxdepth=None):
    maxdepth = float('inf') if maxdepth is None else maxdepth
    while maxdepth:
        guess = goal.value_counts().idxmax()
        if np.all(goal == goal.iloc[0]):
            return Tree.leaf(guess)
        elif not features:
            return Tree.leaf(guess)
        else:
            current_best_feature = best_feature(dataset, goal, features)
            NO = dataset[dataset[current_best_feature] == False]
            YES = dataset[dataset[current_best_feature] == True]
            features.remove(current_best_feature)
            left = DecisionTreeTrain(NO, goal, features, maxdepth)
            right = DecisionTreeTrain(YES, goal, features, maxdepth)
            return Tree(data = current_best_feature, left = left, right = right)



'''DecisionTreeTrain(dataset,dataset['ok'],features, maxdepth=3)
Tree('systems') { left = Tree('ai') { left = Tree('theory') { left = Tree('morning') { left = Tree('easy') { left = Leaf(True), right = Leaf(True) }, right = Leaf(True) }, right = Leaf(True) }, right = Leaf(True) }, right = Leaf(True) }'''
