from random import seed
from random import randrange
from csv import reader
import matplotlib.pyplot as plt 
 
 
def float_conversion(f, y):          #column = y
    for i in range(len(f)):
        x = f[i]
        a = x[y].strip()
        x[y] = float(a)    
 
def int_conversion(f, y):
    a = []
    for x in f:
        x[y] = float(x[y])
        a = a + [x[y]]
    m1 = float(min(a))
    m2 = float(max(a))
    
    ax = float(m2 - m1)
    ac = float(ax/3)      
    i = float(m1 + ac)
    j = float(m1 + (2*ac))
    
    for x in f:
        x[y] = float(x[y])
        if x[y] >= m1 and x[y] < i:
            x[y] = 0
        elif x[y] >= i and x[y] < j:
            x[y] = 1
        else:
            x[y] = 2
    return


def tree(f, algo, n, *arguments):
    fs = []
    fs1 = list(f)
    l = len(f)
    size_of_fd = int(l/n)
    for i in range(n):
        fd = []
        while len(fd) < size_of_fd:
            ind = randrange(len(fs1))
            a = fs1.pop(ind)
            fd = fd + [a]
        fs = fs + [fd]
    
    K_fds = fs
    s = []
    for ij in range(len(K_fds)):
        k_fd = K_fds[ij]
        x_set = list(K_fds)             #train_set = x    test_set = y
        x_set.remove(k_fd)
        x_set = sum(x_set, [])
        y_set = []
        for j in range(len(k_fd)):
            x = k_fd[j]
            x1 = list(x)               #x_copy = x1    
            y_set = y_set + [x1]
            x1[-1] = 0
        pred = algo(x_set, y_set, *arguments)
        act = []
        for k in range(len(k_fd)):
            x = k_fd[k]
            act = act + [x[-1]]
            
        l1 = len(act)
        c1 = 0
        for j in range(l1):
            if act[j] == pred[j]:
                c1 = c1 + 1
        d = float(l1)
        e = (c1/d)*100.0
        acc = e
        s = s + [acc]
        
    return s
 

def seperate(f):                    
    c = []
    for i in range(len(f)):
        x = f[i]
        c= c + [x[-1]]
    
    bind = 100
    bval = 100
    bsc = 100
    b_gps = 0
    
    l = len(f[0])
    a = l-1
    for ind in range(a):
        for i in range(len(f)):
            x = f[i]
            u = []
            v = []
            for x1 in f:
                if x1[ind] < x[ind]:
                    u = u + [x1]
                else:
                    v = v + [x1]
            gps = (u,v)
            a1 = []
            for j in range(len(gps)):
                gp = gps[j]
                a1 = a1 + [len(gp)]
            no_of_inst = float(sum(a1))    
            gini_ind = 0.0
            for k in range(len(gps)):
                gp = gps[k]
                s1 = float(len(gp))
                if s1 == 0:
                    continue
                sc1 = 0.0
                for m in range(len(c)):
                    class_val = c[m]
                    b = [row[-1] for row in gp].count(class_val)
                    p = b / s1
                    sc1 = sc1 + (p * p)
                x2 = (1.0 - sc1)
                y1 = (s1 / no_of_inst)
                gini_ind = gini_ind + (x2*y1)
            g = gini_ind
            if g < bsc:
                bind = ind
                bval = x[ind]
                bsc = g
                b_gps = gps
    return {'index':bind, 'value':bval, 'groups':b_gps}    

 
def terminal_nd1(gp):
    r = []
    for i in range(len(gp)):
        x = gp[i]
        r = r + [x[-1]]
    a = set(r)
    b = max(a, key=r.count)   
    return b 

 
def divide(nd1, md, ms, d):
    g = 'groups'
    lt = 'left'
    rt = 'right'
    
    l, r = nd1[g]
    del(nd1[g])
    
    if not l or not r:
        nd1[lt] = terminal_nd1(l + r)
        nd1[rt] = terminal_nd1(l + r)
        return
   
    if d >= md:
        nd1[lt] = terminal_nd1(l)
        nd1[rt] = terminal_nd1(r)
        return
   
    if len(l) > ms:
        nd1[lt] = seperate(l)
        divide(nd1[lt], md, ms, d+1)
    else:
        nd1[lt] = terminal_nd1(l)
   
    if len(r) > ms:
        nd1[rt] = seperate(r)
        divide(nd1[rt], md, ms, d+1)    
    else:
        nd1[rt] = terminal_nd1(r)    
 
def prediction(nd, x):
    v = 'value'
    i = 'index'
    l = 'left'
    r = 'right'
    d = dict
    if nd[v] > x[nd[i]]:
        if isinstance(nd[l], d):
            return prediction(nd[l], x)
        else:
            return nd[l]
    else:
        if isinstance(nd[r], d):
            return prediction(nd[r], x)
        else:
            return nd[r]    
  

 
def bagging(x_t, y_t, md, ms, ss, n_t):
    t = []
    ij = 0
    while ij < n_t:
        ij = ij+1
        s1 = []
        l1 = (len(x_t) * ss)
        n1 = round(l1)          #n = n_sample
        while len(s1) < n1:
            a1 = len(x_t)
            j = randrange(a1)
            s1 = s1 + [x_t[j]]
    
        s = s1
        r = seperate(s)
        divide(r, md, ms, 1)
        t1 = r
        t.append(t1)
    predict = []
    for j in range(len(y_t)):
        x = y_t[j]       
        p = []
        for k in range(len(t)):
            t1 = t[k]
            p = p + [prediction(t1,x)]
        a1 = set(p)    
        b1 = max(a1, key= p.count)
        predict = predict + [b1]
    
    return(predict)
     
    
    
seed(1101)
f = 'C:/Project/hou_all.csv'
data = []
with open(f, 'r') as f1:
    file1 = reader(f1)
    for x in file1:        #row = x
        data = data + [x]
    
f1 = data
a = len(f1)
f = f1[1:a]

length = len(f[0])

for i in range(0, length-1):
    float_conversion(f, i)
    
int_conversion(f, length-1)

x = []
y = []

no_of_fds = 5
sample_size_tree = 0.5
size = 2

for depth in [6]:
    for no_of_trees in [1,2]:
        x = x + [no_of_trees]
        accuracy = tree(f, bagging, no_of_fds, depth, size, sample_size_tree, no_of_trees)
        print('Depth: %d' % depth)
        print('Number of Trees: %d' % no_of_trees)
        print('Cross validation Scores: %s' % accuracy)
        a = float(len(accuracy))
        b = sum(accuracy)
        s = (b/a)
        print('Accuracy of Model: %.2f%%' % s)
        y = y + [s]
        
plt.plot(x,y)
plt.xlabel('Number of trees')
plt.ylabel('Accuracy')