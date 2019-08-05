import math
import numpy as np
import scipy
from scipy import spatial
import time
import itertools as itr
import dionysus as dio
import matplotlib.pyplot as plt
from  matplotlib import animation
import Manu_persistence as Mp

#stage  =  4
#islet = 530270
#stage =  2
#islet  = 203210
#islet  = 10341

stage =  3
islet  = 12170

stage =  1
islet  = 240745

#stage =  3
#islet  = 100134

ad_thresh = 30
beta_thresh = ad_thresh
print('Loading Stage:', stage)

#all_islets =  np.load('./Data/stage'+str(stage)+'.npy')
start = time.time()
all_islets =  np.loadtxt('./Data/stage'+str(stage)+'.tsv', delimiter='\t')
finish = time.time()
print('Time taken to load all the islets of this stage:', finish-start)
input('Press key to continue.')

islet_idxs  = np.where(all_islets[:,0] == islet)[0]

islet_info = all_islets[islet_idxs, :]

beta_cells = np.where(islet_info[:,4] == 2)[0].tolist() 
alpha_cells = np.where(islet_info[:,4] == 1)[0].tolist() 
delta_cells = np.where(islet_info[:,4] == 3)[0].tolist()
ad_cells = alpha_cells + delta_cells
#ad_cells = np.where(islet_info[:,4] == 1 or islet_info[:,4] = 3 )[0] 
locations = islet_info[:, 2:4]
print('Loaded information for islet',  islet)
print('Calculating pairwise dist between all of the cells...')
start = time.time()
all_pair_dist = scipy.spatial.distance.cdist(locations, locations)
finish = time.time()
print('Time taken to calculate pairwise dist between all cells:', finish-start)

input('Press key to create and add simplices')
print('Defining simplices (edges and triangles) for max edge length = ', ad_thresh)

#Iterate through ad_cells to find 1-simplices and 2-simplices within threshold
n_ad = len(ad_cells)
max_n_edges = int(n_ad * (n_ad-1)/2)
max_n_triangles = int(n_ad * (n_ad-1) * (n_ad-2)/6)

# Matrix to store vertices in an added simplex
ad_dist = np.zeros((max_n_edges + max_n_triangles, 4))

# List to store unique ad cells with edges
ad_cells_with_edge = []


filtration_list = []
all_columns = []

count = 0

def check_intersects_beta(a1_idx, a2_idx):

    intersects_beta = False
    ad_pair_dist = all_pair_dist[a1_idx, a2_idx]
    possible_beta = []
    for beta in beta_cells:
        if min(all_pair_dist[a1_idx, beta]\
                , all_pair_dist[a2_idx, beta])\
            < ad_pair_dist:
                possible_beta.append(beta)

    for beta_pair in itr.combinations(possible_beta, 2):
        b1_idx = beta_pair[0]
        b2_idx = beta_pair[1]

        ad1_b1_pair_dist = all_pair_dist[a1_idx, b1_idx]
        ad2_b1_pair_dist = all_pair_dist[a2_idx, b1_idx]

        ad1_b2_pair_dist = all_pair_dist[a1_idx, b2_idx]
        ad2_b2_pair_dist = all_pair_dist[a2_idx, b2_idx]

        if min(ad1_b1_pair_dist\
               ,ad2_b1_pair_dist) > ad_pair_dist\
            or \
               min(ad1_b2_pair_dist\
               ,ad2_b2_pair_dist) > ad_pair_dist:
            break
        else:
            if all_pair_dist[b1_idx, b2_idx] < ad_pair_dist:
                A1 = locations[a1_idx]
                A2 = locations[a2_idx]
                
                B1 = locations[b1_idx]
                B2 = locations[b2_idx]

                #t*(A2-A1) - u*(B2-B1)= B1 - A1
                C1 = A2-A1
                C2 = B1-B2
                #print(C1, C2)
                mat = np.array([[C1[0], C2[0]]\
                                ,[C1[1], C2[1]]])

                C3 = B1 - A1

                pars = np.linalg.solve(mat, C3)
                #print(pars)

                if max(pars) <= 1 and min(pars) >=0:
                    intersects_beta = True
                    #print(intersects_beta)
                    #plt.show()
                    break
        #plt.show()
        #print(intersects_beta)

    #print('final ', intersects_beta)

    return intersects_beta
            

def check_intersects_ad(b1_idx, b2_idx):

    intersects_ad = False
    beta_pair_dist = all_pair_dist[b1_idx, b2_idx]
    possible_ad = []
    for ad in ad_cells:
        if min(all_pair_dist[b1_idx, ad]\
                , all_pair_dist[b2_idx, ad])\
            < beta_pair_dist:
                possible_ad.append(ad)

    for ad_pair in itr.combinations(possible_ad, 2):
        ad1_idx = ad_pair[0]
        ad2_idx = ad_pair[1]

        ad1_b1_pair_dist = all_pair_dist[ad1_idx, b1_idx]
        ad2_b1_pair_dist = all_pair_dist[ad2_idx, b1_idx]

        ad1_b2_pair_dist = all_pair_dist[ad1_idx, b2_idx]
        ad2_b2_pair_dist = all_pair_dist[ad2_idx, b2_idx]

        if min(ad1_b1_pair_dist\
               ,ad1_b2_pair_dist) > beta_pair_dist\
            or \
               min(ad2_b1_pair_dist\
               ,ad2_b2_pair_dist) > beta_pair_dist:
            break
        else:
            if all_pair_dist[ad1_idx, ad2_idx] < beta_pair_dist:
                A1 = locations[ad1_idx]
                A2 = locations[ad2_idx]
                
                B1 = locations[b1_idx]
                B2 = locations[b2_idx]

                #t*(A2-A1) - u*(B2-B1)= B1 - A1
                C1 = A2-A1
                C2 = B1-B2
                #print(C1, C2)
                mat = np.array([[C1[0], C2[0]]\
                                ,[C1[1], C2[1]]])

                C3 = B1 - A1

                pars = np.linalg.solve(mat, C3)
                #print(pars)

                if max(pars) <= 1 and min(pars) >=0:
                    intersects_ad = True
                    #print(intersects_beta)
                    #plt.show()
                    break
        #plt.show()
        #print(intersects_beta)

    #print('final ', intersects_beta)

    return intersects_ad
            



def check_contains_beta(v1_idx, v2_idx, v3_idx, max_possible_dist):

    contains_beta = False
    for beta_idx in beta_cells:
        if max(all_pair_dist[beta_idx, v1_idx]\
                , all_pair_dist[beta_idx, v2_idx]\
                , all_pair_dist[beta_idx, v3_idx]\
                ) < max_possible_dist:
            v1 = locations[v1_idx]
            v2 = locations[v2_idx]
            v3 = locations[v3_idx]
            beta = locations[beta_idx]
            # Barycentric check for within triangle

            # Compute vectors
            VV0 = v2 - v1
            VV1 = v3 - v1 
            VV2 = beta - v1

            # Compute dot products
            dot00 = np.dot(VV0 , VV0)
            dot01 = np.dot(VV0 , VV1)
            dot02 = np.dot(VV0 , VV2)
            dot11 = np.dot(VV1 , VV1)
            dot12 = np.dot(VV1 , VV2)

            # Compute barycentric coordinates
            invDenom = 1/(dot00 * dot11 - dot01 * dot01)
            u = (dot11 * dot02 - dot01 * dot12) * invDenom
            v = (dot00 * dot12 - dot01 * dot02) * invDenom

            # Check if point is in triangle
            #contains_beta = (u >= 0) and (v >= 0) and (u + v < 1)
            if (u >= 0) and (v >= 0) and (u + v < 1):
                contains_beta = True
                break
            
    return contains_beta


def check_contains_ad(v1_idx, v2_idx, v3_idx, max_possible_dist):

    contains_ad = False
    for ad_idx in ad_cells:
        if max(all_pair_dist[ad_idx, v1_idx]\
                , all_pair_dist[ad_idx, v2_idx]\
                , all_pair_dist[ad_idx, v3_idx]\
                ) < max_possible_dist:
            v1 = locations[v1_idx]
            v2 = locations[v2_idx]
            v3 = locations[v3_idx]
            ad = locations[ad_idx]
            # Barycentric check for within triangle

            # Compute vectors
            VV0 = v2 - v1
            VV1 = v3 - v1 
            VV2 = ad - v1

            # Compute dot products
            dot00 = np.dot(VV0 , VV0)
            dot01 = np.dot(VV0 , VV1)
            dot02 = np.dot(VV0 , VV2)
            dot11 = np.dot(VV1 , VV1)
            dot12 = np.dot(VV1 , VV2)

            # Compute barycentric coordinates
            invDenom = 1/(dot00 * dot11 - dot01 * dot01)
            u = (dot11 * dot02 - dot01 * dot12) * invDenom
            v = (dot00 * dot12 - dot01 * dot02) * invDenom

            # Check if point is in triangle
            #contains_beta = (u >= 0) and (v >= 0) and (u + v < 1)
            if (u >= 0) and (v >= 0) and (u + v < 1):
                contains_ad = True
                break
            
    return contains_ad



simplices = []

# Adding 1, 2-simplices for ad cells
print('Creating and adding 0, 1, 2-simplices...')
start = time.time()
for ad_idx1 in range(0, n_ad-1):
    ad_1_neighbors = []
    for ad_idx2 in range(ad_idx1+1, n_ad):
        ad_1 = ad_cells[ad_idx1]
        ad_2 = ad_cells[ad_idx2]
        pair_dist  = all_pair_dist[ad_1, ad_2]
        if pair_dist < ad_thresh:
            intersects_beta = check_intersects_beta(ad_1, ad_2)
            if not intersects_beta:
                #print(ad_1, ad_2)
                #simplices.append(([ad_1, ad_2], pair_dist))
                v = [ad_1, ad_2]
                t = pair_dist
                #f.append(dio.Simplex(v, t))
                simplices.append([(ad_1, ad_2), t])
                ad_1_neighbors.append(ad_2)

                if ad_1 not in ad_cells_with_edge:
                    ad_cells_with_edge.append(ad_1)
                if ad_2 not in ad_cells_with_edge:
                    ad_cells_with_edge.append(ad_2)

    for it in itr.combinations(ad_1_neighbors, 2):
        if all_pair_dist[it[0], it[1]] < ad_thresh:
            intersects_beta = check_intersects_beta(it[0], it[1])
            if  not intersects_beta:
                t = max(all_pair_dist[ad_1, it[0]]\
                    ,all_pair_dist[it[0], it[1]]\
                    ,all_pair_dist[ad_1, it[1]]\
                    )
                # Check whether contains beta
                contains_beta = check_contains_beta(ad_1\
                                                    , it[0]\
                                                    ,it[1]\
                                                    , t)
                if not contains_beta:
                    v = [ad_1, it[0], it[1]]
                    #print(v)
                    t =  max(all_pair_dist[ad_1, it[0]]\
                                            ,all_pair_dist[it[0], it[1]]\
                                            ,all_pair_dist[ad_1, it[1]]\
                                            )
                    #f.append(dio.Simplex(v, t))
                    simplices.append([(ad_1, it[0], it[1]), t])


# Add the 0-simplices for ad cells
for v in ad_cells:
    #f.append(dio.Simplex([v], 0))
    simplices.append([(v,), 0])

# Add the 0-simplices for beta cells
for v in beta_cells:
    #f.append(dio.Simplex([v], 0))
    simplices.append([(v,), 0])

# Add the 1,2-simplices for beta cells
n_beta = len(beta_cells)
for beta_idx1 in range(0, n_beta-1):
    b_1_neighbors = []
    for beta_idx2 in range(beta_idx1+1, n_beta):
        b_1 = beta_cells[beta_idx1]
        b_2 = beta_cells[beta_idx2]
        pair_dist = all_pair_dist[b_1, b_2]
        if pair_dist < beta_thresh:
            intersects_ad = check_intersects_ad(b_1, b_2)
            if not intersects_ad:
                v = [b_1, b_2]
                t = pair_dist
                #f.append(dio.Simplex(v, t))
                simplices.append([(b_1, b_2), t])
                b_1_neighbors.append(b_2)

                #if b_1 not in ad_cells_with_edge:
                #    ad_cells_with_edge.append(ad_1)
                #if b_2 not in ad_cells_with_edge:
                #    ad_cells_with_edge.append(ad_2)

    for it in itr.combinations(b_1_neighbors, 2):
        if all_pair_dist[it[0], it[1]] < beta_thresh:
            intersects_ad = check_intersects_ad(it[0], it[1])
            if  not intersects_ad:
                t = max(all_pair_dist[b_1, it[0]]\
                    ,all_pair_dist[it[0], it[1]]\
                    ,all_pair_dist[b_1, it[1]]\
                    )
                # Check whether contains beta
                contains_ad = check_contains_ad(b_1\
                                                ,it[0]\
                                                ,it[1]\
                                                , t)
                if not contains_ad:
                    v = [b_1, it[0], it[1]]
                    #print(v)
                    t =  max(all_pair_dist[b_1, it[0]]\
                             ,all_pair_dist[it[0], it[1]]\
                             ,all_pair_dist[b_1, it[1]]\
                             )
                    #f.append(dio.Simplex(v, t))
                    simplices.append([(b_1, it[0], it[1]), t])


finish = time.time()

print('Time taken to create and add 0, 1, 2-simplices:', finish-start)

# Evaluate the persistence homologies
input('Press key to add and sort simplices, and evaluate persistent homology by Dionysus2...')
# Create the blank filtration
f = dio.Filtration()

for i, simplex in enumerate(simplices):
    f.append(dio.Simplex(simplex[0], simplex[1]))
# Sort the simplices
f.sort()
start = time.time()
per_hom = dio.homology_persistence(f)
finish = time.time()
print('Time taken to evaluate persistent homology by Dionysus:', finish-start) 

input('Press key to add and sort simplices, and evaluate persistent homology by Manu...')
start = time.time()

all_columns = [[],[],[]]
filtration_list = []
max_ones = []
boundary_dict = dict()

# Sort the simplices
simplices = Mp.sort_simplices(simplices)

for i in range(len(simplices)):
    max_ones.append([])


for i,simplex in enumerate(simplices):
    parameter = simplex[1]
    simplex = frozenset(simplex[0])
    filtration_list, max_ones, boundary_dict = \
            Mp.add_simplex(i, simplex, filtration_list, max_ones, parameter\
                            , boundary_dict)
homology_gp = Mp.evaluate_persistence_homology(filtration_list, max_ones)
finish = time.time()
print('Time taken to evaluate persistent homology by Manu:', finish-start) 
input('w')



#homology_gp = [[]]
#dim_prev =  0
#
#for i, fil in enumerate(filtration_list):
#
#    if len(fil[2]) ==0:
#        if max_ones[fil[0]] == []:
#            birth = fil[0]
#            death = math.inf
#            dim_now = len(fil[1])
#            if dim_now > dim_prev:
#                while len(homology_gp) <  dim_now:
#                    homology_gp.append([])
#                dim_prev = dim_now
#            one_vertex = next(iter(filtration_list[birth][1]))
#            if one_vertex in beta_cells:
#                ctype = 'beta'
#            elif one_vertex in ad_cells:
#                ctype = 'ad'
#            else:
#                input('Cell not found?')
#            homology_gp[dim_now-1].append([birth, death, fil[3], ctype])
#    else:
#        birth = max(fil[2])
#        death = fil[0]
#        if filtration_list[death][4] != filtration_list[birth][4]:
#            dim_now = len(filtration_list[birth][1])
#            if dim_now > dim_prev:
#                while len(homology_gp) <  dim_now:
#                    homology_gp.append([])
#                dim_prev = dim_now
#            homology_gp[dim_now-1].append([birth, death, fil[3], ctype])
#
# Sort the bars by birth
for i, hom in enumerate(homology_gp):
    hom.sort(key=lambda hom : hom[0])
    homology_gp[i] = hom



plt.scatter(locations[ad_cells, 0], locations[ad_cells, 1], s = 3, color = 'r',label='alpha or delta cell')
plt.scatter(locations[beta_cells, 0], locations[beta_cells, 1], s = 5, color  = 'g', marker ='*',  label='beta cell')
plt.legend()
x_lim_max = ad_thresh+5

cycles = []

colors = ['r', 'b', 'g', 'y']

for i, hom in enumerate(homology_gp[1]):
    if math.isinf(hom[1]):
        #if hom[3] == 'beta':
        #    color = 'g'
        #elif hom[3] == 'ad':
        #    color = 'r'
        #else:
        #    input('Unexpected ctype')

        cycle_simplices = []
        vertices = []

        for k, col in enumerate(hom[2]):
            simplex = filtration_list[col][1]
            cycle_simplices.append((simplex, k))
            for v in simplex:
                if v not in vertices:
                    vertices.append(v)
                    cycle_simplices.append((frozenset({v,}), 0))

        cycle_simplices = Mp.sort_simplices(cycle_simplices)

        cycle_filtration_list = []
        cycle_max_ones = []
        for s in cycle_simplices:
            cycle_max_ones.append([])
        cycle_boundary_dict = dict()

        for k, simplex in enumerate(cycle_simplices):
            cycle_filtration_list, cycle_max_ones, cycle_boundary_dict = \
                Mp.add_simplex(k, simplex[0], cycle_filtration_list\
                            , cycle_max_ones, k, cycle_boundary_dict)



        cycle_homology = Mp.evaluate_persistence_homology(cycle_filtration_list, cycle_max_ones)

        for k, cycle_hom in enumerate(cycle_homology[1]):
            if math.isinf(cycle_hom[1]):
                for j, col in enumerate(cycle_hom[2]):
                    xx = []
                    yy = []
                    for v in cycle_filtration_list[col][1]:
                        xx.append(locations[v, 0])
                        yy.append(locations[v, 1])
                    plt.plot(xx, yy)
        plt.pause(1)




        #print(cycle_simplices)
        #input('w')
         
        #    xx = []
        #    yy = []
        #    for v in filtration_list[col][1]:
        #        xx.append(locations[v, 0])
        #        yy.append(locations[v, 1])
        #    if len(xx) == 2:
        #        plt.plot(xx, yy, color = color, alpha=0.5)
        #    elif len(xx) == 3:
        #        plt.fill(xx, yy, color = colors[int(i % 4)], alpha=0.5)

        #plt.pause(0.5)


input('w')


























##########################################################
# The persistence diagrams from Dionysus
##########################################################
dgms2 = dio.init_diagrams(per_hom, f)

births = []
deaths = []

for i in range(len(per_hom)):
    if per_hom.pair(i) < i:continue
    dim = f[i].dimension()
    if dim == 1:
        if per_hom.pair(i)!=per_hom.unpaired:
            births.append(i)
            deaths.append(per_hom.pair(i))
        else:
            births.append(i)

fig, (ax1, ax2, ax3) = plt.subplots(3,1)

# This is to try to plot the cycles
# The idea is to cycle through the bars in H_1

n_bars = [len(homology_gp[0]),len(homology_gp[1])]
colors = ["r", "b", "k"]

#for i in range(n_bars):
#    if homology_gp[1][i][1] == math.inf:
#        col_operations = homology_gp[1][i][2]
#        c = colors[int (i % 3)]
#        xx = []
#        yy = []
#        for j, col in enumerate(col_operations):
#            for pt in filtration_list[col][1]:
#                xx.append(locations[pt, 0])
#                yy.append(locations[pt, 1])
#        xx = np.unique(np.array(xx))
#        yy = np.unique(np.array(yy))
#        ax1.fill(xx, yy, color=c, alpha=0.2)
#        #print(filtration_list[col][1])
#        plt.pause(1)
#        #print(col_operations)
#        #input('w')
#input('w')





# Set up the homology barplot for H_1
#dim = 0
#n_cycles = len(dgms[dim])
birth_prev = 0
#print(homology_gp[1])
#input('w')

pts_of_interest = [[], []]
cycle_births = []
col_ops_hom = []

for indices in homology_gp[1]:
    #pts_of_interest[1].append(indices[0])
    if math.isinf(indices[1]):
        cycle_births.append(indices[0])
        col_ops_hom.append(indices[2])
        print(cycle_births, col_ops_hom)
    #if indices[1] != math.inf:
    #    pts_of_interest.append(indices[1])

for indices in homology_gp[0]:
    pts_of_interest[0].append(indices[0])
    #if indices[1] != math.inf:
    #    pts_of_interest.append(indices[1])

#pts_of_interest.sort()
#all_births = pts_of_interest[0] + pts_of_interest[1]


ax1.scatter(locations[ad_cells, 0], locations[ad_cells, 1], s = 3, color = 'r',label='alpha or delta cell')
ax1.scatter(locations[beta_cells, 0], locations[beta_cells, 1], s = 5, color  = 'g', marker ='*',  label='beta cell')
ax1.legend()
x_lim_max = ad_thresh+5
#for i in range(n_bars):
#for i in pts_of_interest:
#for i in pts_of_interest:
for k, i in enumerate(cycle_births):
    #ax1.clear()
    ax2.clear()
    ax3.clear()
    #birth  = int(simplices[homology_gp[1][i][0]][1])
    #birth = homology_gp[1][i][0]
    birth = i
    #death = homology_gp[1][i][1]
    #print(birth, death)
    #print(birth)


    for j in range(birth_prev, birth):
        xx = []
        yy = []
        #print(simplices[j][0])
        for pt in simplices[j][0]:
            xx.append(locations[pt, 0])
            yy.append(locations[pt, 1])
        #print(xx)

        if len(xx) == 1:
            #ax1.scatter(xx, yy, lw=1,c = 'r', alpha=0.5)
            pause_time = 0.005
            continue
        elif len(xx) == 2:
            pause_time = 0.05
            ax1.plot(xx, yy, lw=1,c = 'k', alpha=0.1)
        elif len(yy) == 3:
            pause_time = 0.5
            ax1.fill(xx, yy, c = 'b', alpha=0.1)
        else:
            input('Unexpected')
    #pause_time = 1
    #col_ops = homology_gp[1][k]
    col_ops = col_ops_hom[k]
    print(col_ops)
    #print(col_ops)
    v_in_cycle =[]
    vertex_x = []
    vertex_y = []
    for col in col_ops:
        for vertex in simplices[col][0]:
            if vertex not in  v_in_cycle:
                v_in_cycle.append(vertex)
                vertex_x.append(locations[vertex, 0])
                vertex_y.append(locations[vertex, 1])

        print(simplices[col])
        #print(v_in_cycle)
        print(vertex_x, vertex_y)
    points = np.transpose(np.array([vertex_x, vertex_y]))
    
    color='r'
    ax1.scatter(vertex_x, vertex_y, c=color, lw=2, alpha=0.5)
    hull = scipy.spatial.ConvexHull(points)
    for s in hull.simplices:
        ax1.plot(points[s, 0], points[s, 1], 'k-')



    birth_prev = birth
    #birth_prev = 0

    ax1.set_title('Threshold '+str(simplices[birth][1]))

    ax2.set_ylabel('H_1') 
    ax2.set_xlabel('Threshold') 
    
    ax3.set_ylabel('H_0') 
    ax3.set_xlabel('Threshold') 
    
    ax1.set_xlabel('x-coord of cell')
    ax1.set_ylabel('y-coord of cell')


    flag = 0
    
    # Plotting bar diagram  for Homology group  1
    dim = 1
    #n_bars =  len(homology_gp[dim])
    #print(homology_gp[dim])
    #input('w')
    #ax2.plot([ad_dist[birth,3], ad_dist[birth,3]], [0, n_bars], lw = 1)
    ax2.plot([simplices[birth][1], simplices[birth][1]], [0, n_bars[dim]], lw = 1)
    ax2.plot([ad_thresh, ad_thresh], [0, n_bars[dim]], lw = 3, c='k')

    for j, xx  in enumerate(homology_gp[dim]):
        yy = j+1
        #if math.isinf(xx.death):
        if math.isinf(xx[1]):
            #ax2.plot([xx.birth, x_lim_max], [yy, yy], alpha=0.5)
            ax2.plot([simplices[xx[0]][1], x_lim_max], [yy, yy], alpha=0.5)
            #ax2.set_xlim([0, x_lim_max])
        else:
            #ax2.plot([xx.birth, xx.death], [yy, yy], alpha=0.5)
            ax2.plot([simplices[xx[0]][1], simplices[xx[1]][1]], [yy, yy], alpha=0.5)
    ax2.set_xlim([0, x_lim_max])


    # Plotting bar diagram  for Homology group  0
    dim = 0
    #n_bars =  len(homology_gp[dim])
    #print(homology_gp[dim])
    #input('w')
    #ax2.plot([ad_dist[birth,3], ad_dist[birth,3]], [0, n_bars], lw = 1)
    ax3.plot([simplices[birth][1], simplices[birth][1]], [0, n_bars[dim]], lw = 1)
    ax3.plot([ad_thresh, ad_thresh], [0, n_bars[dim]], lw = 3, c='k')

    for j, xx  in enumerate(homology_gp[dim]):
        yy = j+1
        #if math.isinf(xx.death):
        if math.isinf(xx[1]):
            #ax2.plot([xx.birth, x_lim_max], [yy, yy], alpha=0.5)
            ax3.plot([simplices[xx[0]][1], x_lim_max], [yy, yy], alpha=0.5)
            #ax2.set_xlim([0, x_lim_max])
        else:
            #ax2.plot([xx.birth, xx.death], [yy, yy], alpha=0.5)
            ax3.plot([simplices[xx[0]][1], simplices[xx[1]][1]], [yy, yy], alpha=0.5)
    ax3.set_xlim([0, x_lim_max])





    #dim = 1
    #n_bars =  len(dgms2[dim])
    #ax3.plot([ad_dist[birth,3], ad_dist[birth,3]], [0, n_bars], lw = 1)
    #ax3.plot([ad_thresh, ad_thresh], [0, n_bars], lw = 3, c='k')
    #for j, xx  in enumerate(dgms2[dim]):
    #    yy = j+1
    #    if math.isinf(xx.death):
    #    #if math.isinf(xx[1]):
    #        ax3.plot([xx.birth, x_lim_max], [yy, yy], alpha=0.5)
    #        #ax2.plot([simplices[xx[0]][1], x_lim_max], [yy, yy], alpha=0.5)
    #        #ax2.set_xlim([0, x_lim_max])
    #    else:
    #        ax3.plot([xx.birth, xx.death], [yy, yy], alpha=0.5)
    #        #ax2.plot([simplices[xx[0]][1], simplices[xx[1]][1]], [yy, yy], alpha=0.5)
    #ax3.set_xlim([0, x_lim_max])

    ## Plotting bar diagram  for Homology group  0
    #dim = 0
    #n_bars =  len(dgms[dim])
    #ax3.plot([ad_dist_sorted[i,3], ad_dist_sorted[i,3]], [0, n_bars], lw = 1)
    #ax3.plot([ad_thresh, ad_thresh], [0, n_bars], lw = 3, c='k')
    #for j, xx  in enumerate(dgms[dim]):
    #    yy = j+1
    #    #if math.isinf(xx.death):
    #    if math.isinf(xx[1]):
    #        #ax3.plot([xx.birth, x_lim_max], [yy, yy], alpha=0.5)
    #        ax3.plot([simplices[xx[0]][1], x_lim_max], [yy, yy], alpha=0.5)
    #        #ax3.set_xlim([0, x_lim_max])
    #    else:
    #        #ax3.plot([xx.birth, xx.death], [yy, yy], alpha=0.5)
    #        ax3.plot([simplices[xx[0]][1], simplices[xx[1]][1]], [yy, yy], alpha=0.5)
    #ax3.set_xlim([0, x_lim_max])

    plt.tight_layout()
    plt.pause(pause_time)
    input('w')
    #fig.canvas.draw()
    #fig.canvas.flush_events()
    #plt.draw()

plt.show()






