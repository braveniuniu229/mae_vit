"""
@author : braveniuniu
@when : 2023-11-08
"""
import argparse
import os
from scipy.sparse.linalg import spsolve
from rbf.sputils import expand_rows
from rbf.pde.fd import weight_matrix
from rbf.pde.geometry import contains
from rbf.pde.nodes import poisson_disc_nodes
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def getParser():
    paser = argparse.ArgumentParser(descrepition='heatData')
    paser.add_argument('--type_num',default = 1000,type = int)
    paser.add_argument('--sourcerandom',default = False,type = bool)
    paser.add_argument('--source_num_per_type',default = 10,type = int)
    paser.add_argument('--obs_num',default = 16,type = int)
    return paser()

#高斯核
def F(a1, b1, c1, x, y):
    return c1 * np.exp(-((x - a1) ** 2 + (y - b1) ** 2) / 0.05)

#生成全部源项
def PDE(source, a,b,c,x, y):
    out = 0
    for i in range(source):
        out -= F(a[i], b[i], c[i], x, y)
    return out




def  makeMatrix_A(nodes,groups,n,phi,order,N):
    # create the components for the "left hand side" matrix.
    A_interior = weight_matrix(
        x=nodes[groups['interior']],
        p=nodes,
        n=n,
        diffs=[[2, 0], [0, 2]],
        phi=phi,
        order=order)
    A_boundary = weight_matrix(
        x=nodes[groups['boundary:all']],
        p=nodes,
        n=1,
        diffs=[0, 0])
    # Expand and add the components together
    A = expand_rows(A_interior, groups['interior'], N)
    A += expand_rows(A_boundary, groups['boundary:all'], N)
    return  A





def makesquaredata(types,source_num,nodes,groups,n,phi,order,observation_idx,A,source_data,Data_list,F_list,Obs,Source):
    for i in range(types):
        #热源位置
        a = np.random.uniform(0.1, 1.9, source_num)  # (4,)
        b = np.random.uniform(0.1, 1.9, source_num)
        print("Types{}".format(i))
        # print("a", a)
        # print("b", b)
        f_list =[]
        data_list=[]
        obs = []
        source = []

        for j in range(source_data):
            # 设置c为传导率
            c = np.random.uniform(1000, 6000, source_num)
            source.append([a,b,c])
            yr = PDE(source_num,a,b,c,nodes[:,0], nodes[:,1])
            # create "right hand side" vector
            d = np.zeros((N,))
            d[groups['interior']] = yr[groups['interior']]
            d[groups['boundary:all']] = 30
            # 这里是生成原始分辨率的温度场
            u_soln = spsolve(A, d)
            # Create a grid for interpolating the solution
            xg, yg = np.meshgrid(np.linspace(0, 2.0, 64), np.linspace(0, 2.0, 64))
            points = np.array([xg.flatten(), yg.flatten()]).T  #(4096,2)   #最终的分辨率
            # We can use any method of scattered interpolation (e.g.,
            # scipy.interpolate.LinearNDInterpolator). Here we repurpose the RBF-FD method
            # to do the interpolation with a high order of accuracy
            I = weight_matrix(
                x=points,
                p=nodes,
                n=n,
                diffs=[0, 0],
                phi=phi,
                order=order)
            #生成温度场的解
            u_itp = I.dot(u_soln)
            observation = u_itp[observation_idx]
            F1 = PDE(source_num,a,b,c,points[:,0], points[:,1])
            f_list.append(F1.tolist())
            data_list.append(u_itp.tolist())
            obs.append(observation.tolist())
            # ug = u_itp.reshape((64, 64))  # fold back into a grid
            #
            # F11 = F1.reshape((64, 64))
            if (j+1)%100==0:
                # Exact_plot(i, j, F11)
                # Pred_plot(i, j, ug)
                print(f"type:{i+1}\tcount:{j+1}/{source_data}have generated")

        print(f"type{i+1} have already finished!")
        Data_list.append(data_list)
        F_list.append(f_list)
        Obs.append(obs)
        Source.append(source)
    print(f"{source_num}个类型的数据全部生成完成")
    print("现在开始写入数据集。。。。。。。。")
    T = np.array(Data_list)
    F = np.array(F_list)
    O =np.array(Obs)
    S = np.array(Source)
    X = np.array(points)
    file_name='Heat'+'_Types'+str(types)+'_source'+str(source_num)+'_number'+str(source_data)+'fixed'+'.npz'
    np.savez(file_name, T=T,F=F,O=O,S=S,X=X)
    print(f"数据集生成完成！文件名为{file_name}")


def main():
    #基础设置
    vert = np.array([[0.0, 0.0], [2.0, 0.0],
                 [2.0, 2.0], [0.0, 2.0]])
    smp = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    spacing = 0.05  # approximate spacing between nodes
    n = 25  # stencil size. Increase this will generally improve accuracy
    phi = 'phs3'  # radial basis function used to compute the weights. Odd
    # order polyharmonic splines (e.g., phs3) have always performed
    # well for me and they do not require the user to tune a shape
    #  parameter. Use higher order polyharmonic splines for higher
    # order PDEs.
    order = 2  # Order of the added polynomials. This should be at least as
    # large as the order of the PDE being solved (2 in this case). Larger
    # values may improve accuracy
    #
    # generate nodes
    nodes, groups, _ = poisson_disc_nodes(spacing, (vert, smp))
    N = nodes.shape[0]   #Amount of original nodes
    # 生成从0到4095的16个均匀间隔的数字
    # 注意，我们使用4095作为终点因为索引是从0开始的
    observation_idx = np.linspace(0, 4095, 16, endpoint=True)

    # 四舍五入并转换为整数
    observation_idx = np.round(observation_idx).astype(int)
    Data_list = []  # 温度场样本    （types，source_data,64,64）
    Source = []  # 测点样本    （types，source_data,16）
    F_list = []  # 热源样本   （types，source_data,64,64）
    Obs = []  # 热源系数 （a，b，c）形状为（3，4）   最终（types，source_data,3，4）
    parser = getParser()
    args = parser.parse_args()
    types = args.type_num  # 布局数量
    source_num = 4 if args.sourcerandom else np.randint(3, 7)  # 热源数量
    source_data = args.source_data
    A = makeMatrix_A(nodes=nodes,groups=groups,n=n,phi=phi,N=N)
    #开始制作数据集
    makesquaredata(types=types,
                   source_num=source_num,
                   nodes=nodes,
                   groups=groups,
                   n=n,
                   phi=phi,
                   order=order,
                   observation_idx=observation_idx,
                   A=A,
                   Data_list=Data_list,
                   Source=Source,
                   F_list=F_list,
                   Obs=Obs,
                   source_data=source_data)
