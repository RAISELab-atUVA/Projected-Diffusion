import torch
import os
import numpy as np
import casadi as ca


class Casadi_Solver:

    def __init__(self, batchsize):

        self.batchsize = batchsize


    def solve(self, primal_input, const_dict):

        print(const_dict)

        # Batch size, number of inequalities, number of equalities
        N = self.batchsize
        verbose = False
        violation = False

        primal = primal_input.detach().cpu() + torch.ones_like(primal_input)

        # plot_points_and_circles(primal[:, :, :2], init=True)

        for idx in range(N):        
        
            ## Bounds ##
            lb = ca.DM(np.repeat(0, 2*64))
            ub = ca.DM(np.repeat(2, 2*64))
            
            cl = -ca.inf
            cu = 0

            # Formulate constraints in Casadi syntax
            constraint_batch = lambda x_i: self.constraints_ca(x_i, const_dict)
            # constraint_batch = lambda x_i: self.constraints_ca_simple(x_i)
            
            gt = ca.reshape(primal.numpy()[idx, :, :2].flatten(), 128, 1)
            
            x = ca.MX.sym('x', 128, 1)
            
            opt_gap = lambda x_i: ca.sumsqr(gt-x_i)

            
            objective = ca.MX(opt_gap(x))

                     # If it exceeds 500 iterations, usually need to change starting point
            opts = {"ipopt.max_iter": 150, "ipopt.print_level": 0, "print_time": 0,
                    "ipopt.constr_viol_tol": 1e-3, "ipopt.tol": 1e-3, "ipopt.mu_strategy": 'adaptive'}
                                            # JC: tolerance changed
            # opts = { 'qpsol': 'qpoases',  # QP solver
            #          'qpsol_options': {'error_on_fail': False} }
            
            nlp = {'x': x, 'f': objective, 'g': constraint_batch(x)}
            
            solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

            result = solver(x0=gt, lbx=lb, ubx=ub, ubg=cu, lbg=cl)
            
            x_sol = np.array(result['x']).flatten()

            points = torch.from_numpy(np.reshape(np.stack(x_sol) , (1, primal.shape[1], 2)))
            # plot_points_and_circles(points)

            solution = torch.cat([(points - torch.ones_like(primal_input[:, : , :2])), primal_input[:, :, 2:]], dim=-1).float() 

            status = solver.stats()
            if status['success']:
                pass
            else:
                # Projection failed (likely stuck at local infeasiblity - we'll take a step and run again)
                print("Solver status:", status['return_status'])
                solution = primal_input
                violation = True

        return solution, violation
        

    def constraints_ca_ex(self, x):
        constraints = []
        rect_centers = [ca.MX([1.0, 1.0])]

        for i in range(64):
            p = x[i, :]
            for rect_center in rect_centers:
                constraints += [ca.fmin(ca.fabs(p[0] - rect_center[0]), ca.fabs(p[1] - rect_center[1]))] 

        return ca.vertcat(*constraints)


    def constraints_ca(self, x, const_dict):

        circle_centers = [
                 [-0.43378472328186035, 0.3334643840789795], [0.3313474655151367, 0.6288051009178162],
                 [-0.5656964778900146, -0.484994500875473], [0.42124247550964355, -0.6656165719032288],
                 [0.05636655166745186, -0.5149664282798767], [-0.36961784958839417, -0.12315540760755539],
                 [-0.8740217089653015, -0.4034936726093292], [-0.6359214186668396, 0.6683124899864197],
                 [0.808782160282135, 0.5287870168685913], [-0.023786112666130066, 0.4590069353580475],
                 [0.11544948071241379, -0.12676022946834564], [0.1455741971731186, 0.16420497000217438],
                 [0.628413736820221, -0.43461447954177856], [0.17965620756149292, -0.8926276564598083],
                 [0.6775968670845032, 0.8817358016967773], [-0.3608766794204712, 0.8313458561897278]
        ]

        rect_centers = [
                 [0.607781708240509, 0.19512386620044708], [0.5575312972068787, 0.5508843064308167],
                 [-0.3352295458316803, -0.6887519359588623], [-0.6572632193565369, 0.31827881932258606],
                 [-0.664594292640686, -0.016457155346870422], [0.8165988922119141, -0.19856023788452148],
                 [-0.8222246170043945, -0.6448580026626587], [-0.2855989933013916, -0.36841487884521484],
                 [-0.8946458101272583, 0.8962447643280029], [-0.23994405567646027, 0.6021060943603516],
                 [-0.006193588487803936, 0.8456171751022339], [0.305103600025177, -0.3661990463733673],
                 [-0.10704007744789124, 0.1318950206041336], [0.7156378626823425, -0.6923345923423767]
        ]


        # Constraints list
        constraints = []

        point_v = False

        if point_v:

            # Rectangle constraints
            for i in range(64):
                p1 = ca.vertcat(*[x[i*2], x[1+i*2]])
                for rect_center in rect_centers:
                    # constraints += [0.21 - ca.fmin(ca.fabs(p1[0] - ca.DM([rect_center[0] + 1.0])), 0.21 - ca.fabs(p1[1] - ca.DM([rect_center[1] + 1.0])))]
                    constraints += [0.142 - ca.norm_2(p1 - ca.DM([rect_center[0] + 1.0, rect_center[1] + 1.0]))]

            # Circle constraints
            for i in range(64):
                p1 = ca.vertcat(*[x[i*2], x[1+i*2]])
                for circle_center in circle_centers:
                                #  Radius                     Distance
                    constraints += [0.2 - ca.norm_2(p1 - ca.DM([circle_center[0] + 1.0, circle_center[1] + 1.0]))]

        else:

            eps = 0.01 #const_dict['eps']

            # Extra obstacle constraints
            for i in range(64 - 1):
                p1 = ca.vertcat(*[x[i*2], x[1+i*2]])
                p2 = ca.vertcat(*[x[(i+1)*2], x[1+(i+1)*2]])

                constraints.append((0.09+eps) - self.line_circle_intersection(p1, p2, ca.DM([-0.4 + 1.0, 0.1 + 1.0])))
                constraints.append((0.12+eps) - self.line_circle_intersection(p1, p2, ca.DM([-0.075 + 1.0, -0.85 + 1.0])))
                constraints.append((0.09+eps) - self.line_circle_intersection(p1, p2, ca.DM([-0.1 + 1.0, -0.1 + 1.0])))

                constraints.append((0.16+eps) - self.line_circle_intersection(p1, p2, ca.DM([0.45 + 1.0, -0.1 + 1.0])))
                constraints.append((0.11+eps) - self.line_circle_intersection(p1, p2, ca.DM([0.35 + 1.0, 0.35 + 1.0])))
                constraints.append((0.085+eps) - self.line_circle_intersection(p1, p2, ca.DM([-0.6 + 1.0, -0.8 + 1.0])))
                constraints.append((0.085+eps) - self.line_circle_intersection(p1, p2, ca.DM([-0.6 + 1.0, -0.9 + 1.0])))
                constraints.append((0.11+eps) - self.line_circle_intersection(p1, p2, ca.DM([-0.65 + 1.0, -0.25 + 1.0])))

            # eps = const_dict['rect']
            # # Rectangle constraints
            # for i in range(64 - 1):
            #     p1 = ca.vertcat(*[x[i*2], x[1+i*2]])
            #     p2 = ca.vertcat(*[x[(i+1)*2], x[1+(i+1)*2]])
            #     for rect_center in rect_centers:
            #                         #  Radius                     Distance from Nearest Point on Line
            #         constraints.append(eps - self.line_circle_intersection(p1, p2, ca.DM([rect_center[0] + 1.0, rect_center[1] + 1.0])))

            eps = const_dict['circle']
            # Circle constraints
            for i in range(64 - 1):
                p1 = ca.vertcat(*[x[i*2], x[1+i*2]])
                p2 = ca.vertcat(*[x[(i+1)*2], x[1+(i+1)*2]])
                for circle_center in circle_centers:
                    constraints.append(eps - self.line_circle_intersection(p1, p2, ca.DM([circle_center[0] + 1.0, circle_center[1] + 1.0])))

        return ca.vertcat(*constraints)


    def constraints_ca_simple(self, x):

        circle_centers = [
                 [-0.43378472328186035, 0.3334643840789795], [0.3313474655151367, 0.6288051009178162],
                 [-0.5656964778900146, -0.484994500875473], [0.42124247550964355, -0.6656165719032288],
                 [0.05636655166745186, -0.5149664282798767], [-0.36961784958839417, -0.12315540760755539],
                 [-0.8740217089653015, -0.4034936726093292], [-0.6359214186668396, 0.6683124899864197],
                 [0.808782160282135, 0.5287870168685913], [-0.023786112666130066, 0.4590069353580475],
                 [0.1455741971731186, 0.16420497000217438],
                 [0.628413736820221, -0.43461447954177856], [0.17965620756149292, -0.8926276564598083],
                 [0.6775968670845032, 0.8817358016967773], [-0.3608766794204712, 0.8313458561897278]]


        # Constraints list
        constraints = []

        point_v = False

        if point_v:

            # Rectangle constraints
            for i in range(64):
                p1 = ca.vertcat(*[x[i*2], x[1+i*2]])
                for rect_center in rect_centers:
                    # constraints += [0.21 - ca.fmin(ca.fabs(p1[0] - ca.DM([rect_center[0] + 1.0])), 0.21 - ca.fabs(p1[1] - ca.DM([rect_center[1] + 1.0])))]
                    constraints += [0.1525 - ca.norm_2(p1 - ca.DM([rect_center[0] + 1.0, rect_center[1] + 1.0]))]

            # Circle constraints
            for i in range(64):
                p1 = ca.vertcat(*[x[i*2], x[1+i*2]])
                for circle_center in circle_centers:
                                #  Radius                     Distance
                    constraints += [0.1645 - ca.norm_2(p1 - ca.DM([circle_center[0] + 1.0, circle_center[1] + 1.0]))]

        else:

            # Extra obstacle constraints
            for i in range(64 - 1):
                p1 = ca.vertcat(*[x[i*2], x[1+i*2]])
                p2 = ca.vertcat(*[x[(i+1)*2], x[1+(i+1)*2]])

                constraints.append(0.09 - self.line_circle_intersection(p1, p2, ca.DM([-0.15 + 1.0, 0.15 + 1.0])))
                constraints.append(0.16 - self.line_circle_intersection(p1, p2, ca.DM([-0.075 + 1.0, -0.85 + 1.0])))
                constraints.append(0.16 - self.line_circle_intersection(p1, p2, ca.DM([-0.1 + 1.0, -0.1 + 1.0])))
                constraints.append(0.16 - self.line_circle_intersection(p1, p2, ca.DM([0.45 + 1.0, -0.1 + 1.0])))
                constraints.append(0.16 - self.line_circle_intersection(p1, p2, ca.DM([0.5 + 1.0, 0.35 + 1.0])))
                constraints.append(0.16 - self.line_circle_intersection(p1, p2, ca.DM([-0.6 + 1.0, -0.85 + 1.0])))
                constraints.append(0.16 - self.line_circle_intersection(p1, p2, ca.DM([0.05 + 1.0, 0.85 + 1.0])))
                constraints.append(0.16 - self.line_circle_intersection(p1, p2, ca.DM([-0.8 + 1.0, 0.15 + 1.0])))
                constraints.append(0.16 - self.line_circle_intersection(p1, p2, ca.DM([0.8 + 1.0, -0.8 + 1.0])))
                constraints.append(0.162 - self.line_circle_intersection(p1, p2, ca.DM([0.45 + 1.0, -0.1 + 1.0])))
                constraints.append(0.135 - self.line_circle_intersection(p1, p2, ca.DM([-0.25 + 1.0, -0.5 + 1.0])))
                constraints.append(0.135 - self.line_circle_intersection(p1, p2, ca.DM([0.8 + 1.0, 0.1 + 1.0])))


            # Circle constraints
            for i in range(64 - 1):
                p1 = ca.vertcat(*[x[i*2], x[1+i*2]])
                p2 = ca.vertcat(*[x[(i+1)*2], x[1+(i+1)*2]])
                for circle_center in circle_centers:
                    constraints.append(0.163 - self.line_circle_intersection(p1, p2, ca.DM([circle_center[0] + 1.0, circle_center[1] + 1.0])))

        return ca.vertcat(*constraints)


    
    def line_circle_intersection(self, p1, p2, circle_center):
        line_dir = p2 - p1
        diff = circle_center - p1
        t = ca.dot(diff, line_dir) / ca.dot(line_dir, line_dir)
        t = ca.fmax(ca.fmin(t, 1), 0) 
        closest_point = p1 + t * line_dir
        dist = ca.norm_2(circle_center - closest_point)
        return dist


import matplotlib.pyplot as plt

def plot_points_and_circles(point_arr, init=False):
    """
    Plots a series of points connected by lines and circular objects.

    :param points: A list of tuples or a 2D array-like of points (x, y).
    :param circles: A list of tuples, where each tuple is (center, radius) 
                    and center is a tuple (x, y).
    """

    circle_centers = np.array(
                [[-0.43378472328186035, 0.3334643840789795], [0.3313474655151367, 0.6288051009178162],
                 [-0.5656964778900146, -0.484994500875473], [0.42124247550964355, -0.6656165719032288],
                 [0.05636655166745186, -0.5149664282798767], [-0.36961784958839417, -0.12315540760755539],
                 [-0.8740217089653015, -0.4034936726093292], [-0.6359214186668396, 0.6683124899864197],
                 [0.808782160282135, 0.5287870168685913], [-0.023786112666130066, 0.4590069353580475],
                 [0.11544948071241379, -0.12676022946834564], [0.1455741971731186, 0.16420497000217438],
                 [0.628413736820221, -0.43461447954177856], [0.17965620756149292, -0.8926276564598083],
                 [0.6775968670845032, 0.8817358016967773], [-0.3608766794204712, 0.8313458561897278],
                 ]).T

    circle_centers = [(1 + circle_centers[0, i], 1 + circle_centers[1, i]) for i in range(16)]

    circle_radii = [0.125, 0.125,0.125, 0.125, 0.125, 0.125, 0.125, 0.125,
                    0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]

    points = [(point_arr[0, i, 0],point_arr[0, i, 1]) for i in range(64) ]


    # Create a new figure
    fig, ax = plt.subplots()

    # Plot the points and connect them with lines
    x_values, y_values = zip(*points)

    # Plot the circles
    for center, radius in zip(circle_centers, circle_radii):
        circle = plt.Circle(center, radius, color='r', fill=False)
        ax.add_patch(circle)

    ax.plot(x_values, y_values, marker='o', linestyle='-')
    
    # Set equal scaling
    ax.set_aspect('equal', adjustable='box')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Points and Circles')

    # Show the plot
    if init: plt.savefig('./obstacle_simple_init.png')
    else: plt.savefig('./obstacle_simple.png')
