import torch
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from torch_robotics.trajectory.utils import interpolate_traj_via_points
from mpd.models.diffusion_models.solver import Casadi_Solver
import os
import sys


class Projection:
    
    def __init__(self, hard_conds, const_dict):
        self.hard_conds = hard_conds
        self.const_dict = const_dict
        self.traj_p = False
        
        self.start_x = self.hard_conds[0][0]
        self.end_x = self.hard_conds[63][0]
        
        # Env Constraints
        self.box_center = torch.tensor(
                [[0.607781708240509, 0.19512386620044708], [0.5575312972068787, 0.5508843064308167],
                 [-0.3352295458316803, -0.6887519359588623], [-0.6572632193565369, 0.31827881932258606],
                 [-0.664594292640686, -0.016457155346870422], [0.8165988922119141, -0.19856023788452148],
                 [-0.8222246170043945, -0.6448580026626587], [-0.2855989933013916, -0.36841487884521484],
                 [-0.8946458101272583, 0.8962447643280029], [-0.23994405567646027, 0.6021060943603516],
                 [-0.006193588487803936, 0.8456171751022339], [0.305103600025177, -0.3661990463733673],
                 [-0.10704007744789124, 0.1318950206041336], [0.7156378626823425, -0.6923345923423767]
                 ]
                ).to(device=torch.device('cuda:0'))
        
        self.box_w = 0.22
        
        self.circle_centers = torch.tensor(
                [[-0.43378472328186035, 0.3334643840789795], [0.3313474655151367, 0.6288051009178162],
                 [-0.5656964778900146, -0.484994500875473], [0.42124247550964355, -0.6656165719032288],
                 [0.05636655166745186, -0.5149664282798767], [-0.36961784958839417, -0.12315540760755539],
                 [-0.8740217089653015, -0.4034936726093292], [-0.6359214186668396, 0.6683124899864197],
                 [0.808782160282135, 0.5287870168685913], [-0.023786112666130066, 0.4590069353580475],
                 [0.11544948071241379, -0.12676022946834564], [0.1455741971731186, 0.16420497000217438],
                 [0.628413736820221, -0.43461447954177856], [0.17965620756149292, -0.8926276564598083],
                 [0.6775968670845032, 0.8817358016967773], [-0.3608766794204712, 0.8313458561897278],
                 ]
                ).to(device=torch.device('cuda:0'))
        
        self.circle_r = 0.145
        
        
    def apply(self, x):        
        solver = True
        violation = False
        
        # if self.traj_p: print("| projection | Iter.")
        
        # Reshape to make easier to edit
        x_p = x.reshape(x.shape[0], 4, 64)
        
        # Start and end points
        x_p = self.project_hard_conds(x_p)
        
        if self.traj_p:
            if solver:
                solver = Casadi_Solver(x.shape[0])
                x_p, violation = solver.solve(x_p.to(device=torch.device('cpu')).permute(0, 2, 1).detach().clone(), self.const_dict)
                x_p = x_p.permute(0, 2, 1).to(device=torch.device('cuda:0'))
            else:
                x_p = torch.stack([self.point_collisions(x_pi) for i, x_pi in enumerate(x_p)])
        
        x_p = self.project_hard_conds(x_p)
        
        
        # Reshape for model iterations
        x = x_p.reshape(x.shape)
        
        # print(f"| ep | {x[:, :, 0]} == {self.hard_conds[0]}")
        return x, violation
        
    def uniform_x(self, x):
        step = (self.end_x - self.start_x) / 64
        
        for i in range(1, 63):
            x[0, i] = (i*step) + self.start_x
        
        return x
    
    def project_hard_conds(self, x):
        x[:, :, 0] = self.hard_conds[0]
        x[:, :, 63] = self.hard_conds[63]
        
        return x
    
    # Non-convex, so need to code by hand
#     def point_collisions(self, x):
        
#         # print(self.box_center.shape, self.box_w_h.shape)
#         # print(x.shape)
#         # raise RunTimeError

#         # Convert PyTorch tensor to NumPy array
#         x_numpy = x.detach().cpu().numpy()

#         # Define CVXPY problem
#         x_cvx = cp.Variable(x_numpy.shape)
        
#         constraints = []
#         epsilon = 0.21
#         # for i in range(64):
#         for i in range(2):
#             for j in range(self.box_center.shape[0]):
#                 # dist = cp.norm(x_cvx[:2, i] - self.box_center[j], 2)
#                 # constraints.append(dist >= 0.21)
#                 # dist_squared = cp.sum_squares(x_cvx[:2, i] - self.box_center[j])
#                 # constraints.append(dist_squared >= 0.21**2)
#                 dx = cp.abs(x_cvx[i, 0] - self.box_center[j, 0])
#                 constraints.append(dx >= epsilon)
#                 dy = cp.abs(x_cvx[i, 1] - self.box_center[j, 1])
#                 constraints.append(dy >= epsilon)
        
#         objective = cp.Minimize(cp.norm(x_cvx - x_numpy, 2))

#         problem = cp.Problem(objective, constraints)

#         # Solve the problem
#         problem.solve()

#         return torch.tensor(x_cvx.value, dtype=torch.float64)

    def point_collisions(self, x):
        epsilon = self.box_w
        radius = self.circle_r
        
        # Correct Square Regions
        for i in range(x.shape[1]):
            point = x[:2, i]  
            for center in self.box_center:
                if all(center - epsilon <= point) and all(point <= center + epsilon):
                    x[:2, i] = self.find_nearest_edge(point, center, epsilon)
                    break
               
        # Correct Circular Regions
        for i in range(x.shape[1]):
            point = x[:2, i]
            for center in self.circle_centers:
                if torch.linalg.norm(point - center) < radius:
                    x[:2, i] = self.project_to_circle(point, center, radius)
                    break
        
        return x
    
    
    def find_nearest_edge(self, point, center, epsilon):
        x_min, y_min = center - epsilon
        x_max, y_max = center + epsilon

        # Find the nearest x coordinate
        nearest_x = min(max(point[0], x_min), x_max)

        # Find the nearest y coordinate
        nearest_y = min(max(point[1], y_min), y_max)

        # Project to the nearest edge if inside the box
        if nearest_x == point[0] and nearest_y == point[1]:
            if abs(point[0] - x_min) < abs(point[0] - x_max):
                nearest_x = x_min
            else:
                nearest_x = x_max

            if abs(point[1] - y_min) < abs(point[1] - y_max):
                nearest_y = y_min
            else:
                nearest_y = y_max

        return torch.tensor([nearest_x, nearest_y]).to(device=torch.device('cuda:0'))

    
    def project_to_circle(self, point, center, radius):
        direction = point - center
        distance_to_center = torch.linalg.norm(direction)
        if distance_to_center < radius:
            # Project to the circumference
            projected_point = center + (direction / distance_to_center) * radius
            return projected_point
        else:
            return point
        
        
#     def traj_collisions(self, x):
        
#         x_interpolated = interpolate_traj_via_points(x.T, num_interpolation=5).T
        
#         epsilon = self.box_w
#         radius = self.circle_r
        
#         # Correct Square Regions
#         for i in range(x.shape[1]-1):
#             point = x[:2, i]  
            
#             for j in range(5):
#                 point_interpolated = x_interpolated[:2, (i*5)+j]
                
#                 for center in self.box_center:
#                     if all(center - epsilon <= point_interpolated) and all(point_interpolated <= center + epsilon):
#                         x[:2, i], x[:2, i+1] = self.adjust_points(point, x[:2, i+1], point_interpolated, center, epsilon)
#                         break
               
#         # Correct Circular Regions
#         for i in range(x.shape[1]-1):
#             point = x[:2, i]
            
#             for j in range(5):
#                 point_interpolated = x_interpolated[:2, (i*5)+j]
                
#                 for center in self.circle_centers:
#                     if torch.linalg.norm(point_interpolated - center) < radius:
#                         x[:2, i], x[:2, i+1] = self.adjust_points(point, x[:2, i+1], point_interpolated, center, radius)
#                         break
        
#         return x
    
    
#     def adjust_points(self, p1, p2, collision_point, region_center, region_radius):
#         # Vector from region center to collision point
#         direction = collision_point - region_center
#         direction_norm = direction / torch.linalg.norm(direction)
        
#         if torch.all(direction == 0):
#             return p1, p2 

#         # Distance from region boundary to collision point
#         displacement = torch.linalg.norm(collision_point - region_center) - region_radius

#         # Adjust original points slightly more than displacement
#         adjustment = (displacement + 0.1) * direction_norm

#         return p1 - adjustment, p2 - adjustment
    
    def normalize_sample(self, tensor):
        max_val = torch.max(torch.abs(tensor))
        return (tensor / max_val)
    

        
    def traj_collisions(self, x):
        intersections = True
        max_iter = 50
        while intersections and max_iter > 0:
            intersections = False
            max_iter -= 1

            for i in range(x.shape[1] - 1):
                point1 = x[:2, i]
                point2 = x[:2, i + 1]
                line = LineString([point1, point2])

                # Check and adjust for rectangles
                for center in self.box_center:
                    rectangle = self.create_rectangle(center.detach().cpu().numpy(), self.box_w)
                    # while line.intersects(rectangle):
                    if line.intersects(rectangle):
                        intersections = True
                        # Adjust the points - Implement your logic here
                        x[:2, i], x[:2, i + 1] = self.adjust_line_segment(point1, point2, rectangle)
                        line = LineString([x[:2, i], x[:2, i + 1]])

                # Check and adjust for circles
                for center in self.circle_centers:
                    circle = self.create_circle(center.detach().cpu().numpy(), self.circle_r)
                    # while line.intersects(circle):
                    if line.intersects(circle):
                        intersections = True
                        # Adjust the points - Implement your logic here
                        x[:2, i], x[:2, i + 1] = self.adjust_line_segment(point1, point2, circle)
                        line = LineString([x[:2, i], x[:2, i + 1]])
                        
        if max_iter == 0: print("| warning | Max iter reached.")
        return x
                
    
    def adjust_line_segment(self, p1, p2, region):
        # Create a LineString from the points
        line = LineString([p1, p2])

        # Find the intersection point(s)
        intersection = line.intersection(region)

        # If there's no intersection or it's a complex intersection, return the points as-is
        if not intersection or not isinstance(intersection, Point):
            return p1, p2

        # Calculate retraction direction for p2 (away from the intersection)
        direction = np.array(p2) - np.array(intersection)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction /= norm

        # Define a retraction distance (adjust this based on your requirements)
        retraction_distance = 0.1  # Example distance

        # Retract p2
        p2_adjusted = np.array(p2) + direction * retraction_distance

        return p1, p2_adjusted

    def create_rectangle(self, center, epsilon):
        """Create a rectangle (as a Polygon) from a center point and epsilon."""
        return Polygon([
            (center[0] - epsilon, center[1] - epsilon),
            (center[0] - epsilon, center[1] + epsilon),
            (center[0] + epsilon, center[1] + epsilon),
            (center[0] + epsilon, center[1] - epsilon)
        ])

    def create_circle(self, center, radius):
        """Create a circle (as a Polygon) from a center point and radius."""
        return Point(center).buffer(radius)

    
    def blockPrint(self):
        sys.stdout = open(os.devnull, 'w')

    def enablePrint(self):
        sys.stdout = sys.__stdout__