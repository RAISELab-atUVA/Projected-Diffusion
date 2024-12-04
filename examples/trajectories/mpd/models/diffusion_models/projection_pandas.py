import torch
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from torch_robotics.trajectory.utils import interpolate_traj_via_points
from solver_pandas import Casadi_Solver
import os
import sys
import math


class Projection:
    
    def __init__(self, hard_conds, const_dict):
        self.hard_conds = hard_conds
        self.const_dict = const_dict
        self.traj_p = False
        
        self.start_x = self.hard_conds[0][0]
        self.end_x = self.hard_conds[63][0]
        
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        
        
    def apply(self, x):  
        # return x, None  

        solver = True
        violation = False
        
        # if self.traj_p: print("| projection | Iter.")
        
        # Reshape to make easier to edit
        x_p = x.reshape(x.shape[0], x.shape[1], 64)

        # print(self.forward_kinematics(x[:,:,1].flatten().cpu()))
        # raise NotImplementedError()
        
        # Start and end points
        x_p = self.project_hard_conds(x_p)
        
        if self.traj_p:
                solver = Casadi_Solver(x.shape[0])
                x_p, violation = solver.solve(x_p.to(device=torch.device('cpu')).permute(0, 2, 1).detach().clone(), self.const_dict)
                x_p = x_p.permute(0, 2, 1).to(device=self.device)
                
        
        x_p = self.project_hard_conds(x_p)
        
        
        # Reshape for model iterations
        x = x_p.reshape(x.shape)
        
        return x.cuda(), violation

    def static(self, x, c):
        
        print(x.shape)
        # x[3, :, 1] = math.pi
        # for i in range(x.shape[0]):
            # x[i, :, :] = torch.zeros((1, 7))
        x = torch.zeros_like(x)
            
            # x[3, :, :]

        # print(x[4])
        position = self.forward_kinematics_custom(x[1,:,:].flatten().cpu())

        return x, position
        

    def project_hard_conds(self, x):
        # print(self.hard_conds[0])
        x[:, :, 0] = self.hard_conds[0]
        x[:, :, 63] = self.hard_conds[63]
        
        return x


    def rotation_matrix(self, axis, theta):
        """
        Compute the rotation matrix for a rotation by `theta` radians around `axis`
        """
        axis = np.asarray(axis)
        axis = axis / np.sqrt(np.dot(axis, axis))
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a*a, b*b, c*c, d*d
        bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
        return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                        [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                        [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


    
    def find_third_point(self, angle_xy, angle_z, distance, offset):
        """
        Compute the position of the third corner of a triangle in 3D space.

        Args:
        - angle_xy: The orientation angle of the triangle on the (x, y) plane in degrees.
        - angle_z: The angle at the second point between the line segment to the unknown point and the xy-plane in degrees.
        - distance: The distance between the second point and the unknown point.

        Returns:
        - A tuple (x, y, z) representing the coordinates of the third point.
        """

        # Convert angles from degrees to radians
        angle_xy_rad = angle_xy + offset*math.pi #math.radians(angle_xy) 
        angle_z_rad = angle_z  + 0.35*math.pi

        print(angle_xy_rad, angle_z_rad)

        # Calculate the change in the z-coordinate from B to C
        delta_z = math.sin(angle_z_rad) * distance

        # Calculate the projection of BC on the xy-plane
        d_xy = math.cos(angle_z_rad) * distance

        # Calculate the changes in the x and y coordinates from B to C
        delta_x = math.cos(angle_xy_rad) * d_xy
        delta_y = math.sin(angle_xy_rad) * d_xy

        # The second point's coordinates
        bx, by, bz = (0, 0, 0.325)

        # Calculate the coordinates of the third point
        cx = bx + delta_x
        cy = by + delta_y
        cz = bz + delta_z

        return (cx, cy, cz)


    def forward_kinematics_custom(self, x_i):

        seg_len = [
                0.325,
                0.325,
                0.13,
                0.1,
                0.08,
            ]


        angles = x_i[:7] 

        # Convert angles to radians
        angles_rad = np.radians(angles.numpy())

        # Initial position
        position = np.array([0, 0, 0])

        # start
        positions = [tuple(position.tolist())]
        
        # joint 1
        positions.append((0, 0, seg_len[0]))

        #joint 2
        # positions.append((0, 0, 0.6))
        # positions.append((0, 0.2, 0.6))
        positions.append((0.1, 0.1, 0.6))
        # positions.append((0.2, 0, 0.6))
        # positions.append((0.2, 0.2, 0.6))
        

        # Initial direction
        # direction = np.array([1, 0, 0])

        # # Axis for rotations
        # z_axis = np.array([0, 0, 1])
        # y_axis = np.array([0, 1, 0])

        # for i, angle in enumerate(angles_rad):
        #     if i < 3: continue
        #     if i % 2 == 0:  # 1st, 3rd, 5th, 7th joints (0-indexed)
        #         # Rotate around z-axis
        #         rot_matrix = self.rotation_matrix(z_axis, angle)
        #     else:
        #         # Rotate around y-axis for elevation change
        #         rot_matrix = self.rotation_matrix(y_axis, angle)
            
        #     # Update direction
        #     direction = np.dot(rot_matrix, direction)

        #     # Move to next joint position
        #     position = position + direction * seg_len[i // 2]
        #     positions.append(tuple(position.tolist()))

        print(len(positions))
        return positions


    def forward_kinematics(self, x_i):
        seg_len = [
                0.125,
                0.125,
                0.13,
                0.1,
                0.08,
            ]

        seg_len = [x*3 for x in seg_len]

        angles = x_i[:7]  # Assuming the first 7 elements are angle degrees

        # Convert angles to radians
        angles_rad = np.radians(angles.numpy())

        # Initial position
        position = np.array([0, 0, 0])
        positions = [tuple(position.tolist())]

        # Initial direction
        direction = np.array([1, 0, 0])

        # Axis for rotations
        z_axis = np.array([0, 0, 1])
        y_axis = np.array([0, 1, 0])

        for i, angle in enumerate(angles_rad):
            if i % 2 == 0:  # 1st, 3rd, 5th, 7th joints (0-indexed)
                # Rotate around z-axis
                rot_matrix = self.rotation_matrix(z_axis, angle)
            else:
                # Rotate around y-axis for elevation change
                rot_matrix = self.rotation_matrix(y_axis, angle)
            
            # Update direction
            direction = np.dot(rot_matrix, direction)

            # Move to next joint position
            position = position + direction * seg_len[i // 2]
            positions.append(tuple(position.tolist()))

        print(len(positions))
        return positions
            