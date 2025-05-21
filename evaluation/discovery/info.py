import numpy as np

class scene:
  def __init__(self,):
    self.scenes = {
      "Reflection": {
        #![](https://cdn.jsdelivr.net/gh/DishengL/ResearchPics/render_output_Blank_Reflection_circle_320x320.png)
        "file_name": "Dataset/Real_reflection_v2__256P/real_rendered_reflection_256P",
        "variables": {0: "incident_degree", 1: "reflection_degree"},
        "adjacency_matrix": np.array([[0, 1], [0, 0]]),
        "sample_result": """
Based on the law of reflection, the angle of incidence is equal to the angle of reflection. Therefore, the incident angle determines the reflection angle.
The causality adjacency matrix for the variables ""incident_degree"" and ""reflection_degree"" can be represented as:

```
[[0, 1],
[0, 0]]
```

Explanation:
- ""incident_degree"" causes ""reflection_degree"" (hence, matrix[0][1] = 1).
- There is no direct causal relationship where ""reflection_degree"" causes ""incident_degree"" (hence, matrix[1][0] = 0).
"""
      },

      "Spring": {
        # ![](https://cdn.jsdelivr.net/gh/DishengL/ResearchPics/render_output_Blank_Spring.png)
        "file_name": "Dataset/Real_spring_v3_256P/Real_spring_v3_256P",
        "variables": {0: "spring_constant", 1: "weight", 2: "defomation of spring"},
        "adjacency_matrix": np.array([[0,0,1],[0,0,1],[0,0,0]]),
        "sample_result": """
To determine the causal relationships between the spring constant, weight, and deformation of the spring, we can use Hooke's Law, which states that the force exerted by a spring is directly proportional to the deformation (displacement) of the spring, given by:

\[ F = k \cdot x \]

Where:
- \( F \) is the force applied (related to weight),
- \( k \) is the spring constant,
- \( x \) is the deformation of the spring.

From this, we can infer:

1. **spring_constant** (\( k \)) affects **defomation of spring** (\( x \)): If the spring constant increases, for the same weight, the deformation decreases.
2. **weight** affects **defomation of spring** (\( x \)): An increase in weight causes more deformation.
3. The **spring_constant** (\( k \)) and **weight** do not directly affect each other.

Based on these relationships, the causality adjacency matrix would be:

```
[[0, 0, 1],
 [0, 0, 1],
 [0, 0, 0]]
```

Explanation:
- Element (1,3) is 1 because the spring constant affects deformation.
- Element (2,3) is 1 because the weight affects deformation.
- The other entries are 0 because there is no direct causal relationship otherwise.
"""
      },

      "Seesaw": {
        # ![](https://cdn.jsdelivr.net/gh/DishengL/ResearchPics/20241120_150057.png)
        "file_name": "Dataset/Real_seesaw_v3_256P/Real_seesaw_v3_256P",
        "variables": {0: "left_force", 1: "right_force", 2: "Seesaw Moment (tilt side)"},
        "adjacency_matrix": np.array([[0,0,1],[0,0,1],[0,0,0]]),
        
        "variables_2": {0: "seesaw_left_arm", 1: "left_weight", 2: "seesaw_right_arm", 3: "right_weight", 4: "seesaw_torque"},
        "adjacency_matrix_2": np.array([[0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1],
                                        [0, 0, 0, 0, 0]]),
        "sample_result": """
To determine the causality adjacency matrix, consider the relationship between the forces applied on either side of the seesaw and its moment or tilt. Changes in `left_force` and `right_force` can directly influence the `Seesaw Moment`, deciding which side tilts.
Here's the filled matrix:
```
[[0, 0, 1],
 [0, 0, 1],
 [0, 0, 0]]
```

### Explanation:
- `left_force` and `right_force` affect the `Seesaw Moment` by altering the balance; hence both have a direct causal relationship to the tilt.
- The `Seesaw Moment` (tilt) does not directly cause changes in either `left_force` or `right_force`; rather it is a result of the two forces.
"""      
      },

      "Magnets": {
          # Image illustrating the setup
          # ![](https://cdn.jsdelivr.net/gh/DishengL/ResearchPics/20241123_161201_Over_3D_256p.png)
          "file_name": "Dataset/Real_magnet_v3_256P/Real_magnet_v3",
          
          "variables": {
              0: "needle_position_x",             # Position of the needle in horizontal direction
              1: "needle_position_y",             # Position of the needle in vertical direction
              2: "magnetic_bar_orientation",   # Orientation of the magnetic bar
              3: "needle_orientation"          # Orientation of the needle
          },
          "adjacency_matrix": np.array([
              # Rows and columns correspond to the indices in "variables"
              # 0: needle_position, 1: magnetic_bar_orientation, 2: needle_orientation
              [0, 0, 0, 1],  # needle_position_x affects needle_orientation
              [0, 0, 0, 1],  # needle_position_x affects needle_orientation
              [0, 0, 0, 1],   # magnetic_bar_orientation affects needle_orientation
              [0, 0, 0, 0]   # needle_orientation does not affect others
          ])
      },
      
      "Convex": {
          # Image illustrating the setup
          "file_name": "Dataset/Real_convex_len_v3_512x256/convex_len_render_images",
          "variables": {
              0: "the distance from object to the convex len",             # Position of the needle in horizontal direction
              1: "the distance from image to the convex len",             # Position of the needle in vertical direction
              2: "the Magnification",   # Orientation of the magnetic bar
          },
          "adjacency_matrix": np.array([
              # Rows and columns correspond to the indices in "variables"
              [0,1, 1],
              [0,0, 1],
              [0,0, 0]
          ])
      },
      
      "Parabola": {
          # Image illustrating the setup
          "file_name": "Dataset/Real_parabola_v4_512x256/generated_images",
          "variables": {
                0: "Deformation of spring",
                1: "Emergence angle",
                2: "Height of highest point",
                3: "Horizontal distance",
          },
          "adjacency_matrix": np.array([
              # Rows and columns correspond to the indices in "variables"
              [0,0, 1, 1],
              [0,0, 1, 1],
              [0,0, 0, 0],
              [0,0, 0, 0]
          ]),
          "sample_result": """
To determine the causal relationships, we can consider the principles of projectile motion and springs. Here's a possible breakdown:

1. **Deformation of spring**: The more a spring is compressed (or stretched), the more energy is stored, affecting the speed of the object when released, which in turn influences the height and horizontal distance.

2. **Emergence angle**: The angle affects the trajectory, which determines both the height and the horizontal distance.

3. **Height of highest point**: This is determined by the initial speed and angle but doesn‚Äôt directly affect any other variables in this context.

4. **Horizontal distance**: This is determined by the initial speed and angle.

Using these insights, a possible causality adjacency matrix is:

```
[[0, 1, 1, 1],
 [0, 0, 1, 1],
 [0, 0, 0, 0],
 [0, 0, 0, 0]]
```

Explanation:
- **Deformation of spring** causes emergence angle, height of highest point, and horizontal distance due to the initial speed provided by the spring.
- **Emergence angle** directly affects the height of the highest point and the horizontal distance (as it determines the projectile's trajectory).
- **Height of highest point** and **horizontal distance** do not cause changes in other variables in this scenario.
"""
      },
      
      "Waterflow": {
          # Image illustrating the setup
          "file_name": "Dataset/Real_water_flow_v5_256/Water_flow_scene_render",
          
          "variables": {
                0: "Ball volumn",
                1: "Diameter of the bottom of cup",
                2: "Water height in the cup",
                3: "Hole height on the cup",
                4: "length of water flow away from the cup",
          },
          "adjacency_matrix": np.array([
              # Rows and columns correspond to the indices in "variables"
              [0,0,1,0,0],
              [0,0,1,0,0],
              [0,0,0,0,1],
              [0,0,0,0,1],
              [0,0,0,0,0],
          ]),
          "sample_result": """
To fill out the causality adjacency matrix, we need to assess the potential causal relationships between the variables based on the images and scenario. Here are some possible interpretations:

1. **Ball volume** affects **water height in the cup**. The larger the ball, the higher the water level.
2. **Diameter of the bottom of the cup** affects **water height in the cup**. A narrower bottom may cause water to rise higher.
3. **Water height in the cup** affects **length of water flow away from the cup**. Higher water levels can increase flow length due to increased pressure.
4. **Hole height on the cup** affects **length of water flow away from the cup**. Higher holes might result in stronger flows.

The proposed causality adjacency matrix could look like this:

```
[[0, 0, 1, 0, 0],
 [0, 0, 1, 0, 0],
 [0, 0, 0, 0, 1],
 [0, 0, 0, 0, 1],
 [0, 0, 0, 0, 0]]
```

### Explanation:
- **Ball volume (1) causes water height (3)**: The presence of the ball displaces water, raising its level.
- **Diameter of the bottom of the cup (2) causes water height (3)**: The shape of the cup can impact the water height due to displacement being more noticeable in narrow containers.
- **Water height (3) causes length of water flow (5)**: With more water height, there's greater pressure, which can lead to a longer water stream.
- **Hole height (4) causes length of water flow (5)**: The height of the hole affects the pressure and thus the flow's distance.

This matrix assumes no direct causations between other combinations of variables based on typical fluid mechanics.
"""
      },
  
      "Pendulum": {
          # Image illustrating the setup
          "file_name": "Dataset/Real_pendulum_v5_256P/Real_pendulum",
          "variables": {
              0: "Light position",
              1: "Pendulum angle",
              2: "Pendulum length",
              3: "Shadow position",
              4: "Shadow length"
          },
          "adjacency_matrix": np.array([
              # Rows and columns correspond to the indices in "variables"
              [0,0,0,1,1],
              [0,0,0,1,1],
              [0,0,0,1,0],
              [0,0,0,0,1],
              [0,0,0,0,0],
          ])
      },
      
      "V2": {
        # Image illustrating the setup
        "file_name": "Dataset/Hypothetic_v2_linear/Hypothetic_v2_linear",
        "variables": {
          0: "ball's volumn",
          1: "cube's volumn",
          },
        "adjacency_matrix": np.array([
          # Rows and columns correspond to the indices in "variables"
          [0, 1], [0,0]
          ])
      },

      "V2_nonlinear": {
        # Image illustrating the setup
        "file_name": "Dataset/Hypothetic_v2_nonlinear/Hypothetic_v2_nonlinear",
        "variables": {
          0: "ball's volumn",
          1: "cube's volumn",
          },
        "adjacency_matrix": np.array([
          # Rows and columns correspond to the indices in "variables"
          [0, 1], [0,0]
          ])
      },

      "V3_V": {
        "file_name": "Dataset/Hypo_v3_v_structure_256/Hypo_v3_v_structure_256",
        "variables": {
          0: "ball's volumn",
          1: "cuboid's height",
          2: "base area of cone",
          },
        "adjacency_matrix": np.array([
          # Rows and columns correspond to the indices in "variables"
          [0, 0, 1], [0,0,1], [0,0,0]
          ])
        },

      "V3_V_nonlinear": {
        "file_name": "Dataset/Hypothetic_V3_nonlinear_vstructure/Hypothetic_V3_nonlinear_vstructure",
        "variables": {
          0: "ball's volumn",
          1: "cuboid's height",
          2: "base area of cone",
          },
        "adjacency_matrix": np.array([
          # Rows and columns correspond to the indices in "variables"
          [0, 0, 1], [0,0,1], [0,0,0]
          ])
        },  
      
      "V3_F": {
        # Image illustrating the setup
        "file_name": "Dataset/Hypothetic_v3_fully_connected_linear/Hypothetic_v3_fully_connected_linear",
        "variables": {
          0: "ball's volumn",
          1: "cuboid's height",
          2: "base area of cone",
          },
        "adjacency_matrix": np.array([
          # Rows and columns correspond to the indices in "variables"
          [0, 1, 1], [0,0,1], [0,0,0]
        ])},

      "V4_V": {
        "file_name": "Dataset/Hypothetic_v4_linear_v/Hypothetic_v4_linear_v",
        "variables": {
          0: "ball's volumn",
          1: "cuboid's height",
          2: "base area of cuboid",
          3: "base area of cone",
          
        },
        "adjacency_matrix": np.array([
            # Rows and columns correspond to the indices in "variables"
            [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]
        ])
        },

      "V4_V_nonlinear": {
        "file_name": "Dataset/Hypothetic_v4_nonlinear_v/Hypothetic_v4_nonlinear_v",
        "variables": {
          0: "ball's volumn",
          1: "cuboid's height",
          2: "base area of cuboid",
          3: "base area of cone",
          
        },
        "adjacency_matrix": np.array([
            # Rows and columns correspond to the indices in "variables"
            [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]
        ])
        },

      "V4_F": {
        # Image illustrating the setup
        "file_name": "Dataset/Hypothetic_V4_linear_full_connected/Hypothetic_V4_linear_full_connected",
        "variables": {
            0: "ball's volumn",
            1: "cuboid's height",
            2: "base area of cuboid",
            3: "base area of cone",
            
        },
        "adjacency_matrix": np.array([
            # Rows and columns correspond to the indices in "variables"
            [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 1, 0]
        ])
      },
      
      "V5": {
        "file_name": "Dataset/Hypothetic_v5_linear/Hypothetic_v5_linear",
        "variables": {
            0: "ball's volumn",
            1: "cuboid's height",
            2: "base area of cuboid",
            3: "base area of cone",
            4: "cone's height",
        },
        "adjacency_matrix": np.array([
            # Rows and columns correspond to the indices in "variables"
          [0, 0, 1, 0, 0], [0, 0, 1, 0,0], [0, 0, 0, 1,0], 
          [0, 1, 0, 0, 1], [0, 0, 0, 0,0]])
        },

      "V5_nonlinear": {
        "file_name": "Dataset/Hypothetic_V5_nonlinear_1_18/Hypothetic_V5_nonlinear",
        "variables": {
            0: "ball's volumn",
            1: "cuboid's height",
            2: "base area of cuboid",
            3: "base area of cone",
            4: "cone's height",
        },
        "adjacency_matrix": np.array([
            # Rows and columns correspond to the indices in "variables"
          [0, 0, 1, 0, 0], [0, 0, 1, 0,0], [0, 0, 0, 1,0], 
          [0, 1, 0, 0, 1], [0, 0, 0, 0,0]])
        },

      "V5_F": {
        "file_name": "Dataset/Hypothetic_v5_linear_full_connected/Hypothetic_v5_linear_full_connected",
        "variables": {
            0: "ball's volumn",
            1: "cuboid's height",
            2: "base area of cuboid",
            3: "base area of cone",
            4: "cone's height",
        },
        "adjacency_matrix": np.array([
            # Rows and columns correspond to the indices in "variables"
          [0, 1, 1, 0, 1], [0, 0, 1, 0,0], [0, 0, 0, 1,1], 
          [0, 0, 0, 0, 1], [0, 0, 0, 0,0]
          ])
      },

      
      # ![](https://cdn.jsdelivr.net/gh/DishengL/ResearchPics/Yellow%20(570-590%20nm)_20241201_100615.png)
      # "P_reflection": {
      #   "variables": {0: "wave_length", 1: "incident_position", 2: "incident_angle", 3: "reflected_position"},
      #   "adjacency_matrix": np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]])
      # },
      # # ![](https://cdn.jsdelivr.net/gh/DishengL/ResearchPics/1732947800.585619_rendered_image.png)
      # "P_refraction": {
      #   "variables": {0: "wave_length", 1: "incident_position", 2: "incident_angle", 3: "refracted_position"},
      #   "adjacency_matrix": np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]])
      # },
    }

    
  def get_all_scenes(self):
    return self.scenes
  
  def get_scencs_name(self):
    return self.scenes.keys()
  
  def get_scene(self, scene_name):
    # print(self.get_scencs_name())
    print(scene_name)
    return self.scenes[scene_name]