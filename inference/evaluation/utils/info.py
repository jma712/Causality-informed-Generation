import numpy as np

class scene:
  def __init__(self,):
    self.scenes = {
      "reflection": {
        #![](https://cdn.jsdelivr.net/gh/DishengL/ResearchPics/render_output_Blank_Reflection_circle_320x320.png)
        "variables": {0: "incident_degree", 1: "reflection_degree"},
        "adjacency_matrix": np.array([[0, 1], [0, 0]])
      },
      "spring": {
        # ![](https://cdn.jsdelivr.net/gh/DishengL/ResearchPics/render_output_Blank_Spring.png)
        "variables": {0: "spring_constant", 1: "weight", 2: "defomation"},
        "adjacency_matrix": np.array([[0,0,1],[0,0,1],[0,0,0]])
      },
      "Seesaw": {
        # ![](https://cdn.jsdelivr.net/gh/DishengL/ResearchPics/20241120_150057.png)
        "variables_4": {0: "seesaw_left_arm", 1: "left_weight", 2: "seesaw_right_arm", 3: "right_weight", 4: "seesaw_torque"},
        "variables_2": {0: "left_force", 1: "right_force", 2: "seesaw_torque"},
        "adjacency_matrix_2": np.array([[0,0,1],[0,0,1],[0,0,0]]),
        "adjacency_matrix_4": np.array([[0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1],
                                        [0, 0, 0, 0, 0]])
      },
      "Magnets": {
        # ![](https://cdn.jsdelivr.net/gh/DishengL/ResearchPics/20241123_161201_Over_3D_256p.png)
        "variables": {0: "neddle_position", 1: "magnetic_bar_direction", 2: "neddle_direction"},
        "adjacency_matrix": np.array([[0, 0 , 1], [0, 0, 1], [0, 0, 0]])
      },
      # ![](https://cdn.jsdelivr.net/gh/DishengL/ResearchPics/Yellow%20(570-590%20nm)_20241201_100615.png)
      "P_reflection": {
        "variables": {0: "wave_length", 1: "incident_position", 2: "incident_angle", 3: "reflected_position"},
        "adjacency_matrix": np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]])
      },
      # ![](https://cdn.jsdelivr.net/gh/DishengL/ResearchPics/1732947800.585619_rendered_image.png)
      "P_refraction": {
        "variables": {0: "wave_length", 1: "incident_position", 2: "incident_angle", 3: "refracted_position"},
        "adjacency_matrix": np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]])
      },
      "3_V": {
        "variables": {0: "v_ball", 1: "v_cylinder", 2: "angle"},
        "adjacency_matrix": np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
      },
      "4_V": {
        "variables": {0: "v_ball", 1: "h_cylinder", 2: "d_ball_cylinder", 3: "cylinder_h_above"},
        "adjacency_matrix": np.array([[0,1,1,0],[0,0,1,0],[0,0,0,0],[1,0,0,1]])
      },
      "5_V": {
        "variables": {0: "v_ball", 1: "h_cylinder", 2: "d_ball_cylinder", 3: "cylinder_h_above", 4: "angle"},
        "adjacency_matrix": np.array([[0,1,1,0,1],[0,0,1,0,0],[0,0,0,1,1],[0,0,0,0,1],[0,0,0,0,0]])
      },
    }
    
  def get_all_scenes(self):
    return self.scenes
  
  def get_scencs_name(self):
    return self.scenes.keys()
  
  def get_scene(self, scene_name):
    return self.scenes[scene_name]