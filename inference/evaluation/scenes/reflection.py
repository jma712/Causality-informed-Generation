class ground_truth():
    def __init__(self):
      self.variables = {0: "incident_degree", 1: "reflection_degree"}
      self_adjency_matrix = np.array([[0, 1], [0, 0]])
    def get_variables(self):
        return self.variables
    def get_adjency_matrix(self):
        return self_adjency_matrix
      
class OpenAI_ChatGPT_Inference():
  def __init__(self):
      pass
