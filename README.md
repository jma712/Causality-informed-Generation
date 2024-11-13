# Causality-informed-Generation

## Step 1 Causality-driven Data Generation

### Style
- Hypothetical
- Real-world style

### Size and graph structure

1. Hypothetical example with 3 variables:

    Variable a = the volume of a ball;  
    Variable b = the volume of the cylinder;  
    Variable c = the tilt angle of the rectangular prism.

    The causal graph is:  
    <img width="50%" alt="causal_graph_1" src="1.png">

2. Hypothetical example with 4 variables:  

    Variable a = the volume of a ball;  
    Variable b = the height of a cylinder;  
    Variable c = the distance between the ball and the cylinder;  
    Variable d = the cylinder’s height above the ground.  

    The causal graph is:  
    <img width="100%" alt="causal_graph_2" src="2.png">

3. Hypothetical example with 5 variables:  

    Variable a = the volume of a ball;  
    Variable b = the height of a cylinder;  
    Variable c = the distance between the ball and the cylinder;  
    Variable d = the cylinder’s height above the ground;  
    Variable e = the tilt angle of the cylinder.

    The causal graph is:  
    <img width="75%" alt="causal_graph_3" src="3.png">
   
### Noise  

    In the first hypothetical example, the noise e is the height of the rectangular prism above the ground.  
    In the second hypothetical example, the noise e is the volume of the cylinder.  
    In the third hypothetical example, the noise e is the height of the cylinder above the ground.  
    
### Linear/nonlinear  
    Linear:  
    In the first hypothetical example, b = 2a; c = 3a + 5b + 0.5e.  
    In the second hypothetical example, a = 3.5d; b = 3a; c = 4a + 3b + 9d + 0.7e.  
    In the third hypothetical example, b = 5a; c = 6a + 2b; d = 5c; e = 7.5a + 4.5c + 4d + 0.9e.  
    
### Background  

    Each example offers four environment options:  

    well-lit indoor, well-lit outdoor, dimly-lit indoor, and dimly-lit outdoor.  

### Interventional do()  

    In the first hypothetical example, we can let users to make interventioans do(a = A) and do(b = B).  
    In the second hypothetical example, we can let users to make interventioans do(a = A), do(b = B) and do(c = C).  
    In the third hypothetical example, we can let users to make interventioans do(a = A), do(b = B), do(c = C) and do(d = D).
  
## Step 2 SOTA Baselines

### Methods
1. Causal representation learning / causal discovery from image
- CausalVAE

2. LLM?

### Metrics
1. Causal discovery metrics
2. Case studies of intervention
3. Explanations (for LLM)

## Future direction
- 3D/Video
- Graph editing

## References
