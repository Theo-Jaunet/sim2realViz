# Sim2RealViz: Visualizing the Sim2Real Gap in Robot Ego-Pose Estimation


<p align="center">
<img src="https://github.com/Theo-Jaunet/sim2realViz/blob/master/static/assets/images/teaser.jpg" height="450">
 <p align="center">
Using Sim2RealViz, the sim2real gap of Data Augmentation model can be compared agaings other models (e.g. Vanilla or Fine-tuned) and displayed on the real-world environment map along with its performance metrics. In particular, Sim2RealViz shows ① that those models are particulary effective in simulation, but we identified errors in the environment, such as the model failing to regress its position because of a closed-door that was opened in training. Such an error can be selected by instance on the map ② to identify key features extracted by the model either as superimposed on the bird's eye-map ③, or as a first person view ④.
  </p>
</p>


For more information, please refer to the manuscript: 
[Sim2RealViz: Visualizing the Sim2Real Gap in Robot Ego-Pose Estimation](https://arxiv.org/pdf/2109.11801.pdf)

Work by:  Théo Jaunet, Guillaume Bono, Romain Vuillemot, and Christian Wolf



## How to install and run locally

**Step 1:** Clone this repo and install Python dependecies as it follows (you may want to use a vitural environment for that).

  ```
  pip install -r requirements.txt
  ```
  
  
  
**Step 2:** For a direct interaction with models and simulation, this tool requieres both (Habitat-sim + habitat-api) and pytorch

   You can follow installation insctructions here:

   - [habitat](https://github.com/facebookresearch/habitat-sim)
   - [Pytorch](https://pytorch.org/)





**Step 3:** Download the virtual environment data in from [this drive](https://drive.google.com/drive/folders/1NihHdUo0d9lc7g7NvY0-3JzJXUCpEChZ?usp=sharing), and move it to <project_dir>/data/
  
  ```
    mv ~/Downloads/citi.glb data/citi.glb
  ```
  
  
  
  
  
**Step 4:** You can launch the server with the script 'server.py' at the root of this repo.
  
```
python server.py
```

The server should then be accessible at: [http://0.0.0.0:5000](http://0.0.0.0:5000) (it may take a minute or two to launch).
