# Extending DDPG to TD3 and Beyond

### Structure 


```
MultiDDPG 
-- {Q, T, Tr}D3 
  -- plots (plots of learning curves)
  -- tmp (model checkpoints)
  -- videos (videos of model in action)
  -- qrd3_torch.py (model code)
  -- testing_qrd3.py (video creation)
  -- training_qrd3.py (model training)
-- shared
  -- buffer.py (memory buffer code)
  -- cuda.py (run to check if cuda is available)
  -- networks.py (basis code for critics)
  -- utils.py (graphing and video code)
```
### Setup 

```bash
python install .
```

### To train

```bash
cd TD3
python training_td3.py
```

### To generate video
```bash
cd TD3
python testing_td3.py
```

For full code explaination see:
https://medium.com/@alexparistimms/extending-ddpg-to-td3-and-beyond-4f5e87d9c79d