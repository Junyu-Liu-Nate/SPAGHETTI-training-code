### SPAGHETTI: Editing Implicit Shapes Through Part Aware Generation

![](./assets/readme_resources/chair_int.gif)

The code will be available soon.

#### Author's notes:

For training on shapenet, first, you need to parse the data:
- Download the shapenet data (ShapeNetCore.v2) https://shapenet.org/
- install Watertight Manifold
- Change the paths in constants.py to the location of the ShapeNetCore.v2 and the MANIFOLD build path.
- run make_data.py. You can specify the category name to parse. It will run the watertight script and sample points for training afterward.
- Run train.py for training