import pybullet as p
import time
import pybullet_data


conid = p.connect(p.SHARED_MEMORY)
if (conid < 0):
  p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setInternalSimFlags(0)
p.resetSimulation()

p.loadURDF("plane.urdf", useMaximalCoordinates=True)
#p.loadURDF(fileName='assets/blue_plate/model.urdf', basePosition=[0,-0.1,0.075], globalScaling=0.75)
#p.loadURDF(fileName='assets/plastic_banana/model.urdf', basePosition=[0,0.1,0.0], globalScaling=0.65)
#p.loadURDF(fileName='assets/plastic_apple/model.urdf', basePosition=[0,0.2,0.0], globalScaling=0.75)
#p.loadURDF(fileName='assets/plastic_plum/model.urdf', basePosition=[0,-0.2,0.0], globalScaling=0.75)
#p.loadURDF(fileName='assets/plate_holder/model.urdf', basePosition=[-0.2,-0.2,0.0], globalScaling=0.75)

p.loadURDF(fileName='assets/cup.urdf', basePosition=[0,-0.2,0.075], globalScaling=0.5)
p.loadURDF(fileName='assets/cup.urdf', basePosition=[0,-0.1,0.075], globalScaling=0.75)
p.loadURDF(fileName='assets/cap.urdf', basePosition=[0.2,-0.2,0.075], globalScaling=0.5)
p.loadURDF(fileName='assets/cap.urdf', basePosition=[0.2,-0.1,0.075], globalScaling=0.75)

size = 0.005
n = 4
for i in range(n):
    for j in range(n):
        for k in range(n):
            #ob = p.loadURDF(body_name='droplet%d'%(i*j*k), fileName="sphere_1cm.urdf", basePosition=[size*(i-n//2), -0.2 + size*(j-n//2), 0.1 + size*(k-n//2)], useMaximalCoordinates=True, globalScaling=0.7)
            ob = p.loadURDF(fileName="sphere_1cm.urdf", basePosition=[size*(i-n//2), -0.2 + size*(j-n//2), 0.1 + size*(k-n//2)], useMaximalCoordinates=True, globalScaling=0.7)
            p.changeVisualShape(ob, -1, rgbaColor=[0.3, 0.7, 1, 0.5])

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.setGravity(0, 0, -10)

p.setRealTimeSimulation(1)

while True:
    continue
