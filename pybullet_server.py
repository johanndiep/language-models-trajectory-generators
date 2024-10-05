import socket
import pybullet as p
import pybullet_data

# Setup socket communication
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # incase the socket is not closed properly, this will allow to reuse the same port
server_socket.bind(('0.0.0.0', 6000))
server_socket.listen(1)
client_socket, addr = server_socket.accept()

# Setup PyBullet in DIRECT mode
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
robotId = p.loadURDF("r2d2.urdf", [0, 0, 1])

for _ in range(1000):
    p.stepSimulation()
    pos, orn = p.getBasePositionAndOrientation(robotId)

    # Send robot position to client
    data = f"{pos[0]},{pos[1]},{pos[2]},{orn[0]},{orn[1]},{orn[2]},{orn[3]}\n"
    client_socket.sendall(data.encode())

client_socket.close()
server_socket.close()
p.disconnect()
