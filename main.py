from Algorithm1 import *
from Algorithm2 import *
from memory_profiler import profile
from visualization import *

folder_path = '1'
image_paths = get_image_paths(folder_path)
#print (image_paths)
#image_paths = ["Images/20180109213443.xK7FQt_thumb.jpg", "Images/20180109213527.eVHVKN_thumb.jpg"]  # List of image paths
features = compute_features(image_paths)
graph = construct_graph(features)

T= compute_mst(graph)
#print(T)
triplets = extract_triplets(T)
local_poses = find_local_pose_for_triplets(triplets, features)
print(local_poses)


'''


##visualzation 



G = nx.Graph()
for (img1, img2), weight in graph.items():
    G.add_edge(img1, img2, weight=weight)

# Draw the original graph
pos = nx.circular_layout(G)

plt.figure(figsize=(12, 8))
nx.draw_networkx(G, pos)
plt.title("Original Graph")
plt.show()

# Use the same positions for the MST
plt.figure(figsize=(12, 8))
nx.draw_networkx(T, pos, node_color='green')  # T is your MST
plt.title("Minimum Spanning Tree")
plt.show()

'''