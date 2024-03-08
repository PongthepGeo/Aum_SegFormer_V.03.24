#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import re
#-----------------------------------------------------------------------------------------#
from skimage.measure import label, regionprops
from sklearn.preprocessing import StandardScaler
from PIL import Image
#-----------------------------------------------------------------------------------------#
params = {
	'savefig.dpi': 300,  
	'figure.dpi' : 300,
	'axes.labelsize':10,  
	'axes.titlesize':10,
	'axes.titleweight': 'bold',
	'legend.fontsize': 8,
	'xtick.labelsize':8,
	'ytick.labelsize':8,
	'font.family': 'serif',
}
matplotlib.rcParams.update(params)
#-----------------------------------------------------------------------------------------#

def image_segmentation(model, min_value, max_value, min_pixel, max_pixel):
	binary = np.zeros_like(model, dtype='float32') 
	# binary[data_kmean[:, :, 0] == 0.05914314] = 1.0 # hot
	#? transform data into binary: need specific range normal the label target
	for col in range (0, model.shape[0]):
		for row in range (0, model.shape[1]):
			if model[col, row] >= min_value and model[col, row] <= max_value:
				binary[col, row] = 1.
	#? begin image segmentation
	label_out = label(binary, return_num=False)
	for region in regionprops(label_out):
		(min_row, min_col, max_row, max_col) = region.bbox
		if region.area >= min_pixel and region.area <= max_pixel:
			model[min_row:max_row, min_col:max_col] = 100.
#? after computed, we need to remove some pixels that are related to the label 
	for col in range (0, model.shape[0]):
		for row in range (0, model.shape[1]):
			if model[col, row] < min_value and model[col, row] > max_value:
				model[col, row] = 0.
	return model

def loss_history_plot(history_train, history_valid, title_name):
	axis_x = np.linspace(0, len(history_train), len(history_train))
	plt.plot(axis_x, history_train, linestyle='solid',
			 color='red', linewidth=1, marker='o', ms=5, label='train')
	plt.plot(axis_x, history_valid, linestyle='solid',
			 color='blue', linewidth=1, marker='o', ms=5, label='valid')
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.legend(['train', 'valid'])
	plt.title(title_name + ': ' + 'Accuracy', fontweight='bold')
	save_file = "data_out/"
	if not os.path.exists(save_file):
		os.makedirs(save_file)
	plt.savefig(save_file + title_name + ".svg", format="svg",
				bbox_inches="tight", transparent=True, pad_inches=0)
	plt.show()

def custom_sort(file):
	seq_num, x, y = [int(val) for val in re.findall(r"(\d+)_(\d+)_(\d+).png", file)[0]]
	return (seq_num, y, x)

def reconstruct_large_map(input_folder, output_folder, split_size):
	file_list = sorted([f for f in os.listdir(input_folder) if f.endswith(".png")],
					   key=custom_sort)
	large_map_dims = {}
	for file in file_list:
		seq_num, x, y = [int(val) for val in re.findall(r"(\d+)_(\d+)_(\d+).png", file)[0]]
		if seq_num not in large_map_dims:
			large_map_dims[seq_num] = {'max_x': 0, 'max_y': 0}
		large_map_dims[seq_num]['max_x'] = max(large_map_dims[seq_num]['max_x'], x)
		large_map_dims[seq_num]['max_y'] = max(large_map_dims[seq_num]['max_y'], y)
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)
	current_seq_num = None
	large_map = None
	for i, file in enumerate(file_list, start=1):
		seq_num, x, y = [int(val) for val in re.findall(r"(\d+)_(\d+)_(\d+).png", file)[0]]
		if seq_num != current_seq_num:
			if large_map is not None:
				# output_file = os.path.join(output_folder, f"{current_seq_num:03d}_large_map.png")
				output_file = os.path.join(output_folder, f"{current_seq_num:03d}.png")
				large_map_image = Image.fromarray(large_map.astype(np.uint8))
				large_map_image.save(output_file)
			large_map_size = (split_size * (large_map_dims[seq_num]['max_y'] // split_size + 1),
							  split_size * (large_map_dims[seq_num]['max_x'] // split_size + 1), 3)
			large_map = np.zeros(large_map_size, dtype=np.uint8)
			current_seq_num = seq_num
		img = Image.open(os.path.join(input_folder, file)).convert("RGB")
		img = np.array(img)
		large_map[y:y + split_size, x:x + split_size] = img
		print(f"Processing file {i}/{len(file_list)}: {file}")
	if large_map is not None:
		# output_file = os.path.join(output_folder, f"{current_seq_num:03d}_large_map.png")
		output_file = os.path.join(output_folder, f"{current_seq_num:03d}.png")
		large_map_image = Image.fromarray(large_map.astype(np.uint8))
		large_map_image.save(output_file)

def revert_pixels(array, class_colors):
	reverted_array = np.zeros((array.shape[0], array.shape[1], 3), dtype=np.uint8)
	for i in range(array.shape[0]):
		for j in range(array.shape[1]):
			class_id = array[i, j]
			reverted_array[i, j] = class_colors[class_id]
	return reverted_array

def view_converted_inputs(encoded_inputs, class_colors):
	seg_map = encoded_inputs["labels"].numpy()
	reverted_seg_map = revert_pixels(seg_map, class_colors)
	plt.imshow(reverted_seg_map)
	plt.show()

def revert_pixels_pytorch(array, class_colors):
	reverted_array = np.zeros((array.shape[0], array.shape[1], 3), dtype=np.uint8)
	for i in range(array.shape[0]):
		for j in range(array.shape[1]):
			class_id = array[i, j].item()
			reverted_array[i, j] = class_colors[class_id]
	return reverted_array

def split_image_into_patches(image, patch_size=512):
    h, w, _ = image.shape
    patches = []
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            patches.append((i, j, patch))
    return patches

def reconstruct_image(patches, original_shape):
    reconstructed_image = np.zeros(original_shape, dtype=np.uint8)
    for i, j, patch in patches:
        reconstructed_image[i:i+patch.shape[0], j:j+patch.shape[1]] = patch
    return reconstructed_image