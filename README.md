# 1.12.6 修改isaac lab环境生成，提高适应性
	target_h = heights.shape[0] - 2 * border_pixels
	target_w = heights.shape[1] - 2 * border_pixels
	h = min(target_h, z_gen.shape[0])
	w = min(target_w, z_gen.shape[1])

	heights[border_pixels:border_pixels + h, border_pixels:border_pixels + w] = z_gen[:h, :w]

