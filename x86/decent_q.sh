vai_q_tensorflow quantize \
	--input_frozen_graph ./frozen_graph.pb \
	--input_nodes input_1 \
	--input_shapes ?,224,224,3 \
	--output_nodes probs/Softmax \
	--input_fn custom_network_input_fn.calib_input \
	--method 1 \
	--gpu 0 \
	--calib_iter 10 \
	--output_dir ./quantize_results
