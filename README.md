This repo demonstrates tensorflow based object detection with pre-trained models from Google. 
The file, object_detection_demo.py trys to detect objects in a short clip from James Bond movie, "Spectre"

To run the code, the following setup is ncecessary.

1) install protoc if you have not yet
	a) Linux 
		apt-get install protobuf-compiler

	b) Windows/Mac 
		download from

2) At the root of this repo, run:
	a) Linux
		protoc object_detection/protos/*.proto --python_out=.

	b) Windows/Mac
		'path_to_protc_bin_file' + protoc object_detection/protos/*.proto --python_out=.