This repo demonstrates tensorflow based object detection with pre-trained models from Google.
The file, object_detection_demo.py trys to detect objects in a short clip from James Bond movie, "Spectre"

To run the code, the following setup is ncecessary.

1. install protoc if you have not yet
	* Linux
		apt-get install protobuf-compiler

	* Windows/Mac
		download from

2. At the root of this repo, run:
	* Linux
		protoc object_detection/protos/*.proto --python_out=.

	* Windows/Mac
		'path_to_protc_bin_file' + protoc object_detection/protos/*.proto --python_out=.


3. When protoc is set up. Run:
	python object_detection_demo.py
