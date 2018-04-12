## Introduction
Simple CNN classifier. CNN trained in Keras and used in C++. 

## Prerequisites
- Keras

## How to run
- Step 1
collect positive images to 'data/pos', negtive images to 'data/neg' with any size.

- Step 2: build dataset
```
python3 build_data.py
```

- Step 3: train
```
python3 logo_classifier.py
```

- Step 4: convert model for C++
```
python3 dump_to_simple_cpp.py -a saved_models/logo_classifier.json -w saved_models/logo_classifier.h5 -o saved_models/logo_classifier.nnet
```

- Step 5: used in C++
```
cd cpp_example
mkdir build
cd build
cmake ..
make
./demo_logo_classifier
```