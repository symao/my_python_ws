python3 build_data.py
python3 logo_classifier.py
python3 dump_to_simple_cpp.py -a saved_models/logo_classifier.json -w saved_models/logo_classifier.h5 -o saved_models/logo_classifier.nnet
cp saved_models/logo_classifier.nnet /home/symao/workspace/easy_vland/