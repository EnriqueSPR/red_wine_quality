	��?���?��?���?!��?���?	w��Q~�@w��Q~�@!w��Q~�@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$��?���?(��y�?A�	�c�?Y�T���N�?*	33333SW@2F
Iterator::ModelΈ����?!�^[w��C@)y�&1��?1:pl�>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�?Ɯ?!t�-�.>@)���Mb�?1�+�c09@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���{�?!c��Vtx1@)��ZӼ�?1�F�zm�%@:Preprocessing2U
Iterator::Model::ParallelMapV2HP�sׂ?!4�6��#@)HP�sׂ?14�6��#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�w��#��?!V���vN@)/n���?1_��<'�"@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�HP�x?!���e�&@)�HP�x?1���e�&@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP�s�r?!4�6��@)HP�s�r?14�6��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapa2U0*��?!
���*�4@)�����g?17����@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 5.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2s7.4 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9w��Q~�@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	(��y�?(��y�?!(��y�?      ��!       "      ��!       *      ��!       2	�	�c�?�	�c�?!�	�c�?:      ��!       B      ��!       J	�T���N�?�T���N�?!�T���N�?R      ��!       Z	�T���N�?�T���N�?!�T���N�?JCPU_ONLYYw��Q~�@b 