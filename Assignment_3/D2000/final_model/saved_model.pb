??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.3.12v2.3.0-54-gfcc4b966f18??
~
dense_3996/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_namedense_3996/kernel
w
%dense_3996/kernel/Read/ReadVariableOpReadVariableOpdense_3996/kernel*
_output_shapes

:*
dtype0
v
dense_3996/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_3996/bias
o
#dense_3996/bias/Read/ReadVariableOpReadVariableOpdense_3996/bias*
_output_shapes
:*
dtype0
~
dense_3997/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_namedense_3997/kernel
w
%dense_3997/kernel/Read/ReadVariableOpReadVariableOpdense_3997/kernel*
_output_shapes

:*
dtype0
v
dense_3997/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_3997/bias
o
#dense_3997/bias/Read/ReadVariableOpReadVariableOpdense_3997/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
#_self_saveable_object_factories
	optimizer

signatures
trainable_variables
regularization_losses
	variables
		keras_api
?


kernel
bias
#_self_saveable_object_factories
trainable_variables
regularization_losses
	variables
	keras_api
?

kernel
bias
#_self_saveable_object_factories
trainable_variables
regularization_losses
	variables
	keras_api
 
 
 


0
1
2
3
 


0
1
2
3
?
trainable_variables
regularization_losses

layers
non_trainable_variables
metrics
layer_regularization_losses
layer_metrics
	variables
][
VARIABLE_VALUEdense_3996/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_3996/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 


0
1
 


0
1
?
trainable_variables
regularization_losses

layers
metrics
non_trainable_variables
 layer_regularization_losses
!layer_metrics
	variables
][
VARIABLE_VALUEdense_3997/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_3997/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 

0
1
?
trainable_variables
regularization_losses

"layers
#metrics
$non_trainable_variables
%layer_regularization_losses
&layer_metrics
	variables

0
1
 

'0
(1
 
 
 
 
 
 
 
 
 
 
 
 
4
	)total
	*count
+	variables
,	keras_api
D
	-total
	.count
/
_fn_kwargs
0	variables
1	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

)0
*1

+	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

-0
.1

0	variables
?
 serving_default_dense_3996_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCall serving_default_dense_3996_inputdense_3996/kerneldense_3996/biasdense_3997/kerneldense_3997/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *.
f)R'
%__inference_signature_wrapper_2324874
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%dense_3996/kernel/Read/ReadVariableOp#dense_3996/bias/Read/ReadVariableOp%dense_3997/kernel/Read/ReadVariableOp#dense_3997/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__traced_save_2325023
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_3996/kerneldense_3996/biasdense_3997/kerneldense_3997/biastotalcounttotal_1count_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__traced_restore_2325057??
?
?
L__inference_sequential_1998_layer_call_and_return_conditional_losses_2324790
dense_3996_input
dense_3996_2324757
dense_3996_2324759
dense_3997_2324784
dense_3997_2324786
identity??"dense_3996/StatefulPartitionedCall?"dense_3997/StatefulPartitionedCall?
"dense_3996/StatefulPartitionedCallStatefulPartitionedCalldense_3996_inputdense_3996_2324757dense_3996_2324759*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_3996_layer_call_and_return_conditional_losses_23247462$
"dense_3996/StatefulPartitionedCall?
"dense_3997/StatefulPartitionedCallStatefulPartitionedCall+dense_3996/StatefulPartitionedCall:output:0dense_3997_2324784dense_3997_2324786*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_3997_layer_call_and_return_conditional_losses_23247732$
"dense_3997/StatefulPartitionedCall?
IdentityIdentity+dense_3997/StatefulPartitionedCall:output:0#^dense_3996/StatefulPartitionedCall#^dense_3997/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2H
"dense_3996/StatefulPartitionedCall"dense_3996/StatefulPartitionedCall2H
"dense_3997/StatefulPartitionedCall"dense_3997/StatefulPartitionedCall:Y U
'
_output_shapes
:?????????
*
_user_specified_namedense_3996_input
?
?
%__inference_signature_wrapper_2324874
dense_3996_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_3996_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__wrapped_model_23247312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
'
_output_shapes
:?????????
*
_user_specified_namedense_3996_input
?
?
 __inference__traced_save_2325023
file_prefix0
,savev2_dense_3996_kernel_read_readvariableop.
*savev2_dense_3996_bias_read_readvariableop0
,savev2_dense_3997_kernel_read_readvariableop.
*savev2_dense_3997_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_bacceb5b1f404f6aa856dc64fee8494b/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*?
value?B?	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_dense_3996_kernel_read_readvariableop*savev2_dense_3996_bias_read_readvariableop,savev2_dense_3997_kernel_read_readvariableop*savev2_dense_3997_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes.
,: ::::: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
?
?
G__inference_dense_3997_layer_call_and_return_conditional_losses_2324773

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_dense_3996_layer_call_and_return_conditional_losses_2324746

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Elue
IdentityIdentityElu:activations:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_dense_3997_layer_call_and_return_conditional_losses_2324967

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
L__inference_sequential_1998_layer_call_and_return_conditional_losses_2324804
dense_3996_input
dense_3996_2324793
dense_3996_2324795
dense_3997_2324798
dense_3997_2324800
identity??"dense_3996/StatefulPartitionedCall?"dense_3997/StatefulPartitionedCall?
"dense_3996/StatefulPartitionedCallStatefulPartitionedCalldense_3996_inputdense_3996_2324793dense_3996_2324795*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_3996_layer_call_and_return_conditional_losses_23247462$
"dense_3996/StatefulPartitionedCall?
"dense_3997/StatefulPartitionedCallStatefulPartitionedCall+dense_3996/StatefulPartitionedCall:output:0dense_3997_2324798dense_3997_2324800*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_3997_layer_call_and_return_conditional_losses_23247732$
"dense_3997/StatefulPartitionedCall?
IdentityIdentity+dense_3997/StatefulPartitionedCall:output:0#^dense_3996/StatefulPartitionedCall#^dense_3997/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2H
"dense_3996/StatefulPartitionedCall"dense_3996/StatefulPartitionedCall2H
"dense_3997/StatefulPartitionedCall"dense_3997/StatefulPartitionedCall:Y U
'
_output_shapes
:?????????
*
_user_specified_namedense_3996_input
?
?
G__inference_dense_3996_layer_call_and_return_conditional_losses_2324947

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Elue
IdentityIdentityElu:activations:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
L__inference_sequential_1998_layer_call_and_return_conditional_losses_2324910

inputs-
)dense_3996_matmul_readvariableop_resource.
*dense_3996_biasadd_readvariableop_resource-
)dense_3997_matmul_readvariableop_resource.
*dense_3997_biasadd_readvariableop_resource
identity??
 dense_3996/MatMul/ReadVariableOpReadVariableOp)dense_3996_matmul_readvariableop_resource*
_output_shapes

:*
dtype02"
 dense_3996/MatMul/ReadVariableOp?
dense_3996/MatMulMatMulinputs(dense_3996/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3996/MatMul?
!dense_3996/BiasAdd/ReadVariableOpReadVariableOp*dense_3996_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!dense_3996/BiasAdd/ReadVariableOp?
dense_3996/BiasAddBiasAdddense_3996/MatMul:product:0)dense_3996/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3996/BiasAddv
dense_3996/EluEludense_3996/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3996/Elu?
 dense_3997/MatMul/ReadVariableOpReadVariableOp)dense_3997_matmul_readvariableop_resource*
_output_shapes

:*
dtype02"
 dense_3997/MatMul/ReadVariableOp?
dense_3997/MatMulMatMuldense_3996/Elu:activations:0(dense_3997/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3997/MatMul?
!dense_3997/BiasAdd/ReadVariableOpReadVariableOp*dense_3997_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!dense_3997/BiasAdd/ReadVariableOp?
dense_3997/BiasAddBiasAdddense_3997/MatMul:product:0)dense_3997/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3997/BiasAdd?
dense_3997/SoftmaxSoftmaxdense_3997/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3997/Softmaxp
IdentityIdentitydense_3997/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_dense_3996_layer_call_fn_2324956

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_3996_layer_call_and_return_conditional_losses_23247462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
"__inference__wrapped_model_2324731
dense_3996_input=
9sequential_1998_dense_3996_matmul_readvariableop_resource>
:sequential_1998_dense_3996_biasadd_readvariableop_resource=
9sequential_1998_dense_3997_matmul_readvariableop_resource>
:sequential_1998_dense_3997_biasadd_readvariableop_resource
identity??
0sequential_1998/dense_3996/MatMul/ReadVariableOpReadVariableOp9sequential_1998_dense_3996_matmul_readvariableop_resource*
_output_shapes

:*
dtype022
0sequential_1998/dense_3996/MatMul/ReadVariableOp?
!sequential_1998/dense_3996/MatMulMatMuldense_3996_input8sequential_1998/dense_3996/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!sequential_1998/dense_3996/MatMul?
1sequential_1998/dense_3996/BiasAdd/ReadVariableOpReadVariableOp:sequential_1998_dense_3996_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_1998/dense_3996/BiasAdd/ReadVariableOp?
"sequential_1998/dense_3996/BiasAddBiasAdd+sequential_1998/dense_3996/MatMul:product:09sequential_1998/dense_3996/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2$
"sequential_1998/dense_3996/BiasAdd?
sequential_1998/dense_3996/EluElu+sequential_1998/dense_3996/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2 
sequential_1998/dense_3996/Elu?
0sequential_1998/dense_3997/MatMul/ReadVariableOpReadVariableOp9sequential_1998_dense_3997_matmul_readvariableop_resource*
_output_shapes

:*
dtype022
0sequential_1998/dense_3997/MatMul/ReadVariableOp?
!sequential_1998/dense_3997/MatMulMatMul,sequential_1998/dense_3996/Elu:activations:08sequential_1998/dense_3997/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!sequential_1998/dense_3997/MatMul?
1sequential_1998/dense_3997/BiasAdd/ReadVariableOpReadVariableOp:sequential_1998_dense_3997_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_1998/dense_3997/BiasAdd/ReadVariableOp?
"sequential_1998/dense_3997/BiasAddBiasAdd+sequential_1998/dense_3997/MatMul:product:09sequential_1998/dense_3997/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2$
"sequential_1998/dense_3997/BiasAdd?
"sequential_1998/dense_3997/SoftmaxSoftmax+sequential_1998/dense_3997/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2$
"sequential_1998/dense_3997/Softmax?
IdentityIdentity,sequential_1998/dense_3997/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::::Y U
'
_output_shapes
:?????????
*
_user_specified_namedense_3996_input
?%
?
#__inference__traced_restore_2325057
file_prefix&
"assignvariableop_dense_3996_kernel&
"assignvariableop_1_dense_3996_bias(
$assignvariableop_2_dense_3997_kernel&
"assignvariableop_3_dense_3997_bias
assignvariableop_4_total
assignvariableop_5_count
assignvariableop_6_total_1
assignvariableop_7_count_1

identity_9??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*?
value?B?	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp"assignvariableop_dense_3996_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp"assignvariableop_1_dense_3996_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp$assignvariableop_2_dense_3997_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_3997_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_totalIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_countIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_total_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_count_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8?

Identity_9IdentityIdentity_8:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*
T0*
_output_shapes
: 2

Identity_9"!

identity_9Identity_9:output:0*5
_input_shapes$
": ::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
,__inference_dense_3997_layer_call_fn_2324976

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_3997_layer_call_and_return_conditional_losses_23247732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
L__inference_sequential_1998_layer_call_and_return_conditional_losses_2324892

inputs-
)dense_3996_matmul_readvariableop_resource.
*dense_3996_biasadd_readvariableop_resource-
)dense_3997_matmul_readvariableop_resource.
*dense_3997_biasadd_readvariableop_resource
identity??
 dense_3996/MatMul/ReadVariableOpReadVariableOp)dense_3996_matmul_readvariableop_resource*
_output_shapes

:*
dtype02"
 dense_3996/MatMul/ReadVariableOp?
dense_3996/MatMulMatMulinputs(dense_3996/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3996/MatMul?
!dense_3996/BiasAdd/ReadVariableOpReadVariableOp*dense_3996_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!dense_3996/BiasAdd/ReadVariableOp?
dense_3996/BiasAddBiasAdddense_3996/MatMul:product:0)dense_3996/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3996/BiasAddv
dense_3996/EluEludense_3996/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3996/Elu?
 dense_3997/MatMul/ReadVariableOpReadVariableOp)dense_3997_matmul_readvariableop_resource*
_output_shapes

:*
dtype02"
 dense_3997/MatMul/ReadVariableOp?
dense_3997/MatMulMatMuldense_3996/Elu:activations:0(dense_3997/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3997/MatMul?
!dense_3997/BiasAdd/ReadVariableOpReadVariableOp*dense_3997_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!dense_3997/BiasAdd/ReadVariableOp?
dense_3997/BiasAddBiasAdddense_3997/MatMul:product:0)dense_3997/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3997/BiasAdd?
dense_3997/SoftmaxSoftmaxdense_3997/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3997/Softmaxp
IdentityIdentitydense_3997/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
L__inference_sequential_1998_layer_call_and_return_conditional_losses_2324821

inputs
dense_3996_2324810
dense_3996_2324812
dense_3997_2324815
dense_3997_2324817
identity??"dense_3996/StatefulPartitionedCall?"dense_3997/StatefulPartitionedCall?
"dense_3996/StatefulPartitionedCallStatefulPartitionedCallinputsdense_3996_2324810dense_3996_2324812*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_3996_layer_call_and_return_conditional_losses_23247462$
"dense_3996/StatefulPartitionedCall?
"dense_3997/StatefulPartitionedCallStatefulPartitionedCall+dense_3996/StatefulPartitionedCall:output:0dense_3997_2324815dense_3997_2324817*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_3997_layer_call_and_return_conditional_losses_23247732$
"dense_3997/StatefulPartitionedCall?
IdentityIdentity+dense_3997/StatefulPartitionedCall:output:0#^dense_3996/StatefulPartitionedCall#^dense_3997/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2H
"dense_3996/StatefulPartitionedCall"dense_3996/StatefulPartitionedCall2H
"dense_3997/StatefulPartitionedCall"dense_3997/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
1__inference_sequential_1998_layer_call_fn_2324859
dense_3996_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_3996_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_sequential_1998_layer_call_and_return_conditional_losses_23248482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
'
_output_shapes
:?????????
*
_user_specified_namedense_3996_input
?
?
L__inference_sequential_1998_layer_call_and_return_conditional_losses_2324848

inputs
dense_3996_2324837
dense_3996_2324839
dense_3997_2324842
dense_3997_2324844
identity??"dense_3996/StatefulPartitionedCall?"dense_3997/StatefulPartitionedCall?
"dense_3996/StatefulPartitionedCallStatefulPartitionedCallinputsdense_3996_2324837dense_3996_2324839*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_3996_layer_call_and_return_conditional_losses_23247462$
"dense_3996/StatefulPartitionedCall?
"dense_3997/StatefulPartitionedCallStatefulPartitionedCall+dense_3996/StatefulPartitionedCall:output:0dense_3997_2324842dense_3997_2324844*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dense_3997_layer_call_and_return_conditional_losses_23247732$
"dense_3997/StatefulPartitionedCall?
IdentityIdentity+dense_3997/StatefulPartitionedCall:output:0#^dense_3996/StatefulPartitionedCall#^dense_3997/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2H
"dense_3996/StatefulPartitionedCall"dense_3996/StatefulPartitionedCall2H
"dense_3997/StatefulPartitionedCall"dense_3997/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
1__inference_sequential_1998_layer_call_fn_2324832
dense_3996_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_3996_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_sequential_1998_layer_call_and_return_conditional_losses_23248212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
'
_output_shapes
:?????????
*
_user_specified_namedense_3996_input
?
?
1__inference_sequential_1998_layer_call_fn_2324936

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_sequential_1998_layer_call_and_return_conditional_losses_23248482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
1__inference_sequential_1998_layer_call_fn_2324923

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_sequential_1998_layer_call_and_return_conditional_losses_23248212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
M
dense_3996_input9
"serving_default_dense_3996_input:0?????????>

dense_39970
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?f
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
#_self_saveable_object_factories
	optimizer

signatures
trainable_variables
regularization_losses
	variables
		keras_api
2__call__
3_default_save_signature
*4&call_and_return_all_conditional_losses"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_1998", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_1998", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_3996_input"}}, {"class_name": "Dense", "config": {"name": "dense_3996", "trainable": true, "dtype": "float32", "units": 7, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_3997", "trainable": true, "dtype": "float32", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1998", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_3996_input"}}, {"class_name": "Dense", "config": {"name": "dense_3996", "trainable": true, "dtype": "float32", "units": 7, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_3997", "trainable": true, "dtype": "float32", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": {"class_name": "CategoricalCrossentropy", "config": {"reduction": "auto", "name": "categorical_crossentropy", "from_logits": false, "label_smoothing": 0}}, "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?


kernel
bias
#_self_saveable_object_factories
trainable_variables
regularization_losses
	variables
	keras_api
5__call__
*6&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_3996", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3996", "trainable": true, "dtype": "float32", "units": 7, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}
?

kernel
bias
#_self_saveable_object_factories
trainable_variables
regularization_losses
	variables
	keras_api
7__call__
*8&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_3997", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3997", "trainable": true, "dtype": "float32", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 7}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7]}}
 "
trackable_dict_wrapper
"
	optimizer
,
9serving_default"
signature_map
<

0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
<

0
1
2
3"
trackable_list_wrapper
?
trainable_variables
regularization_losses

layers
non_trainable_variables
metrics
layer_regularization_losses
layer_metrics
	variables
2__call__
3_default_save_signature
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
#:!2dense_3996/kernel
:2dense_3996/bias
 "
trackable_dict_wrapper
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
?
trainable_variables
regularization_losses

layers
metrics
non_trainable_variables
 layer_regularization_losses
!layer_metrics
	variables
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
#:!2dense_3997/kernel
:2dense_3997/bias
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
trainable_variables
regularization_losses

"layers
#metrics
$non_trainable_variables
%layer_regularization_losses
&layer_metrics
	variables
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
	)total
	*count
+	variables
,	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	-total
	.count
/
_fn_kwargs
0	variables
1	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
.
)0
*1"
trackable_list_wrapper
-
+	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
-0
.1"
trackable_list_wrapper
-
0	variables"
_generic_user_object
?2?
1__inference_sequential_1998_layer_call_fn_2324832
1__inference_sequential_1998_layer_call_fn_2324859
1__inference_sequential_1998_layer_call_fn_2324923
1__inference_sequential_1998_layer_call_fn_2324936?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
"__inference__wrapped_model_2324731?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? */?,
*?'
dense_3996_input?????????
?2?
L__inference_sequential_1998_layer_call_and_return_conditional_losses_2324910
L__inference_sequential_1998_layer_call_and_return_conditional_losses_2324892
L__inference_sequential_1998_layer_call_and_return_conditional_losses_2324790
L__inference_sequential_1998_layer_call_and_return_conditional_losses_2324804?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_dense_3996_layer_call_fn_2324956?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dense_3996_layer_call_and_return_conditional_losses_2324947?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dense_3997_layer_call_fn_2324976?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dense_3997_layer_call_and_return_conditional_losses_2324967?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
=B;
%__inference_signature_wrapper_2324874dense_3996_input?
"__inference__wrapped_model_2324731z
9?6
/?,
*?'
dense_3996_input?????????
? "7?4
2

dense_3997$?!

dense_3997??????????
G__inference_dense_3996_layer_call_and_return_conditional_losses_2324947\
/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? 
,__inference_dense_3996_layer_call_fn_2324956O
/?,
%?"
 ?
inputs?????????
? "???????????
G__inference_dense_3997_layer_call_and_return_conditional_losses_2324967\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? 
,__inference_dense_3997_layer_call_fn_2324976O/?,
%?"
 ?
inputs?????????
? "???????????
L__inference_sequential_1998_layer_call_and_return_conditional_losses_2324790p
A?>
7?4
*?'
dense_3996_input?????????
p

 
? "%?"
?
0?????????
? ?
L__inference_sequential_1998_layer_call_and_return_conditional_losses_2324804p
A?>
7?4
*?'
dense_3996_input?????????
p 

 
? "%?"
?
0?????????
? ?
L__inference_sequential_1998_layer_call_and_return_conditional_losses_2324892f
7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
L__inference_sequential_1998_layer_call_and_return_conditional_losses_2324910f
7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
1__inference_sequential_1998_layer_call_fn_2324832c
A?>
7?4
*?'
dense_3996_input?????????
p

 
? "???????????
1__inference_sequential_1998_layer_call_fn_2324859c
A?>
7?4
*?'
dense_3996_input?????????
p 

 
? "???????????
1__inference_sequential_1998_layer_call_fn_2324923Y
7?4
-?*
 ?
inputs?????????
p

 
? "???????????
1__inference_sequential_1998_layer_call_fn_2324936Y
7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
%__inference_signature_wrapper_2324874?
M?J
? 
C?@
>
dense_3996_input*?'
dense_3996_input?????????"7?4
2

dense_3997$?!

dense_3997?????????