эЌ
“≤
:
Add
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
B
Equal
x"T
y"T
z
"
Ttype:
2	
Р
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
Н
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
ц
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ
9
VarIsInitializedOp
resource
is_initialized
И
s

VariableV2
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И"serve*1.12.02
b'unknown'8эЧ

global_step/Initializer/zerosConst*
value	B	 R *
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
k
global_step
VariableV2*
_class
loc:@global_step*
shape: *
dtype0	*
_output_shapes
: 
Й
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
T0	*
_class
loc:@global_step*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
_output_shapes
: *
T0	*
_class
loc:@global_step
f
PlaceholderPlaceholder*
dtype0*#
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
h
Placeholder_1Placeholder*
dtype0*#
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
h
Placeholder_2Placeholder*
shape:€€€€€€€€€*
dtype0*#
_output_shapes
:€€€€€€€€€
h
Placeholder_3Placeholder*
dtype0*#
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
h
Placeholder_4Placeholder*
dtype0*#
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
h
Placeholder_5Placeholder*
shape:€€€€€€€€€*
dtype0*#
_output_shapes
:€€€€€€€€€
h
Placeholder_6Placeholder*
shape:€€€€€€€€€*
dtype0*#
_output_shapes
:€€€€€€€€€
h
Placeholder_7Placeholder*
dtype0*#
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
h
Placeholder_8Placeholder*
dtype0*#
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
h
Placeholder_9Placeholder*
dtype0*#
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
i
Placeholder_10Placeholder*
dtype0*#
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
М
Adnn/input_from_feature_columns/input_layer/alcohol/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
€€€€€€€€€
–
=dnn/input_from_feature_columns/input_layer/alcohol/ExpandDims
ExpandDimsPlaceholder_10Adnn/input_from_feature_columns/input_layer/alcohol/ExpandDims/dim*'
_output_shapes
:€€€€€€€€€*
T0
•
8dnn/input_from_feature_columns/input_layer/alcohol/ShapeShape=dnn/input_from_feature_columns/input_layer/alcohol/ExpandDims*
T0*
_output_shapes
:
Р
Fdnn/input_from_feature_columns/input_layer/alcohol/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Т
Hdnn/input_from_feature_columns/input_layer/alcohol/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Т
Hdnn/input_from_feature_columns/input_layer/alcohol/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
ђ
@dnn/input_from_feature_columns/input_layer/alcohol/strided_sliceStridedSlice8dnn/input_from_feature_columns/input_layer/alcohol/ShapeFdnn/input_from_feature_columns/input_layer/alcohol/strided_slice/stackHdnn/input_from_feature_columns/input_layer/alcohol/strided_slice/stack_1Hdnn/input_from_feature_columns/input_layer/alcohol/strided_slice/stack_2*
_output_shapes
: *
T0*
Index0*
shrink_axis_mask
Д
Bdnn/input_from_feature_columns/input_layer/alcohol/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
ь
@dnn/input_from_feature_columns/input_layer/alcohol/Reshape/shapePack@dnn/input_from_feature_columns/input_layer/alcohol/strided_sliceBdnn/input_from_feature_columns/input_layer/alcohol/Reshape/shape/1*
N*
_output_shapes
:*
T0
ш
:dnn/input_from_feature_columns/input_layer/alcohol/ReshapeReshape=dnn/input_from_feature_columns/input_layer/alcohol/ExpandDims@dnn/input_from_feature_columns/input_layer/alcohol/Reshape/shape*
T0*'
_output_shapes
:€€€€€€€€€
О
Cdnn/input_from_feature_columns/input_layer/chlorides/ExpandDims/dimConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
”
?dnn/input_from_feature_columns/input_layer/chlorides/ExpandDims
ExpandDimsPlaceholder_4Cdnn/input_from_feature_columns/input_layer/chlorides/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€
©
:dnn/input_from_feature_columns/input_layer/chlorides/ShapeShape?dnn/input_from_feature_columns/input_layer/chlorides/ExpandDims*
_output_shapes
:*
T0
Т
Hdnn/input_from_feature_columns/input_layer/chlorides/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Ф
Jdnn/input_from_feature_columns/input_layer/chlorides/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ф
Jdnn/input_from_feature_columns/input_layer/chlorides/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
ґ
Bdnn/input_from_feature_columns/input_layer/chlorides/strided_sliceStridedSlice:dnn/input_from_feature_columns/input_layer/chlorides/ShapeHdnn/input_from_feature_columns/input_layer/chlorides/strided_slice/stackJdnn/input_from_feature_columns/input_layer/chlorides/strided_slice/stack_1Jdnn/input_from_feature_columns/input_layer/chlorides/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: 
Ж
Ddnn/input_from_feature_columns/input_layer/chlorides/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
В
Bdnn/input_from_feature_columns/input_layer/chlorides/Reshape/shapePackBdnn/input_from_feature_columns/input_layer/chlorides/strided_sliceDdnn/input_from_feature_columns/input_layer/chlorides/Reshape/shape/1*
N*
_output_shapes
:*
T0
ю
<dnn/input_from_feature_columns/input_layer/chlorides/ReshapeReshape?dnn/input_from_feature_columns/input_layer/chlorides/ExpandDimsBdnn/input_from_feature_columns/input_layer/chlorides/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0
Р
Ednn/input_from_feature_columns/input_layer/citric_acid/ExpandDims/dimConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
„
Adnn/input_from_feature_columns/input_layer/citric_acid/ExpandDims
ExpandDimsPlaceholder_2Ednn/input_from_feature_columns/input_layer/citric_acid/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€
≠
<dnn/input_from_feature_columns/input_layer/citric_acid/ShapeShapeAdnn/input_from_feature_columns/input_layer/citric_acid/ExpandDims*
T0*
_output_shapes
:
Ф
Jdnn/input_from_feature_columns/input_layer/citric_acid/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Ц
Ldnn/input_from_feature_columns/input_layer/citric_acid/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
Ц
Ldnn/input_from_feature_columns/input_layer/citric_acid/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ј
Ddnn/input_from_feature_columns/input_layer/citric_acid/strided_sliceStridedSlice<dnn/input_from_feature_columns/input_layer/citric_acid/ShapeJdnn/input_from_feature_columns/input_layer/citric_acid/strided_slice/stackLdnn/input_from_feature_columns/input_layer/citric_acid/strided_slice/stack_1Ldnn/input_from_feature_columns/input_layer/citric_acid/strided_slice/stack_2*
_output_shapes
: *
T0*
Index0*
shrink_axis_mask
И
Fdnn/input_from_feature_columns/input_layer/citric_acid/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
И
Ddnn/input_from_feature_columns/input_layer/citric_acid/Reshape/shapePackDdnn/input_from_feature_columns/input_layer/citric_acid/strided_sliceFdnn/input_from_feature_columns/input_layer/citric_acid/Reshape/shape/1*
T0*
N*
_output_shapes
:
Д
>dnn/input_from_feature_columns/input_layer/citric_acid/ReshapeReshapeAdnn/input_from_feature_columns/input_layer/citric_acid/ExpandDimsDdnn/input_from_feature_columns/input_layer/citric_acid/Reshape/shape*
T0*'
_output_shapes
:€€€€€€€€€
М
Adnn/input_from_feature_columns/input_layer/density/ExpandDims/dimConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
ѕ
=dnn/input_from_feature_columns/input_layer/density/ExpandDims
ExpandDimsPlaceholder_7Adnn/input_from_feature_columns/input_layer/density/ExpandDims/dim*'
_output_shapes
:€€€€€€€€€*
T0
•
8dnn/input_from_feature_columns/input_layer/density/ShapeShape=dnn/input_from_feature_columns/input_layer/density/ExpandDims*
T0*
_output_shapes
:
Р
Fdnn/input_from_feature_columns/input_layer/density/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Т
Hdnn/input_from_feature_columns/input_layer/density/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
Т
Hdnn/input_from_feature_columns/input_layer/density/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ђ
@dnn/input_from_feature_columns/input_layer/density/strided_sliceStridedSlice8dnn/input_from_feature_columns/input_layer/density/ShapeFdnn/input_from_feature_columns/input_layer/density/strided_slice/stackHdnn/input_from_feature_columns/input_layer/density/strided_slice/stack_1Hdnn/input_from_feature_columns/input_layer/density/strided_slice/stack_2*
_output_shapes
: *
Index0*
T0*
shrink_axis_mask
Д
Bdnn/input_from_feature_columns/input_layer/density/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
ь
@dnn/input_from_feature_columns/input_layer/density/Reshape/shapePack@dnn/input_from_feature_columns/input_layer/density/strided_sliceBdnn/input_from_feature_columns/input_layer/density/Reshape/shape/1*
T0*
N*
_output_shapes
:
ш
:dnn/input_from_feature_columns/input_layer/density/ReshapeReshape=dnn/input_from_feature_columns/input_layer/density/ExpandDims@dnn/input_from_feature_columns/input_layer/density/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0
Т
Gdnn/input_from_feature_columns/input_layer/fixed_acidity/ExpandDims/dimConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
ў
Cdnn/input_from_feature_columns/input_layer/fixed_acidity/ExpandDims
ExpandDimsPlaceholderGdnn/input_from_feature_columns/input_layer/fixed_acidity/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€
±
>dnn/input_from_feature_columns/input_layer/fixed_acidity/ShapeShapeCdnn/input_from_feature_columns/input_layer/fixed_acidity/ExpandDims*
_output_shapes
:*
T0
Ц
Ldnn/input_from_feature_columns/input_layer/fixed_acidity/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Ш
Ndnn/input_from_feature_columns/input_layer/fixed_acidity/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ш
Ndnn/input_from_feature_columns/input_layer/fixed_acidity/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
 
Fdnn/input_from_feature_columns/input_layer/fixed_acidity/strided_sliceStridedSlice>dnn/input_from_feature_columns/input_layer/fixed_acidity/ShapeLdnn/input_from_feature_columns/input_layer/fixed_acidity/strided_slice/stackNdnn/input_from_feature_columns/input_layer/fixed_acidity/strided_slice/stack_1Ndnn/input_from_feature_columns/input_layer/fixed_acidity/strided_slice/stack_2*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0
К
Hdnn/input_from_feature_columns/input_layer/fixed_acidity/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
О
Fdnn/input_from_feature_columns/input_layer/fixed_acidity/Reshape/shapePackFdnn/input_from_feature_columns/input_layer/fixed_acidity/strided_sliceHdnn/input_from_feature_columns/input_layer/fixed_acidity/Reshape/shape/1*
T0*
N*
_output_shapes
:
К
@dnn/input_from_feature_columns/input_layer/fixed_acidity/ReshapeReshapeCdnn/input_from_feature_columns/input_layer/fixed_acidity/ExpandDimsFdnn/input_from_feature_columns/input_layer/fixed_acidity/Reshape/shape*
T0*'
_output_shapes
:€€€€€€€€€
Ш
Mdnn/input_from_feature_columns/input_layer/free_sulfur_dioxide/ExpandDims/dimConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
з
Idnn/input_from_feature_columns/input_layer/free_sulfur_dioxide/ExpandDims
ExpandDimsPlaceholder_5Mdnn/input_from_feature_columns/input_layer/free_sulfur_dioxide/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€
љ
Ddnn/input_from_feature_columns/input_layer/free_sulfur_dioxide/ShapeShapeIdnn/input_from_feature_columns/input_layer/free_sulfur_dioxide/ExpandDims*
T0*
_output_shapes
:
Ь
Rdnn/input_from_feature_columns/input_layer/free_sulfur_dioxide/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Ю
Tdnn/input_from_feature_columns/input_layer/free_sulfur_dioxide/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ю
Tdnn/input_from_feature_columns/input_layer/free_sulfur_dioxide/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
и
Ldnn/input_from_feature_columns/input_layer/free_sulfur_dioxide/strided_sliceStridedSliceDdnn/input_from_feature_columns/input_layer/free_sulfur_dioxide/ShapeRdnn/input_from_feature_columns/input_layer/free_sulfur_dioxide/strided_slice/stackTdnn/input_from_feature_columns/input_layer/free_sulfur_dioxide/strided_slice/stack_1Tdnn/input_from_feature_columns/input_layer/free_sulfur_dioxide/strided_slice/stack_2*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0
Р
Ndnn/input_from_feature_columns/input_layer/free_sulfur_dioxide/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
†
Ldnn/input_from_feature_columns/input_layer/free_sulfur_dioxide/Reshape/shapePackLdnn/input_from_feature_columns/input_layer/free_sulfur_dioxide/strided_sliceNdnn/input_from_feature_columns/input_layer/free_sulfur_dioxide/Reshape/shape/1*
N*
_output_shapes
:*
T0
Ь
Fdnn/input_from_feature_columns/input_layer/free_sulfur_dioxide/ReshapeReshapeIdnn/input_from_feature_columns/input_layer/free_sulfur_dioxide/ExpandDimsLdnn/input_from_feature_columns/input_layer/free_sulfur_dioxide/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0
З
<dnn/input_from_feature_columns/input_layer/pH/ExpandDims/dimConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
≈
8dnn/input_from_feature_columns/input_layer/pH/ExpandDims
ExpandDimsPlaceholder_8<dnn/input_from_feature_columns/input_layer/pH/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€
Ы
3dnn/input_from_feature_columns/input_layer/pH/ShapeShape8dnn/input_from_feature_columns/input_layer/pH/ExpandDims*
T0*
_output_shapes
:
Л
Adnn/input_from_feature_columns/input_layer/pH/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
Н
Cdnn/input_from_feature_columns/input_layer/pH/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Н
Cdnn/input_from_feature_columns/input_layer/pH/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
У
;dnn/input_from_feature_columns/input_layer/pH/strided_sliceStridedSlice3dnn/input_from_feature_columns/input_layer/pH/ShapeAdnn/input_from_feature_columns/input_layer/pH/strided_slice/stackCdnn/input_from_feature_columns/input_layer/pH/strided_slice/stack_1Cdnn/input_from_feature_columns/input_layer/pH/strided_slice/stack_2*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0

=dnn/input_from_feature_columns/input_layer/pH/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
н
;dnn/input_from_feature_columns/input_layer/pH/Reshape/shapePack;dnn/input_from_feature_columns/input_layer/pH/strided_slice=dnn/input_from_feature_columns/input_layer/pH/Reshape/shape/1*
T0*
N*
_output_shapes
:
й
5dnn/input_from_feature_columns/input_layer/pH/ReshapeReshape8dnn/input_from_feature_columns/input_layer/pH/ExpandDims;dnn/input_from_feature_columns/input_layer/pH/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0
У
Hdnn/input_from_feature_columns/input_layer/residual_sugar/ExpandDims/dimConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
Ё
Ddnn/input_from_feature_columns/input_layer/residual_sugar/ExpandDims
ExpandDimsPlaceholder_3Hdnn/input_from_feature_columns/input_layer/residual_sugar/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€
≥
?dnn/input_from_feature_columns/input_layer/residual_sugar/ShapeShapeDdnn/input_from_feature_columns/input_layer/residual_sugar/ExpandDims*
T0*
_output_shapes
:
Ч
Mdnn/input_from_feature_columns/input_layer/residual_sugar/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
Щ
Odnn/input_from_feature_columns/input_layer/residual_sugar/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Щ
Odnn/input_from_feature_columns/input_layer/residual_sugar/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ѕ
Gdnn/input_from_feature_columns/input_layer/residual_sugar/strided_sliceStridedSlice?dnn/input_from_feature_columns/input_layer/residual_sugar/ShapeMdnn/input_from_feature_columns/input_layer/residual_sugar/strided_slice/stackOdnn/input_from_feature_columns/input_layer/residual_sugar/strided_slice/stack_1Odnn/input_from_feature_columns/input_layer/residual_sugar/strided_slice/stack_2*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0
Л
Idnn/input_from_feature_columns/input_layer/residual_sugar/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
С
Gdnn/input_from_feature_columns/input_layer/residual_sugar/Reshape/shapePackGdnn/input_from_feature_columns/input_layer/residual_sugar/strided_sliceIdnn/input_from_feature_columns/input_layer/residual_sugar/Reshape/shape/1*
T0*
N*
_output_shapes
:
Н
Adnn/input_from_feature_columns/input_layer/residual_sugar/ReshapeReshapeDdnn/input_from_feature_columns/input_layer/residual_sugar/ExpandDimsGdnn/input_from_feature_columns/input_layer/residual_sugar/Reshape/shape*
T0*'
_output_shapes
:€€€€€€€€€
О
Cdnn/input_from_feature_columns/input_layer/sulphates/ExpandDims/dimConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
”
?dnn/input_from_feature_columns/input_layer/sulphates/ExpandDims
ExpandDimsPlaceholder_9Cdnn/input_from_feature_columns/input_layer/sulphates/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€
©
:dnn/input_from_feature_columns/input_layer/sulphates/ShapeShape?dnn/input_from_feature_columns/input_layer/sulphates/ExpandDims*
T0*
_output_shapes
:
Т
Hdnn/input_from_feature_columns/input_layer/sulphates/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
Ф
Jdnn/input_from_feature_columns/input_layer/sulphates/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ф
Jdnn/input_from_feature_columns/input_layer/sulphates/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
ґ
Bdnn/input_from_feature_columns/input_layer/sulphates/strided_sliceStridedSlice:dnn/input_from_feature_columns/input_layer/sulphates/ShapeHdnn/input_from_feature_columns/input_layer/sulphates/strided_slice/stackJdnn/input_from_feature_columns/input_layer/sulphates/strided_slice/stack_1Jdnn/input_from_feature_columns/input_layer/sulphates/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: 
Ж
Ddnn/input_from_feature_columns/input_layer/sulphates/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
В
Bdnn/input_from_feature_columns/input_layer/sulphates/Reshape/shapePackBdnn/input_from_feature_columns/input_layer/sulphates/strided_sliceDdnn/input_from_feature_columns/input_layer/sulphates/Reshape/shape/1*
T0*
N*
_output_shapes
:
ю
<dnn/input_from_feature_columns/input_layer/sulphates/ReshapeReshape?dnn/input_from_feature_columns/input_layer/sulphates/ExpandDimsBdnn/input_from_feature_columns/input_layer/sulphates/Reshape/shape*
T0*'
_output_shapes
:€€€€€€€€€
Щ
Ndnn/input_from_feature_columns/input_layer/total_sulfur_dioxide/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
€€€€€€€€€
й
Jdnn/input_from_feature_columns/input_layer/total_sulfur_dioxide/ExpandDims
ExpandDimsPlaceholder_6Ndnn/input_from_feature_columns/input_layer/total_sulfur_dioxide/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€
њ
Ednn/input_from_feature_columns/input_layer/total_sulfur_dioxide/ShapeShapeJdnn/input_from_feature_columns/input_layer/total_sulfur_dioxide/ExpandDims*
_output_shapes
:*
T0
Э
Sdnn/input_from_feature_columns/input_layer/total_sulfur_dioxide/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Я
Udnn/input_from_feature_columns/input_layer/total_sulfur_dioxide/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Я
Udnn/input_from_feature_columns/input_layer/total_sulfur_dioxide/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
н
Mdnn/input_from_feature_columns/input_layer/total_sulfur_dioxide/strided_sliceStridedSliceEdnn/input_from_feature_columns/input_layer/total_sulfur_dioxide/ShapeSdnn/input_from_feature_columns/input_layer/total_sulfur_dioxide/strided_slice/stackUdnn/input_from_feature_columns/input_layer/total_sulfur_dioxide/strided_slice/stack_1Udnn/input_from_feature_columns/input_layer/total_sulfur_dioxide/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: 
С
Odnn/input_from_feature_columns/input_layer/total_sulfur_dioxide/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
£
Mdnn/input_from_feature_columns/input_layer/total_sulfur_dioxide/Reshape/shapePackMdnn/input_from_feature_columns/input_layer/total_sulfur_dioxide/strided_sliceOdnn/input_from_feature_columns/input_layer/total_sulfur_dioxide/Reshape/shape/1*
T0*
N*
_output_shapes
:
Я
Gdnn/input_from_feature_columns/input_layer/total_sulfur_dioxide/ReshapeReshapeJdnn/input_from_feature_columns/input_layer/total_sulfur_dioxide/ExpandDimsMdnn/input_from_feature_columns/input_layer/total_sulfur_dioxide/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
T0
Х
Jdnn/input_from_feature_columns/input_layer/volatile_acidity/ExpandDims/dimConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
б
Fdnn/input_from_feature_columns/input_layer/volatile_acidity/ExpandDims
ExpandDimsPlaceholder_1Jdnn/input_from_feature_columns/input_layer/volatile_acidity/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€
Ј
Adnn/input_from_feature_columns/input_layer/volatile_acidity/ShapeShapeFdnn/input_from_feature_columns/input_layer/volatile_acidity/ExpandDims*
T0*
_output_shapes
:
Щ
Odnn/input_from_feature_columns/input_layer/volatile_acidity/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
Ы
Qdnn/input_from_feature_columns/input_layer/volatile_acidity/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
Ы
Qdnn/input_from_feature_columns/input_layer/volatile_acidity/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ў
Idnn/input_from_feature_columns/input_layer/volatile_acidity/strided_sliceStridedSliceAdnn/input_from_feature_columns/input_layer/volatile_acidity/ShapeOdnn/input_from_feature_columns/input_layer/volatile_acidity/strided_slice/stackQdnn/input_from_feature_columns/input_layer/volatile_acidity/strided_slice/stack_1Qdnn/input_from_feature_columns/input_layer/volatile_acidity/strided_slice/stack_2*
_output_shapes
: *
T0*
Index0*
shrink_axis_mask
Н
Kdnn/input_from_feature_columns/input_layer/volatile_acidity/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
Ч
Idnn/input_from_feature_columns/input_layer/volatile_acidity/Reshape/shapePackIdnn/input_from_feature_columns/input_layer/volatile_acidity/strided_sliceKdnn/input_from_feature_columns/input_layer/volatile_acidity/Reshape/shape/1*
N*
_output_shapes
:*
T0
У
Cdnn/input_from_feature_columns/input_layer/volatile_acidity/ReshapeReshapeFdnn/input_from_feature_columns/input_layer/volatile_acidity/ExpandDimsIdnn/input_from_feature_columns/input_layer/volatile_acidity/Reshape/shape*
T0*'
_output_shapes
:€€€€€€€€€
x
6dnn/input_from_feature_columns/input_layer/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
ц
1dnn/input_from_feature_columns/input_layer/concatConcatV2:dnn/input_from_feature_columns/input_layer/alcohol/Reshape<dnn/input_from_feature_columns/input_layer/chlorides/Reshape>dnn/input_from_feature_columns/input_layer/citric_acid/Reshape:dnn/input_from_feature_columns/input_layer/density/Reshape@dnn/input_from_feature_columns/input_layer/fixed_acidity/ReshapeFdnn/input_from_feature_columns/input_layer/free_sulfur_dioxide/Reshape5dnn/input_from_feature_columns/input_layer/pH/ReshapeAdnn/input_from_feature_columns/input_layer/residual_sugar/Reshape<dnn/input_from_feature_columns/input_layer/sulphates/ReshapeGdnn/input_from_feature_columns/input_layer/total_sulfur_dioxide/ReshapeCdnn/input_from_feature_columns/input_layer/volatile_acidity/Reshape6dnn/input_from_feature_columns/input_layer/concat/axis*
T0*
N*'
_output_shapes
:€€€€€€€€€
≈
@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"      *2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
:
Ј
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *Ё√Њ*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
: 
Ј
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *Ё√>*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
: 
Е
Hdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0
Ъ
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
: 
ђ
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:
Ю
:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*
_output_shapes

:*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0
ќ
dnn/hiddenlayer_0/kernel/part_0VarHandleOp*
dtype0*
_output_shapes
: *0
shared_name!dnn/hiddenlayer_0/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
shape
:
П
@dnn/hiddenlayer_0/kernel/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_0/kernel/part_0*
_output_shapes
: 
Ў
&dnn/hiddenlayer_0/kernel/part_0/AssignAssignVariableOpdnn/hiddenlayer_0/kernel/part_0:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0
«
3dnn/hiddenlayer_0/kernel/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes

:
Ѓ
/dnn/hiddenlayer_0/bias/part_0/Initializer/zerosConst*
valueB*    *0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
:
ƒ
dnn/hiddenlayer_0/bias/part_0VarHandleOp*
shape:*
dtype0*
_output_shapes
: *.
shared_namednn/hiddenlayer_0/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0
Л
>dnn/hiddenlayer_0/bias/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_0/bias/part_0*
_output_shapes
: 
«
$dnn/hiddenlayer_0/bias/part_0/AssignAssignVariableOpdnn/hiddenlayer_0/bias/part_0/dnn/hiddenlayer_0/bias/part_0/Initializer/zeros*
dtype0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0
љ
1dnn/hiddenlayer_0/bias/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
:
З
'dnn/hiddenlayer_0/kernel/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes

:
v
dnn/hiddenlayer_0/kernelIdentity'dnn/hiddenlayer_0/kernel/ReadVariableOp*
T0*
_output_shapes

:
°
dnn/hiddenlayer_0/MatMulMatMul1dnn/input_from_feature_columns/input_layer/concatdnn/hiddenlayer_0/kernel*
T0*'
_output_shapes
:€€€€€€€€€

%dnn/hiddenlayer_0/bias/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
:
n
dnn/hiddenlayer_0/biasIdentity%dnn/hiddenlayer_0/bias/ReadVariableOp*
T0*
_output_shapes
:
И
dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMuldnn/hiddenlayer_0/bias*
T0*'
_output_shapes
:€€€€€€€€€
k
dnn/hiddenlayer_0/ReluReludnn/hiddenlayer_0/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
[
dnn/zero_fraction/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 
В
dnn/zero_fraction/EqualEqualdnn/hiddenlayer_0/Reludnn/zero_fraction/zero*
T0*'
_output_shapes
:€€€€€€€€€
x
dnn/zero_fraction/CastCastdnn/zero_fraction/Equal*

SrcT0
*

DstT0*'
_output_shapes
:€€€€€€€€€
h
dnn/zero_fraction/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
p
dnn/zero_fraction/MeanMeandnn/zero_fraction/Castdnn/zero_fraction/Const*
T0*
_output_shapes
: 
†
2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsConst*
dtype0*
_output_shapes
: *>
value5B3 B-dnn/dnn/hiddenlayer_0/fraction_of_zero_values
Ђ
-dnn/dnn/hiddenlayer_0/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsdnn/zero_fraction/Mean*
T0*
_output_shapes
: 
Е
$dnn/dnn/hiddenlayer_0/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_0/activation*
dtype0*
_output_shapes
: 
К
 dnn/dnn/hiddenlayer_0/activationHistogramSummary$dnn/dnn/hiddenlayer_0/activation/tagdnn/hiddenlayer_0/Relu*
_output_shapes
: 
≈
@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"      *2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
:
Ј
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *чь”Њ*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
: 
Ј
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *чь”>*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
: 
Е
Hdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0
Ъ
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
: 
ђ
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes

:
Ю
:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes

:
ќ
dnn/hiddenlayer_1/kernel/part_0VarHandleOp*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
shape
:*
dtype0*
_output_shapes
: *0
shared_name!dnn/hiddenlayer_1/kernel/part_0
П
@dnn/hiddenlayer_1/kernel/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_1/kernel/part_0*
_output_shapes
: 
Ў
&dnn/hiddenlayer_1/kernel/part_0/AssignAssignVariableOpdnn/hiddenlayer_1/kernel/part_0:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0
«
3dnn/hiddenlayer_1/kernel/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes

:*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0
Ѓ
/dnn/hiddenlayer_1/bias/part_0/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0
ƒ
dnn/hiddenlayer_1/bias/part_0VarHandleOp*
shape:*
dtype0*
_output_shapes
: *.
shared_namednn/hiddenlayer_1/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0
Л
>dnn/hiddenlayer_1/bias/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_1/bias/part_0*
_output_shapes
: 
«
$dnn/hiddenlayer_1/bias/part_0/AssignAssignVariableOpdnn/hiddenlayer_1/bias/part_0/dnn/hiddenlayer_1/bias/part_0/Initializer/zeros*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0
љ
1dnn/hiddenlayer_1/bias/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
:
З
'dnn/hiddenlayer_1/kernel/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes

:
v
dnn/hiddenlayer_1/kernelIdentity'dnn/hiddenlayer_1/kernel/ReadVariableOp*
T0*
_output_shapes

:
Ж
dnn/hiddenlayer_1/MatMulMatMuldnn/hiddenlayer_0/Reludnn/hiddenlayer_1/kernel*
T0*'
_output_shapes
:€€€€€€€€€

%dnn/hiddenlayer_1/bias/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
:
n
dnn/hiddenlayer_1/biasIdentity%dnn/hiddenlayer_1/bias/ReadVariableOp*
T0*
_output_shapes
:
И
dnn/hiddenlayer_1/BiasAddBiasAdddnn/hiddenlayer_1/MatMuldnn/hiddenlayer_1/bias*
T0*'
_output_shapes
:€€€€€€€€€
k
dnn/hiddenlayer_1/ReluReludnn/hiddenlayer_1/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
]
dnn/zero_fraction_1/zeroConst*
dtype0*
_output_shapes
: *
valueB
 *    
Ж
dnn/zero_fraction_1/EqualEqualdnn/hiddenlayer_1/Reludnn/zero_fraction_1/zero*'
_output_shapes
:€€€€€€€€€*
T0
|
dnn/zero_fraction_1/CastCastdnn/zero_fraction_1/Equal*

SrcT0
*

DstT0*'
_output_shapes
:€€€€€€€€€
j
dnn/zero_fraction_1/ConstConst*
dtype0*
_output_shapes
:*
valueB"       
v
dnn/zero_fraction_1/MeanMeandnn/zero_fraction_1/Castdnn/zero_fraction_1/Const*
T0*
_output_shapes
: 
†
2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_1/fraction_of_zero_values*
dtype0*
_output_shapes
: 
≠
-dnn/dnn/hiddenlayer_1/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsdnn/zero_fraction_1/Mean*
_output_shapes
: *
T0
Е
$dnn/dnn/hiddenlayer_1/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_1/activation*
dtype0*
_output_shapes
: 
К
 dnn/dnn/hiddenlayer_1/activationHistogramSummary$dnn/dnn/hiddenlayer_1/activation/tagdnn/hiddenlayer_1/Relu*
_output_shapes
: 
Ј
9dnn/logits/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"      *+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
:
©
7dnn/logits/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *  Ањ*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
: 
©
7dnn/logits/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *  А?*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
: 
р
Adnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform9dnn/logits/kernel/part_0/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*
T0*+
_class!
loc:@dnn/logits/kernel/part_0
ю
7dnn/logits/kernel/part_0/Initializer/random_uniform/subSub7dnn/logits/kernel/part_0/Initializer/random_uniform/max7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes
: 
Р
7dnn/logits/kernel/part_0/Initializer/random_uniform/mulMulAdnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniform7dnn/logits/kernel/part_0/Initializer/random_uniform/sub*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:
В
3dnn/logits/kernel/part_0/Initializer/random_uniformAdd7dnn/logits/kernel/part_0/Initializer/random_uniform/mul7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
_output_shapes

:*
T0*+
_class!
loc:@dnn/logits/kernel/part_0
є
dnn/logits/kernel/part_0VarHandleOp*
shape
:*
dtype0*
_output_shapes
: *)
shared_namednn/logits/kernel/part_0*+
_class!
loc:@dnn/logits/kernel/part_0
Б
9dnn/logits/kernel/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/logits/kernel/part_0*
_output_shapes
: 
Љ
dnn/logits/kernel/part_0/AssignAssignVariableOpdnn/logits/kernel/part_03dnn/logits/kernel/part_0/Initializer/random_uniform*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0
≤
,dnn/logits/kernel/part_0/Read/ReadVariableOpReadVariableOpdnn/logits/kernel/part_0*
dtype0*
_output_shapes

:*+
_class!
loc:@dnn/logits/kernel/part_0
†
(dnn/logits/bias/part_0/Initializer/zerosConst*
valueB*    *)
_class
loc:@dnn/logits/bias/part_0*
dtype0*
_output_shapes
:
ѓ
dnn/logits/bias/part_0VarHandleOp*
dtype0*
_output_shapes
: *'
shared_namednn/logits/bias/part_0*)
_class
loc:@dnn/logits/bias/part_0*
shape:
}
7dnn/logits/bias/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/logits/bias/part_0*
_output_shapes
: 
Ђ
dnn/logits/bias/part_0/AssignAssignVariableOpdnn/logits/bias/part_0(dnn/logits/bias/part_0/Initializer/zeros*)
_class
loc:@dnn/logits/bias/part_0*
dtype0
®
*dnn/logits/bias/part_0/Read/ReadVariableOpReadVariableOpdnn/logits/bias/part_0*)
_class
loc:@dnn/logits/bias/part_0*
dtype0*
_output_shapes
:
y
 dnn/logits/kernel/ReadVariableOpReadVariableOpdnn/logits/kernel/part_0*
dtype0*
_output_shapes

:
h
dnn/logits/kernelIdentity dnn/logits/kernel/ReadVariableOp*
T0*
_output_shapes

:
x
dnn/logits/MatMulMatMuldnn/hiddenlayer_1/Reludnn/logits/kernel*
T0*'
_output_shapes
:€€€€€€€€€
q
dnn/logits/bias/ReadVariableOpReadVariableOpdnn/logits/bias/part_0*
dtype0*
_output_shapes
:
`
dnn/logits/biasIdentitydnn/logits/bias/ReadVariableOp*
T0*
_output_shapes
:
s
dnn/logits/BiasAddBiasAdddnn/logits/MatMuldnn/logits/bias*
T0*'
_output_shapes
:€€€€€€€€€
]
dnn/zero_fraction_2/zeroConst*
valueB
 *    *
dtype0*
_output_shapes
: 
В
dnn/zero_fraction_2/EqualEqualdnn/logits/BiasAdddnn/zero_fraction_2/zero*
T0*'
_output_shapes
:€€€€€€€€€
|
dnn/zero_fraction_2/CastCastdnn/zero_fraction_2/Equal*

SrcT0
*

DstT0*'
_output_shapes
:€€€€€€€€€
j
dnn/zero_fraction_2/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
v
dnn/zero_fraction_2/MeanMeandnn/zero_fraction_2/Castdnn/zero_fraction_2/Const*
T0*
_output_shapes
: 
Т
+dnn/dnn/logits/fraction_of_zero_values/tagsConst*7
value.B, B&dnn/dnn/logits/fraction_of_zero_values*
dtype0*
_output_shapes
: 
Я
&dnn/dnn/logits/fraction_of_zero_valuesScalarSummary+dnn/dnn/logits/fraction_of_zero_values/tagsdnn/zero_fraction_2/Mean*
_output_shapes
: *
T0
w
dnn/dnn/logits/activation/tagConst**
value!B Bdnn/dnn/logits/activation*
dtype0*
_output_shapes
: 
x
dnn/dnn/logits/activationHistogramSummarydnn/dnn/logits/activation/tagdnn/logits/BiasAdd*
_output_shapes
: 
W
dnn/head/logits/ShapeShapednn/logits/BiasAdd*
T0*
_output_shapes
:
k
)dnn/head/logits/assert_rank_at_least/rankConst*
value	B :*
dtype0*
_output_shapes
: 
[
Sdnn/head/logits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp
L
Ddnn/head/logits/assert_rank_at_least/static_checks_determined_all_okNoOp

initNoOp

init_all_tablesNoOp

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
:
X
save/IdentityIdentitysave/Read/ReadVariableOp*
T0*
_output_shapes
:
^
save/Identity_1Identitysave/Identity"/device:CPU:0*
T0*
_output_shapes
:
z
save/Read_1/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes

:
`
save/Identity_2Identitysave/Read_1/ReadVariableOp*
T0*
_output_shapes

:
d
save/Identity_3Identitysave/Identity_2"/device:CPU:0*
T0*
_output_shapes

:
t
save/Read_2/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
:
\
save/Identity_4Identitysave/Read_2/ReadVariableOp*
T0*
_output_shapes
:
`
save/Identity_5Identitysave/Identity_4"/device:CPU:0*
T0*
_output_shapes
:
z
save/Read_3/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes

:
`
save/Identity_6Identitysave/Read_3/ReadVariableOp*
T0*
_output_shapes

:
d
save/Identity_7Identitysave/Identity_6"/device:CPU:0*
T0*
_output_shapes

:
m
save/Read_4/ReadVariableOpReadVariableOpdnn/logits/bias/part_0*
dtype0*
_output_shapes
:
\
save/Identity_8Identitysave/Read_4/ReadVariableOp*
T0*
_output_shapes
:
`
save/Identity_9Identitysave/Identity_8"/device:CPU:0*
T0*
_output_shapes
:
s
save/Read_5/ReadVariableOpReadVariableOpdnn/logits/kernel/part_0*
dtype0*
_output_shapes

:
a
save/Identity_10Identitysave/Read_5/ReadVariableOp*
T0*
_output_shapes

:
f
save/Identity_11Identitysave/Identity_10"/device:CPU:0*
T0*
_output_shapes

:
Д
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_1e8b94bc02124ff5b1f8b87df72b5480/part*
dtype0*
_output_shapes
: 
d
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
М
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
{
save/SaveV2/tensor_namesConst"/device:CPU:0* 
valueBBglobal_step*
dtype0*
_output_shapes
:
t
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
Р
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesglobal_step"/device:CPU:0*
dtypes
2	
†
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
m
save/ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: 
Р
save/ShardedFilename_1ShardedFilenamesave/StringJoinsave/ShardedFilename_1/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
Г
save/Read_6/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias/part_0"/device:CPU:0*
dtype0*
_output_shapes
:
l
save/Identity_12Identitysave/Read_6/ReadVariableOp"/device:CPU:0*
_output_shapes
:*
T0
b
save/Identity_13Identitysave/Identity_12"/device:CPU:0*
T0*
_output_shapes
:
Й
save/Read_7/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
p
save/Identity_14Identitysave/Read_7/ReadVariableOp"/device:CPU:0*
_output_shapes

:*
T0
f
save/Identity_15Identitysave/Identity_14"/device:CPU:0*
_output_shapes

:*
T0
Г
save/Read_8/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias/part_0"/device:CPU:0*
dtype0*
_output_shapes
:
l
save/Identity_16Identitysave/Read_8/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
b
save/Identity_17Identitysave/Identity_16"/device:CPU:0*
_output_shapes
:*
T0
Й
save/Read_9/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
p
save/Identity_18Identitysave/Read_9/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_19Identitysave/Identity_18"/device:CPU:0*
T0*
_output_shapes

:
}
save/Read_10/ReadVariableOpReadVariableOpdnn/logits/bias/part_0"/device:CPU:0*
dtype0*
_output_shapes
:
m
save/Identity_20Identitysave/Read_10/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
b
save/Identity_21Identitysave/Identity_20"/device:CPU:0*
T0*
_output_shapes
:
Г
save/Read_11/ReadVariableOpReadVariableOpdnn/logits/kernel/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
q
save/Identity_22Identitysave/Read_11/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_23Identitysave/Identity_22"/device:CPU:0*
T0*
_output_shapes

:
ы
save/SaveV2_1/tensor_namesConst"/device:CPU:0*Э
valueУBРBdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/logits/biasBdnn/logits/kernel*
dtype0*
_output_shapes
:
Є
save/SaveV2_1/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*W
valueNBLB30 0,30B11 30 0,11:0,30B5 0,5B30 5 0,30:0,5B1 0,1B5 1 0,5:0,1
ь
save/SaveV2_1SaveV2save/ShardedFilename_1save/SaveV2_1/tensor_namessave/SaveV2_1/shape_and_slicessave/Identity_13save/Identity_15save/Identity_17save/Identity_19save/Identity_21save/Identity_23"/device:CPU:0*
dtypes

2
®
save/control_dependency_1Identitysave/ShardedFilename_1^save/SaveV2_1"/device:CPU:0*
T0*)
_class
loc:@save/ShardedFilename_1*
_output_shapes
: 
‘
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilenamesave/ShardedFilename_1^save/control_dependency^save/control_dependency_1"/device:CPU:0*
T0*
N*
_output_shapes
:
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0
®
save/Identity_24Identity
save/Const^save/MergeV2Checkpoints^save/control_dependency^save/control_dependency_1"/device:CPU:0*
T0*
_output_shapes
: 
~
save/RestoreV2/tensor_namesConst"/device:CPU:0* 
valueBBglobal_step*
dtype0*
_output_shapes
:
w
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
Я
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
_output_shapes
:*
dtypes
2	
s
save/AssignAssignglobal_stepsave/RestoreV2*
T0	*
_class
loc:@global_step*
_output_shapes
: 
(
save/restore_shardNoOp^save/Assign
ю
save/RestoreV2_1/tensor_namesConst"/device:CPU:0*Э
valueУBРBdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/logits/biasBdnn/logits/kernel*
dtype0*
_output_shapes
:
ї
!save/RestoreV2_1/shape_and_slicesConst"/device:CPU:0*W
valueNBLB30 0,30B11 30 0,11:0,30B5 0,5B30 5 0,30:0,5B1 0,1B5 1 0,5:0,1*
dtype0*
_output_shapes
:
÷
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices"/device:CPU:0*
dtypes

2*D
_output_shapes2
0::::::
b
save/Identity_25Identitysave/RestoreV2_1"/device:CPU:0*
T0*
_output_shapes
:
v
save/AssignVariableOpAssignVariableOpdnn/hiddenlayer_0/bias/part_0save/Identity_25"/device:CPU:0*
dtype0
h
save/Identity_26Identitysave/RestoreV2_1:1"/device:CPU:0*
T0*
_output_shapes

:
z
save/AssignVariableOp_1AssignVariableOpdnn/hiddenlayer_0/kernel/part_0save/Identity_26"/device:CPU:0*
dtype0
d
save/Identity_27Identitysave/RestoreV2_1:2"/device:CPU:0*
T0*
_output_shapes
:
x
save/AssignVariableOp_2AssignVariableOpdnn/hiddenlayer_1/bias/part_0save/Identity_27"/device:CPU:0*
dtype0
h
save/Identity_28Identitysave/RestoreV2_1:3"/device:CPU:0*
T0*
_output_shapes

:
z
save/AssignVariableOp_3AssignVariableOpdnn/hiddenlayer_1/kernel/part_0save/Identity_28"/device:CPU:0*
dtype0
d
save/Identity_29Identitysave/RestoreV2_1:4"/device:CPU:0*
T0*
_output_shapes
:
q
save/AssignVariableOp_4AssignVariableOpdnn/logits/bias/part_0save/Identity_29"/device:CPU:0*
dtype0
h
save/Identity_30Identitysave/RestoreV2_1:5"/device:CPU:0*
_output_shapes

:*
T0
s
save/AssignVariableOp_5AssignVariableOpdnn/logits/kernel/part_0save/Identity_30"/device:CPU:0*
dtype0
≈
save/restore_shard_1NoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_2^save/AssignVariableOp_3^save/AssignVariableOp_4^save/AssignVariableOp_5"/device:CPU:0
2
save/restore_all/NoOpNoOp^save/restore_shard
E
save/restore_all/NoOp_1NoOp^save/restore_shard_1"/device:CPU:0
J
save/restore_allNoOp^save/restore_all/NoOp^save/restore_all/NoOp_1"?
save/Const:0save/Identity_24:0save/restore_all (5 @F8"В
	summariesф
с
/dnn/dnn/hiddenlayer_0/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_0/activation:0
/dnn/dnn/hiddenlayer_1/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_1/activation:0
(dnn/dnn/logits/fraction_of_zero_values:0
dnn/dnn/logits/activation:0"≠

trainable_variablesХ
Т

м
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign5dnn/hiddenlayer_0/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_0/kernel  "(2<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:08
÷
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign3dnn/hiddenlayer_0/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_0/bias "(21dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:08
м
!dnn/hiddenlayer_1/kernel/part_0:0&dnn/hiddenlayer_1/kernel/part_0/Assign5dnn/hiddenlayer_1/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_1/kernel  "(2<dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform:08
÷
dnn/hiddenlayer_1/bias/part_0:0$dnn/hiddenlayer_1/bias/part_0/Assign3dnn/hiddenlayer_1/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_1/bias "(21dnn/hiddenlayer_1/bias/part_0/Initializer/zeros:08
…
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assign.dnn/logits/kernel/part_0/Read/ReadVariableOp:0"
dnn/logits/kernel  "(25dnn/logits/kernel/part_0/Initializer/random_uniform:08
≥
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assign,dnn/logits/bias/part_0/Read/ReadVariableOp:0"
dnn/logits/bias "(2*dnn/logits/bias/part_0/Initializer/zeros:08"k
global_step\Z
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0"%
saved_model_main_op


group_deps"э

	variablesп
м

X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0
м
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign5dnn/hiddenlayer_0/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_0/kernel  "(2<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:08
÷
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign3dnn/hiddenlayer_0/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_0/bias "(21dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:08
м
!dnn/hiddenlayer_1/kernel/part_0:0&dnn/hiddenlayer_1/kernel/part_0/Assign5dnn/hiddenlayer_1/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_1/kernel  "(2<dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform:08
÷
dnn/hiddenlayer_1/bias/part_0:0$dnn/hiddenlayer_1/bias/part_0/Assign3dnn/hiddenlayer_1/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_1/bias "(21dnn/hiddenlayer_1/bias/part_0/Initializer/zeros:08
…
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assign.dnn/logits/kernel/part_0/Read/ReadVariableOp:0"
dnn/logits/kernel  "(25dnn/logits/kernel/part_0/Initializer/random_uniform:08
≥
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assign,dnn/logits/bias/part_0/Read/ReadVariableOp:0"
dnn/logits/bias "(2*dnn/logits/bias/part_0/Initializer/zeros:08*Ъ
predictО
9
free_sulfur_dioxide"
Placeholder_5:0€€€€€€€€€
/
	sulphates"
Placeholder_9:0€€€€€€€€€
1
fixed_acidity 
Placeholder:0€€€€€€€€€
(
pH"
Placeholder_8:0€€€€€€€€€
:
total_sulfur_dioxide"
Placeholder_6:0€€€€€€€€€
/
	chlorides"
Placeholder_4:0€€€€€€€€€
.
alcohol#
Placeholder_10:0€€€€€€€€€
6
volatile_acidity"
Placeholder_1:0€€€€€€€€€
1
citric_acid"
Placeholder_2:0€€€€€€€€€
4
residual_sugar"
Placeholder_3:0€€€€€€€€€
-
density"
Placeholder_7:0€€€€€€€€€:
predictions+
dnn/logits/BiasAdd:0€€€€€€€€€tensorflow/serving/predict