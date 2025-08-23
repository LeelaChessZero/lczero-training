from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class XlaLayoutProto(_message.Message):
    __slots__ = ("minor_to_major",)
    MINOR_TO_MAJOR_FIELD_NUMBER: _ClassVar[int]
    minor_to_major: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, minor_to_major: _Optional[_Iterable[int]] = ...) -> None: ...

class XlaShapeProto(_message.Message):
    __slots__ = ("element_type", "dimensions", "tuple_shapes", "layout", "is_dynamic_dimension")
    class Type(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        PRIMITIVE_TYPE_INVALID: _ClassVar[XlaShapeProto.Type]
        PRED: _ClassVar[XlaShapeProto.Type]
        S4: _ClassVar[XlaShapeProto.Type]
        S8: _ClassVar[XlaShapeProto.Type]
        S16: _ClassVar[XlaShapeProto.Type]
        S32: _ClassVar[XlaShapeProto.Type]
        S64: _ClassVar[XlaShapeProto.Type]
        U4: _ClassVar[XlaShapeProto.Type]
        U8: _ClassVar[XlaShapeProto.Type]
        U16: _ClassVar[XlaShapeProto.Type]
        U32: _ClassVar[XlaShapeProto.Type]
        U64: _ClassVar[XlaShapeProto.Type]
        F16: _ClassVar[XlaShapeProto.Type]
        F32: _ClassVar[XlaShapeProto.Type]
        BF16: _ClassVar[XlaShapeProto.Type]
        F64: _ClassVar[XlaShapeProto.Type]
        F8E5M2: _ClassVar[XlaShapeProto.Type]
        F8E4M3FN: _ClassVar[XlaShapeProto.Type]
        F8E4M3B11FNUZ: _ClassVar[XlaShapeProto.Type]
        F8E5M2FNUZ: _ClassVar[XlaShapeProto.Type]
        F8E4M3FNUZ: _ClassVar[XlaShapeProto.Type]
        C64: _ClassVar[XlaShapeProto.Type]
        C128: _ClassVar[XlaShapeProto.Type]
        TUPLE: _ClassVar[XlaShapeProto.Type]
        OPAQUE_TYPE: _ClassVar[XlaShapeProto.Type]
        TOKEN: _ClassVar[XlaShapeProto.Type]
    PRIMITIVE_TYPE_INVALID: XlaShapeProto.Type
    PRED: XlaShapeProto.Type
    S4: XlaShapeProto.Type
    S8: XlaShapeProto.Type
    S16: XlaShapeProto.Type
    S32: XlaShapeProto.Type
    S64: XlaShapeProto.Type
    U4: XlaShapeProto.Type
    U8: XlaShapeProto.Type
    U16: XlaShapeProto.Type
    U32: XlaShapeProto.Type
    U64: XlaShapeProto.Type
    F16: XlaShapeProto.Type
    F32: XlaShapeProto.Type
    BF16: XlaShapeProto.Type
    F64: XlaShapeProto.Type
    F8E5M2: XlaShapeProto.Type
    F8E4M3FN: XlaShapeProto.Type
    F8E4M3B11FNUZ: XlaShapeProto.Type
    F8E5M2FNUZ: XlaShapeProto.Type
    F8E4M3FNUZ: XlaShapeProto.Type
    C64: XlaShapeProto.Type
    C128: XlaShapeProto.Type
    TUPLE: XlaShapeProto.Type
    OPAQUE_TYPE: XlaShapeProto.Type
    TOKEN: XlaShapeProto.Type
    ELEMENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    DIMENSIONS_FIELD_NUMBER: _ClassVar[int]
    TUPLE_SHAPES_FIELD_NUMBER: _ClassVar[int]
    LAYOUT_FIELD_NUMBER: _ClassVar[int]
    IS_DYNAMIC_DIMENSION_FIELD_NUMBER: _ClassVar[int]
    element_type: XlaShapeProto.Type
    dimensions: _containers.RepeatedScalarFieldContainer[int]
    tuple_shapes: _containers.RepeatedCompositeFieldContainer[XlaShapeProto]
    layout: XlaLayoutProto
    is_dynamic_dimension: _containers.RepeatedScalarFieldContainer[bool]
    def __init__(self, element_type: _Optional[_Union[XlaShapeProto.Type, str]] = ..., dimensions: _Optional[_Iterable[int]] = ..., tuple_shapes: _Optional[_Iterable[_Union[XlaShapeProto, _Mapping]]] = ..., layout: _Optional[_Union[XlaLayoutProto, _Mapping]] = ..., is_dynamic_dimension: _Optional[_Iterable[bool]] = ...) -> None: ...

class XlaProgramShapeProto(_message.Message):
    __slots__ = ("parameters", "result", "parameter_names")
    PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    PARAMETER_NAMES_FIELD_NUMBER: _ClassVar[int]
    parameters: _containers.RepeatedCompositeFieldContainer[XlaShapeProto]
    result: XlaShapeProto
    parameter_names: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, parameters: _Optional[_Iterable[_Union[XlaShapeProto, _Mapping]]] = ..., result: _Optional[_Union[XlaShapeProto, _Mapping]] = ..., parameter_names: _Optional[_Iterable[str]] = ...) -> None: ...

class XlaOpMetadata(_message.Message):
    __slots__ = ("op_type", "op_name", "source_file", "source_line")
    OP_TYPE_FIELD_NUMBER: _ClassVar[int]
    OP_NAME_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FILE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_LINE_FIELD_NUMBER: _ClassVar[int]
    op_type: str
    op_name: str
    source_file: str
    source_line: int
    def __init__(self, op_type: _Optional[str] = ..., op_name: _Optional[str] = ..., source_file: _Optional[str] = ..., source_line: _Optional[int] = ...) -> None: ...

class XlaLiteralProto(_message.Message):
    __slots__ = ("shape", "preds", "s4s", "u4s", "s8s", "u8s", "s32s", "s64s", "u32s", "u64s", "f32s", "f64s", "c64s", "c128s", "tuple_literals", "f16s", "bf16s", "u16s", "s16s", "f8e5m2s", "f8e4m3fns", "f8e4m3b11fnuzs", "f8e5m2fnuzs", "f8e4m3fnuzs", "sparse_indices")
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    PREDS_FIELD_NUMBER: _ClassVar[int]
    S4S_FIELD_NUMBER: _ClassVar[int]
    U4S_FIELD_NUMBER: _ClassVar[int]
    S8S_FIELD_NUMBER: _ClassVar[int]
    U8S_FIELD_NUMBER: _ClassVar[int]
    S32S_FIELD_NUMBER: _ClassVar[int]
    S64S_FIELD_NUMBER: _ClassVar[int]
    U32S_FIELD_NUMBER: _ClassVar[int]
    U64S_FIELD_NUMBER: _ClassVar[int]
    F32S_FIELD_NUMBER: _ClassVar[int]
    F64S_FIELD_NUMBER: _ClassVar[int]
    C64S_FIELD_NUMBER: _ClassVar[int]
    C128S_FIELD_NUMBER: _ClassVar[int]
    TUPLE_LITERALS_FIELD_NUMBER: _ClassVar[int]
    F16S_FIELD_NUMBER: _ClassVar[int]
    BF16S_FIELD_NUMBER: _ClassVar[int]
    U16S_FIELD_NUMBER: _ClassVar[int]
    S16S_FIELD_NUMBER: _ClassVar[int]
    F8E5M2S_FIELD_NUMBER: _ClassVar[int]
    F8E4M3FNS_FIELD_NUMBER: _ClassVar[int]
    F8E4M3B11FNUZS_FIELD_NUMBER: _ClassVar[int]
    F8E5M2FNUZS_FIELD_NUMBER: _ClassVar[int]
    F8E4M3FNUZS_FIELD_NUMBER: _ClassVar[int]
    SPARSE_INDICES_FIELD_NUMBER: _ClassVar[int]
    shape: XlaShapeProto
    preds: _containers.RepeatedScalarFieldContainer[bool]
    s4s: bytes
    u4s: bytes
    s8s: bytes
    u8s: bytes
    s32s: _containers.RepeatedScalarFieldContainer[int]
    s64s: _containers.RepeatedScalarFieldContainer[int]
    u32s: _containers.RepeatedScalarFieldContainer[int]
    u64s: _containers.RepeatedScalarFieldContainer[int]
    f32s: _containers.RepeatedScalarFieldContainer[float]
    f64s: _containers.RepeatedScalarFieldContainer[float]
    c64s: _containers.RepeatedScalarFieldContainer[float]
    c128s: _containers.RepeatedScalarFieldContainer[float]
    tuple_literals: _containers.RepeatedCompositeFieldContainer[XlaLiteralProto]
    f16s: bytes
    bf16s: bytes
    u16s: bytes
    s16s: bytes
    f8e5m2s: bytes
    f8e4m3fns: bytes
    f8e4m3b11fnuzs: bytes
    f8e5m2fnuzs: bytes
    f8e4m3fnuzs: bytes
    sparse_indices: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, shape: _Optional[_Union[XlaShapeProto, _Mapping]] = ..., preds: _Optional[_Iterable[bool]] = ..., s4s: _Optional[bytes] = ..., u4s: _Optional[bytes] = ..., s8s: _Optional[bytes] = ..., u8s: _Optional[bytes] = ..., s32s: _Optional[_Iterable[int]] = ..., s64s: _Optional[_Iterable[int]] = ..., u32s: _Optional[_Iterable[int]] = ..., u64s: _Optional[_Iterable[int]] = ..., f32s: _Optional[_Iterable[float]] = ..., f64s: _Optional[_Iterable[float]] = ..., c64s: _Optional[_Iterable[float]] = ..., c128s: _Optional[_Iterable[float]] = ..., tuple_literals: _Optional[_Iterable[_Union[XlaLiteralProto, _Mapping]]] = ..., f16s: _Optional[bytes] = ..., bf16s: _Optional[bytes] = ..., u16s: _Optional[bytes] = ..., s16s: _Optional[bytes] = ..., f8e5m2s: _Optional[bytes] = ..., f8e4m3fns: _Optional[bytes] = ..., f8e4m3b11fnuzs: _Optional[bytes] = ..., f8e5m2fnuzs: _Optional[bytes] = ..., f8e4m3fnuzs: _Optional[bytes] = ..., sparse_indices: _Optional[_Iterable[int]] = ...) -> None: ...

class XlaWindowDimension(_message.Message):
    __slots__ = ("size", "stride", "padding_low", "padding_high", "window_dilation", "base_dilation", "window_reversal")
    SIZE_FIELD_NUMBER: _ClassVar[int]
    STRIDE_FIELD_NUMBER: _ClassVar[int]
    PADDING_LOW_FIELD_NUMBER: _ClassVar[int]
    PADDING_HIGH_FIELD_NUMBER: _ClassVar[int]
    WINDOW_DILATION_FIELD_NUMBER: _ClassVar[int]
    BASE_DILATION_FIELD_NUMBER: _ClassVar[int]
    WINDOW_REVERSAL_FIELD_NUMBER: _ClassVar[int]
    size: int
    stride: int
    padding_low: int
    padding_high: int
    window_dilation: int
    base_dilation: int
    window_reversal: bool
    def __init__(self, size: _Optional[int] = ..., stride: _Optional[int] = ..., padding_low: _Optional[int] = ..., padding_high: _Optional[int] = ..., window_dilation: _Optional[int] = ..., base_dilation: _Optional[int] = ..., window_reversal: bool = ...) -> None: ...

class XlaWindow(_message.Message):
    __slots__ = ("dimensions",)
    DIMENSIONS_FIELD_NUMBER: _ClassVar[int]
    dimensions: _containers.RepeatedCompositeFieldContainer[XlaWindowDimension]
    def __init__(self, dimensions: _Optional[_Iterable[_Union[XlaWindowDimension, _Mapping]]] = ...) -> None: ...

class XlaConvolutionDimensionNumbers(_message.Message):
    __slots__ = ("input_batch_dimension", "input_feature_dimension", "input_spatial_dimensions", "kernel_input_feature_dimension", "kernel_output_feature_dimension", "kernel_spatial_dimensions", "output_batch_dimension", "output_feature_dimension", "output_spatial_dimensions")
    INPUT_BATCH_DIMENSION_FIELD_NUMBER: _ClassVar[int]
    INPUT_FEATURE_DIMENSION_FIELD_NUMBER: _ClassVar[int]
    INPUT_SPATIAL_DIMENSIONS_FIELD_NUMBER: _ClassVar[int]
    KERNEL_INPUT_FEATURE_DIMENSION_FIELD_NUMBER: _ClassVar[int]
    KERNEL_OUTPUT_FEATURE_DIMENSION_FIELD_NUMBER: _ClassVar[int]
    KERNEL_SPATIAL_DIMENSIONS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_BATCH_DIMENSION_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FEATURE_DIMENSION_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SPATIAL_DIMENSIONS_FIELD_NUMBER: _ClassVar[int]
    input_batch_dimension: int
    input_feature_dimension: int
    input_spatial_dimensions: _containers.RepeatedScalarFieldContainer[int]
    kernel_input_feature_dimension: int
    kernel_output_feature_dimension: int
    kernel_spatial_dimensions: _containers.RepeatedScalarFieldContainer[int]
    output_batch_dimension: int
    output_feature_dimension: int
    output_spatial_dimensions: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, input_batch_dimension: _Optional[int] = ..., input_feature_dimension: _Optional[int] = ..., input_spatial_dimensions: _Optional[_Iterable[int]] = ..., kernel_input_feature_dimension: _Optional[int] = ..., kernel_output_feature_dimension: _Optional[int] = ..., kernel_spatial_dimensions: _Optional[_Iterable[int]] = ..., output_batch_dimension: _Optional[int] = ..., output_feature_dimension: _Optional[int] = ..., output_spatial_dimensions: _Optional[_Iterable[int]] = ...) -> None: ...

class XlaDotDimensionNumbers(_message.Message):
    __slots__ = ("lhs_contracting_dimensions", "rhs_contracting_dimensions", "lhs_batch_dimensions", "rhs_batch_dimensions")
    LHS_CONTRACTING_DIMENSIONS_FIELD_NUMBER: _ClassVar[int]
    RHS_CONTRACTING_DIMENSIONS_FIELD_NUMBER: _ClassVar[int]
    LHS_BATCH_DIMENSIONS_FIELD_NUMBER: _ClassVar[int]
    RHS_BATCH_DIMENSIONS_FIELD_NUMBER: _ClassVar[int]
    lhs_contracting_dimensions: _containers.RepeatedScalarFieldContainer[int]
    rhs_contracting_dimensions: _containers.RepeatedScalarFieldContainer[int]
    lhs_batch_dimensions: _containers.RepeatedScalarFieldContainer[int]
    rhs_batch_dimensions: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, lhs_contracting_dimensions: _Optional[_Iterable[int]] = ..., rhs_contracting_dimensions: _Optional[_Iterable[int]] = ..., lhs_batch_dimensions: _Optional[_Iterable[int]] = ..., rhs_batch_dimensions: _Optional[_Iterable[int]] = ...) -> None: ...

class XlaGatherDimensionNumbers(_message.Message):
    __slots__ = ("offset_dims", "collapsed_slice_dims", "start_index_map", "index_vector_dim")
    OFFSET_DIMS_FIELD_NUMBER: _ClassVar[int]
    COLLAPSED_SLICE_DIMS_FIELD_NUMBER: _ClassVar[int]
    START_INDEX_MAP_FIELD_NUMBER: _ClassVar[int]
    INDEX_VECTOR_DIM_FIELD_NUMBER: _ClassVar[int]
    offset_dims: _containers.RepeatedScalarFieldContainer[int]
    collapsed_slice_dims: _containers.RepeatedScalarFieldContainer[int]
    start_index_map: _containers.RepeatedScalarFieldContainer[int]
    index_vector_dim: int
    def __init__(self, offset_dims: _Optional[_Iterable[int]] = ..., collapsed_slice_dims: _Optional[_Iterable[int]] = ..., start_index_map: _Optional[_Iterable[int]] = ..., index_vector_dim: _Optional[int] = ...) -> None: ...

class HloInstructionProto(_message.Message):
    __slots__ = ("name", "opcode", "shape", "metadata", "literal", "parameter_number", "tuple_index", "window", "convolution_dimension_numbers", "slice_dimensions", "dot_dimension_numbers", "dimensions", "gather_dimension_numbers", "gather_slice_sizes", "indices_are_sorted", "unique_indices", "id", "operand_ids", "called_computation_ids", "comparison_direction")
    class SliceDimensions(_message.Message):
        __slots__ = ("start", "limit", "stride")
        START_FIELD_NUMBER: _ClassVar[int]
        LIMIT_FIELD_NUMBER: _ClassVar[int]
        STRIDE_FIELD_NUMBER: _ClassVar[int]
        start: int
        limit: int
        stride: int
        def __init__(self, start: _Optional[int] = ..., limit: _Optional[int] = ..., stride: _Optional[int] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    OPCODE_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    LITERAL_FIELD_NUMBER: _ClassVar[int]
    PARAMETER_NUMBER_FIELD_NUMBER: _ClassVar[int]
    TUPLE_INDEX_FIELD_NUMBER: _ClassVar[int]
    WINDOW_FIELD_NUMBER: _ClassVar[int]
    CONVOLUTION_DIMENSION_NUMBERS_FIELD_NUMBER: _ClassVar[int]
    SLICE_DIMENSIONS_FIELD_NUMBER: _ClassVar[int]
    DOT_DIMENSION_NUMBERS_FIELD_NUMBER: _ClassVar[int]
    DIMENSIONS_FIELD_NUMBER: _ClassVar[int]
    GATHER_DIMENSION_NUMBERS_FIELD_NUMBER: _ClassVar[int]
    GATHER_SLICE_SIZES_FIELD_NUMBER: _ClassVar[int]
    INDICES_ARE_SORTED_FIELD_NUMBER: _ClassVar[int]
    UNIQUE_INDICES_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    OPERAND_IDS_FIELD_NUMBER: _ClassVar[int]
    CALLED_COMPUTATION_IDS_FIELD_NUMBER: _ClassVar[int]
    COMPARISON_DIRECTION_FIELD_NUMBER: _ClassVar[int]
    name: str
    opcode: str
    shape: XlaShapeProto
    metadata: XlaOpMetadata
    literal: XlaLiteralProto
    parameter_number: int
    tuple_index: int
    window: XlaWindow
    convolution_dimension_numbers: XlaConvolutionDimensionNumbers
    slice_dimensions: _containers.RepeatedCompositeFieldContainer[HloInstructionProto.SliceDimensions]
    dot_dimension_numbers: XlaDotDimensionNumbers
    dimensions: _containers.RepeatedScalarFieldContainer[int]
    gather_dimension_numbers: XlaGatherDimensionNumbers
    gather_slice_sizes: _containers.RepeatedScalarFieldContainer[int]
    indices_are_sorted: bool
    unique_indices: bool
    id: int
    operand_ids: _containers.RepeatedScalarFieldContainer[int]
    called_computation_ids: _containers.RepeatedScalarFieldContainer[int]
    comparison_direction: str
    def __init__(self, name: _Optional[str] = ..., opcode: _Optional[str] = ..., shape: _Optional[_Union[XlaShapeProto, _Mapping]] = ..., metadata: _Optional[_Union[XlaOpMetadata, _Mapping]] = ..., literal: _Optional[_Union[XlaLiteralProto, _Mapping]] = ..., parameter_number: _Optional[int] = ..., tuple_index: _Optional[int] = ..., window: _Optional[_Union[XlaWindow, _Mapping]] = ..., convolution_dimension_numbers: _Optional[_Union[XlaConvolutionDimensionNumbers, _Mapping]] = ..., slice_dimensions: _Optional[_Iterable[_Union[HloInstructionProto.SliceDimensions, _Mapping]]] = ..., dot_dimension_numbers: _Optional[_Union[XlaDotDimensionNumbers, _Mapping]] = ..., dimensions: _Optional[_Iterable[int]] = ..., gather_dimension_numbers: _Optional[_Union[XlaGatherDimensionNumbers, _Mapping]] = ..., gather_slice_sizes: _Optional[_Iterable[int]] = ..., indices_are_sorted: bool = ..., unique_indices: bool = ..., id: _Optional[int] = ..., operand_ids: _Optional[_Iterable[int]] = ..., called_computation_ids: _Optional[_Iterable[int]] = ..., comparison_direction: _Optional[str] = ...) -> None: ...

class HloComputationProto(_message.Message):
    __slots__ = ("name", "instructions", "program_shape", "id", "root_id")
    NAME_FIELD_NUMBER: _ClassVar[int]
    INSTRUCTIONS_FIELD_NUMBER: _ClassVar[int]
    PROGRAM_SHAPE_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    ROOT_ID_FIELD_NUMBER: _ClassVar[int]
    name: str
    instructions: _containers.RepeatedCompositeFieldContainer[HloInstructionProto]
    program_shape: XlaProgramShapeProto
    id: int
    root_id: int
    def __init__(self, name: _Optional[str] = ..., instructions: _Optional[_Iterable[_Union[HloInstructionProto, _Mapping]]] = ..., program_shape: _Optional[_Union[XlaProgramShapeProto, _Mapping]] = ..., id: _Optional[int] = ..., root_id: _Optional[int] = ...) -> None: ...

class HloModuleProto(_message.Message):
    __slots__ = ("name", "entry_computation_name", "entry_computation_id", "computations", "host_program_shape", "id")
    NAME_FIELD_NUMBER: _ClassVar[int]
    ENTRY_COMPUTATION_NAME_FIELD_NUMBER: _ClassVar[int]
    ENTRY_COMPUTATION_ID_FIELD_NUMBER: _ClassVar[int]
    COMPUTATIONS_FIELD_NUMBER: _ClassVar[int]
    HOST_PROGRAM_SHAPE_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    name: str
    entry_computation_name: str
    entry_computation_id: int
    computations: _containers.RepeatedCompositeFieldContainer[HloComputationProto]
    host_program_shape: XlaProgramShapeProto
    id: int
    def __init__(self, name: _Optional[str] = ..., entry_computation_name: _Optional[str] = ..., entry_computation_id: _Optional[int] = ..., computations: _Optional[_Iterable[_Union[HloComputationProto, _Mapping]]] = ..., host_program_shape: _Optional[_Union[XlaProgramShapeProto, _Mapping]] = ..., id: _Optional[int] = ...) -> None: ...

class OptionOverrideProto(_message.Message):
    __slots__ = ("string_field", "bool_field", "int_field", "double_field")
    STRING_FIELD_FIELD_NUMBER: _ClassVar[int]
    BOOL_FIELD_FIELD_NUMBER: _ClassVar[int]
    INT_FIELD_FIELD_NUMBER: _ClassVar[int]
    DOUBLE_FIELD_FIELD_NUMBER: _ClassVar[int]
    string_field: str
    bool_field: bool
    int_field: int
    double_field: float
    def __init__(self, string_field: _Optional[str] = ..., bool_field: bool = ..., int_field: _Optional[int] = ..., double_field: _Optional[float] = ...) -> None: ...

class CompileEnvOptionProto(_message.Message):
    __slots__ = ("key", "value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    value: OptionOverrideProto
    def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[OptionOverrideProto, _Mapping]] = ...) -> None: ...

class XlaDeviceAssignmentProto(_message.Message):
    __slots__ = ("replica_count", "computation_count", "computation_devices")
    class ComputationDevice(_message.Message):
        __slots__ = ("replica_device_ids",)
        REPLICA_DEVICE_IDS_FIELD_NUMBER: _ClassVar[int]
        replica_device_ids: _containers.RepeatedScalarFieldContainer[int]
        def __init__(self, replica_device_ids: _Optional[_Iterable[int]] = ...) -> None: ...
    REPLICA_COUNT_FIELD_NUMBER: _ClassVar[int]
    COMPUTATION_COUNT_FIELD_NUMBER: _ClassVar[int]
    COMPUTATION_DEVICES_FIELD_NUMBER: _ClassVar[int]
    replica_count: int
    computation_count: int
    computation_devices: _containers.RepeatedCompositeFieldContainer[XlaDeviceAssignmentProto.ComputationDevice]
    def __init__(self, replica_count: _Optional[int] = ..., computation_count: _Optional[int] = ..., computation_devices: _Optional[_Iterable[_Union[XlaDeviceAssignmentProto.ComputationDevice, _Mapping]]] = ...) -> None: ...

class ExecutableBuildOptionsProto(_message.Message):
    __slots__ = ("device_ordinal", "result_layout", "num_replicas", "num_partitions", "use_spmd_partitioning", "use_auto_spmd_partitioning", "deduplicate_hlo", "device_assignment", "alias_passthrough_params", "run_backend_only", "allow_spmd_sharding_propagation_to_output", "fdo_profile", "device_memory_size", "auto_spmd_partitioning_mesh_shape", "auto_spmd_partitioning_mesh_ids")
    DEVICE_ORDINAL_FIELD_NUMBER: _ClassVar[int]
    RESULT_LAYOUT_FIELD_NUMBER: _ClassVar[int]
    NUM_REPLICAS_FIELD_NUMBER: _ClassVar[int]
    NUM_PARTITIONS_FIELD_NUMBER: _ClassVar[int]
    USE_SPMD_PARTITIONING_FIELD_NUMBER: _ClassVar[int]
    USE_AUTO_SPMD_PARTITIONING_FIELD_NUMBER: _ClassVar[int]
    DEDUPLICATE_HLO_FIELD_NUMBER: _ClassVar[int]
    DEVICE_ASSIGNMENT_FIELD_NUMBER: _ClassVar[int]
    ALIAS_PASSTHROUGH_PARAMS_FIELD_NUMBER: _ClassVar[int]
    RUN_BACKEND_ONLY_FIELD_NUMBER: _ClassVar[int]
    ALLOW_SPMD_SHARDING_PROPAGATION_TO_OUTPUT_FIELD_NUMBER: _ClassVar[int]
    FDO_PROFILE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_MEMORY_SIZE_FIELD_NUMBER: _ClassVar[int]
    AUTO_SPMD_PARTITIONING_MESH_SHAPE_FIELD_NUMBER: _ClassVar[int]
    AUTO_SPMD_PARTITIONING_MESH_IDS_FIELD_NUMBER: _ClassVar[int]
    device_ordinal: int
    result_layout: XlaShapeProto
    num_replicas: int
    num_partitions: int
    use_spmd_partitioning: bool
    use_auto_spmd_partitioning: bool
    deduplicate_hlo: bool
    device_assignment: XlaDeviceAssignmentProto
    alias_passthrough_params: bool
    run_backend_only: bool
    allow_spmd_sharding_propagation_to_output: _containers.RepeatedScalarFieldContainer[bool]
    fdo_profile: bytes
    device_memory_size: int
    auto_spmd_partitioning_mesh_shape: _containers.RepeatedScalarFieldContainer[int]
    auto_spmd_partitioning_mesh_ids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, device_ordinal: _Optional[int] = ..., result_layout: _Optional[_Union[XlaShapeProto, _Mapping]] = ..., num_replicas: _Optional[int] = ..., num_partitions: _Optional[int] = ..., use_spmd_partitioning: bool = ..., use_auto_spmd_partitioning: bool = ..., deduplicate_hlo: bool = ..., device_assignment: _Optional[_Union[XlaDeviceAssignmentProto, _Mapping]] = ..., alias_passthrough_params: bool = ..., run_backend_only: bool = ..., allow_spmd_sharding_propagation_to_output: _Optional[_Iterable[bool]] = ..., fdo_profile: _Optional[bytes] = ..., device_memory_size: _Optional[int] = ..., auto_spmd_partitioning_mesh_shape: _Optional[_Iterable[int]] = ..., auto_spmd_partitioning_mesh_ids: _Optional[_Iterable[int]] = ...) -> None: ...

class CompileOptionsProto(_message.Message):
    __slots__ = ("argument_layouts", "parameter_is_tupled_arguments", "executable_build_options", "compile_portable_executable", "profile_version", "serialized_multi_slice_config", "env_options")
    ARGUMENT_LAYOUTS_FIELD_NUMBER: _ClassVar[int]
    PARAMETER_IS_TUPLED_ARGUMENTS_FIELD_NUMBER: _ClassVar[int]
    EXECUTABLE_BUILD_OPTIONS_FIELD_NUMBER: _ClassVar[int]
    COMPILE_PORTABLE_EXECUTABLE_FIELD_NUMBER: _ClassVar[int]
    PROFILE_VERSION_FIELD_NUMBER: _ClassVar[int]
    SERIALIZED_MULTI_SLICE_CONFIG_FIELD_NUMBER: _ClassVar[int]
    ENV_OPTIONS_FIELD_NUMBER: _ClassVar[int]
    argument_layouts: _containers.RepeatedCompositeFieldContainer[XlaShapeProto]
    parameter_is_tupled_arguments: bool
    executable_build_options: ExecutableBuildOptionsProto
    compile_portable_executable: bool
    profile_version: int
    serialized_multi_slice_config: bytes
    env_options: _containers.RepeatedCompositeFieldContainer[CompileEnvOptionProto]
    def __init__(self, argument_layouts: _Optional[_Iterable[_Union[XlaShapeProto, _Mapping]]] = ..., parameter_is_tupled_arguments: bool = ..., executable_build_options: _Optional[_Union[ExecutableBuildOptionsProto, _Mapping]] = ..., compile_portable_executable: bool = ..., profile_version: _Optional[int] = ..., serialized_multi_slice_config: _Optional[bytes] = ..., env_options: _Optional[_Iterable[_Union[CompileEnvOptionProto, _Mapping]]] = ...) -> None: ...
