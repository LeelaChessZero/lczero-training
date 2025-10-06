project(
  'lczero-training',
  'cpp',
  version : '0.1',
  meson_version : '>= 1.3.0',
  default_options : [
    'warning_level=3',
    'cpp_std=c++20',
    'werror=true',
  ],
)

# Allow Clang nullability extensions when using Clang compiler
cpp_compiler = meson.get_compiler('cpp')
if cpp_compiler.get_id() == 'clang'
  add_project_arguments('-Wno-nullability-extension', language : 'cpp')
endif
add_project_arguments('-Werror', language : 'cpp')


# External dependencies
zlib_dep = dependency('zlib')

# Python and PyBind11 dependencies for Python extension
python3 = import('python').find_installation()
pybind11_dep = dependency('pybind11')

# Abseil dependencies always resolved from the wrap subproject.
absl_proj = subproject('abseil-cpp')
absl_dep_specs = [
  ['log', 'absl_log_dep'],
  ['log_initialize', 'absl_log_dep'],
  ['check', 'absl_log_dep'],
  ['hash', 'absl_hash_dep'],
  ['raw_hash_set', 'absl_container_dep'],
  ['synchronization', 'absl_synchronization_dep'],
  ['random_random', 'absl_random_dep'],
  ['flags', 'absl_flags_dep'],
  ['flags_parse', 'absl_flags_dep'],
  ['throw_delegate', 'absl_base_dep'],
]
absl_deps = {}
foreach spec : absl_dep_specs
  absl_deps += {
    spec[0].underscorify() : absl_proj.get_variable(spec[1]).as_system()
  }
endforeach

# Test dependencies
gtest_dep = dependency('gtest').as_system()
gtest_main_dep = dependency('gtest_main').as_system()

# Common dependency sets
external_deps = [zlib_dep]
core_absl_deps = [
  absl_deps['log'],
  absl_deps['check'],
  absl_deps['hash'],
  absl_deps['raw_hash_set'],
  absl_deps['synchronization'],
  absl_deps['random_random'],
]
loader_deps = external_deps + core_absl_deps
main_deps = loader_deps + [absl_deps['log_initialize']]
# test_deps will be defined after proto_dep
cli_deps = [
  absl_deps['log'],
  absl_deps['log_initialize'],
  absl_deps['flags'],
  absl_deps['flags_parse'],
]

# Protobuf compilation setup
compile_proto = find_program('libs/lc0/scripts/compile_proto.py')
proto_gen = generator(compile_proto, output: ['@BASENAME@.pb.h'],
  arguments : [
    '--proto_path=@CURRENT_SOURCE_DIR@',
    '--cpp_out=@BUILD_DIR@',
    '@INPUT@'])

lc0_proto_gen = generator(compile_proto, output: ['@BASENAME@.pb.h'],
  arguments : [
    '--proto_path=' + join_paths(meson.current_source_dir(), 'libs/lc0'),
    '--cpp_out=@BUILD_DIR@',
    '@INPUT@'])

includes = include_directories('csrc', 'libs/lc0/src')

add_project_arguments('-DNO_PEXT', language : 'cpp')

rescorer_files = [
  'libs/lc0/src/chess/board.cc',
  'libs/lc0/src/chess/gamestate.cc',
  'libs/lc0/src/chess/position.cc',
  'libs/lc0/src/neural/decoder.cc',
  'libs/lc0/src/neural/encoder.cc',
  'libs/lc0/src/trainingdata/reader.cc',
  'libs/lc0/src/trainingdata/trainingdata.cc',
  'libs/lc0/src/trainingdata/writer.cc',
  'libs/lc0/src/utils/commandline.cc',
  'libs/lc0/src/utils/configfile.cc',
  'libs/lc0/src/utils/esc_codes.cc',
  'libs/lc0/src/utils/optionsdict.cc',
  'libs/lc0/src/utils/optionsparser.cc',
  'libs/lc0/src/utils/random.cc',
  'libs/lc0/src/utils/string.cc',
]

if host_machine.system() == 'windows'
  rescorer_files += 'libs/lc0/src/utils/filesystem.win32.cc'
else
  rescorer_files += 'libs/lc0/src/utils/filesystem.posix.cc'
endif

files = [
  'csrc/loader/stages/shuffling_chunk_pool.cc',
  'csrc/loader/stages/chunk_source_loader.cc',
  'csrc/loader/stages/chunk_rescorer.cc',
  'csrc/loader/stages/chunk_unpacker.cc',
  'csrc/loader/stages/file_path_provider.cc',
  'csrc/loader/stages/stage_factory.cc',
  'csrc/loader/chunk_source/debug_chunk_source.cc',
  'csrc/loader/chunk_source/rawfile_chunk_source.cc',
  'csrc/loader/chunk_source/tar_chunk_source.cc',
  'csrc/loader/stages/shuffling_frame_sampler.cc',
  'csrc/loader/stages/tensor_generator.cc',
  'csrc/loader/data_loader.cc',
  'csrc/loader/data_loader_metrics.cc',
  'csrc/utils/gz.cc',
  'csrc/utils/stream_shuffler.cc',
  'libs/lc0/src/syzygy/syzygy.cc',
  'libs/lc0/src/trainingdata/rescorer.cc',
  'libs/lc0/src/utils/files.cc',
  'libs/lc0/src/utils/logging.cc',
  'libs/lc0/src/utils/protomessage.cc',
] + rescorer_files

# Process protobuf files for C++
proto_files = [
  proto_gen.process('proto/data_loader_config.proto',
    preserve_path_from : meson.current_source_dir()),
  proto_gen.process('proto/training_metrics.proto',
    preserve_path_from : meson.current_source_dir()),
  proto_gen.process('proto/stage_control.proto',
    preserve_path_from : meson.current_source_dir()),
  lc0_proto_gen.process('libs/lc0/proto/net.proto',
    preserve_path_from : join_paths(meson.current_source_dir(), 'libs/lc0')),
  lc0_proto_gen.process('libs/lc0/proto/onnx.proto',
    preserve_path_from : join_paths(meson.current_source_dir(), 'libs/lc0')),
  lc0_proto_gen.process('libs/lc0/proto/hlo.proto',
    preserve_path_from : join_paths(meson.current_source_dir(), 'libs/lc0'))
]

# Create a dependency for protobuf files that tests can use
proto_dep = declare_dependency(sources: proto_files)
test_deps = [gtest_dep, gtest_main_dep, proto_dep]


files += proto_files

loader_lib = static_library(
  'loader',
  files,
  include_directories : includes,
  dependencies : loader_deps,
)

exe = executable(
  'loader',
  'csrc/loader/loader_main.cpp',
  include_directories : includes,
  dependencies : cli_deps + [proto_dep],
  link_with : loader_lib,
)

stream_shuffler_test = executable(
  'stream_shuffler_test',
  'csrc/utils/stream_shuffler_test.cc',
  include_directories : includes,
  dependencies : test_deps + [absl_deps['random_random']],
  link_with : loader_lib,
)

queue_test = executable(
  'queue_test',
  'csrc/utils/queue_test.cc',
  include_directories : includes,
  dependencies : test_deps + [absl_deps['synchronization'], absl_deps['log']],
)

file_path_provider_test = executable(
  'file_path_provider_test',
  'csrc/loader/stages/file_path_provider_test.cc',
  include_directories : includes,
  dependencies : test_deps + [absl_deps['synchronization'], absl_deps['log']],
  link_with : loader_lib,
)

chunk_source_loader_test = executable(
  'chunk_source_loader_test',
  'csrc/loader/stages/chunk_source_loader_test.cc',
  include_directories : includes,
  dependencies : test_deps + [absl_deps['synchronization']],
  link_with : loader_lib,
)

shuffling_chunk_pool_test = executable(
  'shuffling_chunk_pool_test',
  'csrc/loader/stages/shuffling_chunk_pool_test.cc',
  include_directories : includes,
  dependencies : test_deps + [absl_deps['synchronization'], absl_deps['log']],
  link_with : loader_lib,
)

chunk_rescorer_test = executable(
  'chunk_rescorer_test',
  'csrc/loader/stages/chunk_rescorer_test.cc',
  include_directories : includes,
  dependencies : test_deps + [absl_deps['synchronization'], absl_deps['log']],
  link_with : loader_lib,
)

chunk_unpacker_test = executable(
  'chunk_unpacker_test',
  'csrc/loader/stages/chunk_unpacker_test.cc',
  include_directories : includes,
  dependencies : test_deps + [absl_deps['synchronization'], absl_deps['log']],
  link_with : loader_lib,
)

shuffling_frame_sampler_test = executable(
  'shuffling_frame_sampler_test',
  'csrc/loader/stages/shuffling_frame_sampler_test.cc',
  include_directories : includes,
  dependencies : test_deps + [absl_deps['synchronization'], absl_deps['random_random']],
  link_with : loader_lib,
)

tensor_test = executable(
  'tensor_test',
  'csrc/utils/tensor_test.cc',
  include_directories : includes,
  dependencies : test_deps + [absl_deps['throw_delegate']],
)

tensor_generator_test = executable(
  'tensor_generator_test',
  'csrc/loader/stages/tensor_generator_test.cc',
  include_directories : includes,
  dependencies : test_deps + [absl_deps['synchronization']],
  link_with : loader_lib,
)

stats_test = executable(
  'stats_test',
  'csrc/utils/metrics/stats_test.cc',
  include_directories : includes,
  dependencies : test_deps + [absl_deps['synchronization']],
  link_with : loader_lib,
)

load_metric_test = executable(
  'load_metric_test',
  'csrc/utils/metrics/load_metric_test.cc',
  include_directories : includes,
  dependencies : test_deps + [absl_deps['synchronization']],
  link_with : loader_lib,
)

stage_factory_test = executable(
  'stage_factory_test',
  'csrc/loader/stages/stage_factory_test.cc',
  include_directories : includes,
  dependencies : test_deps + [absl_deps['synchronization'], absl_deps['log']],
  link_with : loader_lib,
)

data_loader_test = executable(
  'data_loader_test',
  'csrc/loader/data_loader_test.cc',
  include_directories : includes,
  dependencies : test_deps + [absl_deps['synchronization'], absl_deps['log']],
  link_with : loader_lib,
)
test('stream_shuffler_test', stream_shuffler_test)
test('queue_test', queue_test)
test('file_path_provider_test', file_path_provider_test)
test('chunk_source_loader_test', chunk_source_loader_test)
test('shuffling_chunk_pool_test', shuffling_chunk_pool_test)
test('chunk_rescorer_test', chunk_rescorer_test)
test('chunk_unpacker_test', chunk_unpacker_test)
test('shuffling_frame_sampler_test', shuffling_frame_sampler_test)
test('tensor_test', tensor_test)
test('tensor_generator_test', tensor_generator_test)
test('stats_test', stats_test)
test('load_metric_test', load_metric_test)
test('stage_factory_test', stage_factory_test)
test('data_loader_test', data_loader_test)

file_path_provider_main = executable(
  'file_path_provider_main',
  'csrc/loader/stages/file_path_provider_main.cc',
  include_directories : includes,
  dependencies : cli_deps + [proto_dep],
  link_with : loader_lib,
)

dump_chunk = executable(
  'dump_chunk',
  'csrc/tools/dump_chunk_main.cc',
  include_directories : includes,
  dependencies : cli_deps + [zlib_dep],
)

startpos_policy_distribution = executable(
  'startpos_policy_distribution',
  'csrc/tools/startpos_policy_distribution_main.cc',
  include_directories : includes,
  dependencies : cli_deps,
  link_with : loader_lib,
)

# Python extension module
python3.extension_module(
  '_lczero_training',
  'csrc/loader/pybind_module.cc',
  include_directories : includes,
  dependencies : [pybind11_dep, proto_dep, absl_deps['log_initialize']] + loader_deps,
  link_with : loader_lib,
  install : true,
  subdir : 'lczero_training',
)
