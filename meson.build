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
libarchive_dep = dependency('libarchive')
zlib_dep = dependency('zlib')

# Abseil dependencies
absl_deps = {}
foreach name : [
  'log', 'log_initialize', 'check', 'hash', 'raw_hash_set', 
  'synchronization', 'random_random', 'flags', 'flags_parse', 'throw_delegate'
]
  absl_deps += {name.underscorify() : dependency('absl_' + name).as_system()}
endforeach

# Test dependencies
gtest_dep = dependency('gtest').as_system()
gtest_main_dep = dependency('gtest_main').as_system()

# Common dependency sets
external_deps = [libarchive_dep, zlib_dep]
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
test_deps = [gtest_dep, gtest_main_dep]
cli_deps = [
  absl_deps['log'],
  absl_deps['log_initialize'],
  absl_deps['flags'],
  absl_deps['flags_parse'],
]

includes = include_directories('src', 'libs/lc0/src')

files = [
  'src/loader/chunk_feed/shuffling_chunk_pool.cc',
  'src/loader/chunk_feed/chunk_source_loader.cc',
  'src/loader/chunk_feed/chunk_unpacker.cc',
  'src/loader/chunk_feed/file_path_provider.cc',
  'src/loader/chunk_feed/rawfile_chunk_source.cc',
  'src/loader/chunk_feed/tar_chunk_source.cc',
  'src/loader/shuffling_frame_sampler.cc',
  'src/loader/tensor_generator.cc',
  'src/loader/data_loader.cc',
  'src/utils/gz.cc',
  'src/utils/stream_shuffler.cc',
  'libs/lc0/src/utils/files.cc',
  'libs/lc0/src/utils/logging.cc',
]

loader_lib = static_library(
  'loader',
  files,
  include_directories : includes,
  dependencies : loader_deps,
)

exe = executable(
  'loader',
  'src/loader/loader_main.cpp',
  include_directories : includes,
  dependencies : cli_deps,
  link_with : loader_lib,
)

stream_shuffler_test = executable(
  'stream_shuffler_test',
  'src/utils/stream_shuffler_test.cc',
  include_directories : includes,
  dependencies : test_deps + [absl_deps['random_random']],
  link_with : loader_lib,
)

queue_test = executable(
  'queue_test',
  'src/utils/queue_test.cc',
  include_directories : includes,
  dependencies : test_deps + [absl_deps['synchronization']],
)

file_path_provider_test = executable(
  'file_path_provider_test',
  'src/loader/chunk_feed/file_path_provider_test.cc',
  include_directories : includes,
  dependencies : test_deps + [absl_deps['synchronization'], absl_deps['log']],
  link_with : loader_lib,
)

chunk_source_loader_test = executable(
  'chunk_source_loader_test',
  'src/loader/chunk_feed/chunk_source_loader_test.cc',
  include_directories : includes,
  dependencies : test_deps + [absl_deps['synchronization']],
  link_with : loader_lib,
)

shuffling_chunk_pool_test = executable(
  'shuffling_chunk_pool_test',
  'src/loader/chunk_feed/shuffling_chunk_pool_test.cc',
  include_directories : includes,
  dependencies : test_deps + [absl_deps['synchronization'], absl_deps['log']],
  link_with : loader_lib,
)

chunk_unpacker_test = executable(
  'chunk_unpacker_test',
  'src/loader/chunk_feed/chunk_unpacker_test.cc',
  include_directories : includes,
  dependencies : test_deps + [absl_deps['synchronization'], absl_deps['log']],
  link_with : loader_lib,
)

shuffling_frame_sampler_test = executable(
  'shuffling_frame_sampler_test',
  'src/loader/shuffling_frame_sampler_test.cc',
  include_directories : includes,
  dependencies : test_deps + [absl_deps['synchronization'], absl_deps['random_random']],
  link_with : loader_lib,
)

tensor_test = executable(
  'tensor_test',
  'src/utils/tensor_test.cc',
  include_directories : includes,
  dependencies : test_deps + [absl_deps['throw_delegate']],
)

tensor_generator_test = executable(
  'tensor_generator_test',
  'src/loader/tensor_generator_test.cc',
  include_directories : includes,
  dependencies : test_deps + [absl_deps['synchronization']],
  link_with : loader_lib,
)

stats_test = executable(
  'stats_test',
  'src/utils/metrics/stats_test.cc',
  include_directories : includes,
  dependencies : test_deps + [absl_deps['synchronization']],
)

load_metric_test = executable(
  'load_metric_test',
  'src/utils/metrics/load_metric_test.cc',
  include_directories : includes,
  dependencies : test_deps + [absl_deps['synchronization']],
)
test('stream_shuffler_test', stream_shuffler_test)
test('queue_test', queue_test)
test('file_path_provider_test', file_path_provider_test)
test('chunk_source_loader_test', chunk_source_loader_test)
test('shuffling_chunk_pool_test', shuffling_chunk_pool_test)
test('chunk_unpacker_test', chunk_unpacker_test)
test('shuffling_frame_sampler_test', shuffling_frame_sampler_test)
test('tensor_test', tensor_test)
test('tensor_generator_test', tensor_generator_test)
test('stats_test', stats_test)
test('load_metric_test', load_metric_test)

file_path_provider_main = executable(
  'file_path_provider_main',
  'src/loader/chunk_feed/file_path_provider_main.cc',
  include_directories : includes,
  dependencies : cli_deps,
  link_with : loader_lib,
)
