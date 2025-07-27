project(
  'ice_skate_loader',
  'cpp',
  version : '0.1',
  meson_version : '>= 1.3.0',
  default_options : ['warning_level=3', 'cpp_std=c++20', 'werror=true'],
)

libarchive_dep = dependency('libarchive')
zlib_dep = dependency('zlib')
absl_log_dep = dependency('absl_log', fallback: ['abseil-cpp', 'absl_log_dep']).as_system()
absl_log_initialize_dep = dependency('absl_log_initialize', fallback: ['abseil-cpp', 'absl_log_initialize_dep']).as_system()
absl_hash_dep = dependency('absl_hash', fallback: ['abseil-cpp', 'absl_hash_dep']).as_system()
absl_container_dep = dependency('absl_raw_hash_set', fallback: ['abseil-cpp', 'absl_raw_hash_set_dep']).as_system()
absl_synchronization_dep = dependency('absl_synchronization', fallback: ['abseil-cpp', 'absl_synchronization_dep']).as_system()
absl_random_dep = dependency('absl_random_random', fallback: ['abseil-cpp', 'absl_random_random_dep']).as_system()
gtest_dep = dependency('gtest', fallback: ['gtest', 'gtest_dep']).as_system()
gtest_main_dep = dependency('gtest_main', fallback: ['gtest', 'gtest_main_dep']).as_system()

inc = include_directories('src')

files = [
  'src/reader/tar.cc',
  'src/reader/gz.cc',
  'src/chunk_feed/discovery.cc',
  'src/misc/stream_shuffler.cc',
]

loader_lib = static_library(
  'loader',
  files,
  include_directories : inc,
  dependencies : [libarchive_dep, zlib_dep, absl_log_dep, absl_hash_dep, absl_container_dep, absl_synchronization_dep, absl_random_dep],
  cpp_args : ['-Werror'],
)

exe = executable(
  'loader',
  'src/loader_main.cpp',
  install : true,
  dependencies : [],
  link_with : loader_lib,
  cpp_args : ['-Werror'],
)

stream_shuffler_test = executable(
  'stream_shuffler_test',
  'src/misc/stream_shuffler_test.cc',
  include_directories : inc,
  dependencies : [gtest_dep, gtest_main_dep, absl_random_dep],
  link_with : loader_lib,
  cpp_args : ['-Werror'],
)

test('stream_shuffler', stream_shuffler_test)

discovery_main = executable(
  'discovery_main',
  'src/chunk_feed/discovery_main.cc',
  include_directories : inc,
  dependencies : [absl_log_dep, absl_log_initialize_dep],
  link_with : loader_lib,
  cpp_args : ['-Werror'],
)
