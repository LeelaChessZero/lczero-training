project(
  'ice_skate_loader',
  'cpp',
  version : '0.1',
  meson_version : '>= 1.3.0',
  default_options : [
    'warning_level=3',
    'cpp_std=c++20',
    'werror=true',
  ],
)

add_project_arguments('-Wno-nullability-extension', language : 'cpp')
add_project_arguments('-Werror', language : 'cpp')

libarchive_dep = dependency('libarchive')
zlib_dep = dependency('zlib')
absl_log_dep = dependency(
  'absl_log',
).as_system()
absl_log_initialize_dep = dependency(
  'absl_log_initialize',
).as_system()
absl_check_dep = dependency(
  'absl_check',
).as_system()
absl_hash_dep = dependency(
  'absl_hash',
).as_system()
absl_container_dep = dependency(
  'absl_raw_hash_set',
).as_system()
absl_synchronization_dep = dependency(
  'absl_synchronization',
).as_system()
absl_random_dep = dependency(
  'absl_random_random',
).as_system()
gtest_dep = dependency(
  'gtest',
).as_system()
gtest_main_dep = dependency(
  'gtest_main',
).as_system()

inc = include_directories('src')

files = [
  'src/chunk_feed/discovery.cc',
  'src/data_loader.cc',
  'src/misc/stream_shuffler.cc',
  'src/reader/gz.cc',
  'src/chunk_feed/tar_chunk_source.cc',
]

loader_lib = static_library(
  'loader',
  files,
  include_directories : inc,
  dependencies : [
    libarchive_dep,
    zlib_dep,
    absl_log_dep,
    absl_check_dep,
    absl_hash_dep,
    absl_container_dep,
    absl_synchronization_dep,
    absl_random_dep,
  ],
)

exe = executable(
  'loader',
  'src/loader_main.cpp',
  dependencies : [
    libarchive_dep,
    zlib_dep,
    absl_log_dep,
    absl_log_initialize_dep,
    absl_check_dep,
    absl_hash_dep,
    absl_container_dep,
    absl_synchronization_dep,
    absl_random_dep,
  ],
  link_with : loader_lib,
)

stream_shuffler_test = executable(
  'stream_shuffler_test',
  'src/misc/stream_shuffler_test.cc',
  include_directories : inc,
  dependencies : [
    gtest_dep,
    gtest_main_dep,
    absl_random_dep,
  ],
  link_with : loader_lib,
)

test('stream_shuffler', stream_shuffler_test)

discovery_main = executable(
  'discovery_main',
  'src/chunk_feed/discovery_main.cc',
  include_directories : inc,
  dependencies : [
    absl_log_dep,
    absl_log_initialize_dep,
  ],
  link_with : loader_lib,
)
