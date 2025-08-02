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
absl_flags_dep = dependency(
  'absl_flags',
).as_system()
absl_flags_parse_dep = dependency(
  'absl_flags_parse',
).as_system()
gtest_dep = dependency(
  'gtest',
).as_system()
gtest_main_dep = dependency(
  'gtest_main',
).as_system()

includes = include_directories('src', 'libs/lc0/src')

files = [
  'src/loader/chunk_feed/chunk_set.cc',
  'src/loader/chunk_feed/discovery.cc',
  'src/loader/chunk_feed/rawfile_chunk_source.cc',
  'src/loader/chunk_feed/tar_chunk_source.cc',
  'src/loader/data_loader.cc',
  'src/utils/gz.cc',
  'src/utils/stream_shuffler.cc',
  'libs/lc0/src/utils/files.cc',
]

loader_lib = static_library(
  'loader',
  files,
  include_directories : includes,
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
  'src/loader/loader_main.cpp',
  include_directories : includes,
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
  'src/utils/stream_shuffler_test.cc',
  include_directories : includes,
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
  'src/loader/chunk_feed/discovery_main.cc',
  include_directories : includes,
  dependencies : [
    absl_log_dep,
    absl_log_initialize_dep,
    absl_flags_dep,
    absl_flags_parse_dep,
  ],
  link_with : loader_lib,
)
