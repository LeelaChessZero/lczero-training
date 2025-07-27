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

inc = include_directories('src')

files = [
  'src/reader/tar.cc',
  'src/reader/gz.cc',
  'src/misc/stream_shuffler.cc',
]

loader_lib = static_library(
  'loader',
  files,
  include_directories : inc,
  dependencies : [libarchive_dep, zlib_dep, absl_log_dep],
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

test('basic', exe)
