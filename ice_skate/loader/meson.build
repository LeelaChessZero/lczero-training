project(
  'ice_skate_loader',
  'cpp',
  version : '0.1',
  meson_version : '>= 1.3.0',
  default_options : ['warning_level=3', 'cpp_std=c++20'],
)

libarchive_dep = dependency('libarchive')
zlib_dep = dependency('zlib')
absl_log_dep = dependency('absl_log', fallback: ['abseil-cpp', 'absl_log_dep']).as_system()

files = [
  'tar.cc',
  'gz.cc',
]

tar_lib = static_library(
  'tar',
  files,
  dependencies : [libarchive_dep, zlib_dep, absl_log_dep],
)

exe = executable(
  'loader',
  'loader.cpp',
  install : true,
  dependencies : [],
  link_with : tar_lib,
)

test('basic', exe)
