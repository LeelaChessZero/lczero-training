# Overview, Glossary, and File Formats

This document serves as a glossary of terms used in the project and describes
the file formats utilized.

* Training data **frame** is a data structure that holds information needed for
  the NN training about a single chess position. Currently it's a fixed sized
  struct, e.g. [V6TrainingData](../libs/lc0/src/trainingdata/trainingdata.h).
* **Chunk** is a sequence of frames from a single game. Currently, they are
  stored in a gzipped file where frames are concatenated together.
* **Chunk Source** is a file that contains one or more chunks. In older versions
  of the code, it was a single gzipped file. In the new version, it also may be
  an uncompressed .tar file.
