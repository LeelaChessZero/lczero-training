using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;

namespace TMDataShuffler
{
    class Program
    {
        static void Main(string[] args)
        {
            Random rnd = new Random();
            const int record_size = 85 * sizeof(float);
            const int slices = 100;
            using (var stream = File.OpenRead(args[0]))
            using (var inputGZipStream = new GZipStream(stream, CompressionMode.Decompress))
            {
                List<FileStream> streams = new List<FileStream>();
                List<GZipStream> gzs = new List<GZipStream>();
                for (int i = 0; i < slices; i++)
                {
                    streams.Add(File.OpenWrite(args[0] + $".{i}"));
                    gzs.Add(new GZipStream(streams.Last(), CompressionLevel.Fastest));
                }
                byte[] dataChunk = new byte[record_size];
                while (true)
                {
                    int readTotal = inputGZipStream.Read(dataChunk);
                    if (readTotal <= 0) break;
                    if (readTotal != record_size) throw new Exception("Bah!");
                    gzs[rnd.Next(slices)].Write(dataChunk);
                }
                for (int i = 0; i < slices; i++)
                {
                    gzs[i].Close();
                    streams[i].Close();
                }
            }
            using (var stream = File.OpenWrite(args[0]+".new"))
            using (var outputGZipStream = new GZipStream(stream, CompressionLevel.Fastest))
            {
                for (int i = 0; i < slices; i++)
                {
                    using var istream = File.OpenRead(args[0] + $".{i}");
                    using var gzs = new GZipStream(istream, CompressionMode.Decompress);
                    byte[] dataChunk = new byte[record_size];
                    while (true)
                    {
                        int readTotal = gzs.Read(dataChunk);
                        if (readTotal <= 0) break;
                        if (readTotal != record_size) throw new Exception("Bah!");
                        outputGZipStream.Write(dataChunk);
                    }
                }
            }

        }
    }
}
