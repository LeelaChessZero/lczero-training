using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using Microsoft.VisualBasic;

namespace TMDataProcessor
{
    class Counter
    {
        public long time;
        public long offset_time;
        public float[] p;
        public float[] s;
        public int[] n;

        public bool IsEmpty => p[0] == 0.0 && n[0] == 0;

        public void Read(BinaryReader reader)
        {
            time = reader.ReadInt64();
            offset_time = reader.ReadInt64();
            p = new float[256];
            for (int i = 0; i < 256; i++)
            {
                p[i] = reader.ReadSingle();
            }
            s = new float[256];
            for (int i = 0; i < 256; i++)
            {
                s[i] = reader.ReadSingle();
            }
            n = new int[256];
            for (int i = 0; i < 256; i++)
            {
                n[i] = (int)reader.ReadUInt32();
            }
        }
    }

    class TD
    {
        public float[] n_part;
        public float[] ns_part;
        public float[] s_part;
        public float[] p_part;
        public float fraction;
        public float scale;
        public float ratio;
        public float target;
        public float dqtarget;

        public void Write(BinaryWriter writer)
        {
            for (int i = 0; i < n_part.Length; i++)
            {
                writer.Write(n_part[i]);
            }
            for (int i = 0; i < ns_part.Length; i++)
            {
                writer.Write(ns_part[i]);
            }
            for (int i = 0; i < s_part.Length; i++)
            {
                writer.Write(s_part[i]);
            }
            for (int i = 0; i < p_part.Length; i++)
            {
                writer.Write(p_part[i]);
            }
            writer.Write(fraction);
            writer.Write(scale);
            writer.Write(ratio);
            writer.Write(target);
            writer.Write(dqtarget);
        }
    }
    class Program
    {
        static void Main(string[] args)
        {
            var searches = GenerateSearches(args);

            GenerateTD(searches);
            //Process(searches);
        }

        private static IEnumerable<List<Counter>> GenerateSearches(string[] args)
        {
            var cur = new List<Counter>();
            for (int i = 0; i < args.Length; i++)
            {
                using (var stream = File.OpenRead(args[i]))
                using (var gzs = new GZipStream(stream, CompressionMode.Decompress))
                {
                    byte[] dataChunk = new byte[3088];
                    while (true)
                    {
                        int readTotal = gzs.Read(dataChunk);
                        if (readTotal <= 0) break;
                        if (readTotal != 3088) throw new Exception("Bah!");
                        using (var memStream = new MemoryStream(dataChunk))
                        using (var reader = new BinaryReader(memStream))
                        {
                            Counter next = new Counter();
                            next.Read(reader);
                            if (next.IsEmpty)
                            {
                                yield return cur;
                                cur = new List<Counter>();
                            }
                            else
                            {
                                cur.Add(next);
                            }
                        }
                    }
                }
            }

            yield return cur;
        }

        private static void GenerateTD(IEnumerable<List<Counter>> searches)
        {
            const int nPartSize = 20;
            List<TD> pending = new List<TD>();
            Random rnd = new Random();
            const int mixSize = 10000000;
            const int backLook = 4;
            int falsePositives = 0;
            int truePositives = 0;
            int falseNegatives = 0;
            int trueNegatives = 0;
            double sLoss = 0.0;
            double error = 0.0;
            if (File.Exists("td.dat")) File.Delete("td.dat");
            using var stream = File.OpenWrite("td.dat");
            using var gzs = new GZipStream(stream, CompressionLevel.Fastest);

            foreach (var search in searches)
            {
                if (search.Count < 4 + backLook) continue;
                if (search.Last().offset_time < 400) continue;
                int firstLimit = -1;
                int secondLimit = -1;
                for (int i = 0; i < search.Count; i++)
                {
                    if (i > backLook && firstLimit == -1 && search[i].offset_time >= 200)
                    {
                        firstLimit = i;
                    }

                    if (search[i].offset_time >= 400)
                    {
                        secondLimit = i;
                        break;
                    }
                }

                const int downsample = 2;
                for (int i = 0; i < search.Count / downsample; i++)
                {
                    int second = search.Count - 1 - rnd.Next(search.Count - secondLimit);
                    double ratio = Math.Pow(rnd.NextDouble(), 2.0);
                    int first = (int)(ratio * (second - 1) + (1 - ratio) * firstLimit);
                    if (search[first].offset_time >= search[second].offset_time) continue;
                    //rnd.Next(second - 1 - backLook) + backLook;
                    int finalMaxN = search[second].n.Max();
                    int finalSumN = search[second].n.Sum();
                    if (finalSumN == 0) continue;
                    int finalIdx = search[second].n.ToList().IndexOf(finalMaxN);
                    int startMaxN = search[first].n.Max();
                    int startSumN = search[first].n.Sum();
                    int startIdx = search[first].n.ToList().IndexOf(startMaxN);
                    int zeroLimit = -1;
                    for (int j = first; j >= 0; j--)
                    {
                        if (search[j].offset_time <= search[first].offset_time - 200 && search[j].n.Sum() < startSumN)
                        {
                            zeroLimit = j;
                            break;
                        }
                    }
                    if (zeroLimit < 0) continue;
                    int zeroth = rnd.Next(zeroLimit+1);
                    int zerothSumN = search[zeroth].n.Sum();
                    // Downsample cases where we can safely abort, as they are too frequent.
                    //if (startIdx == finalIdx && rnd.Next(10) != 0) continue;

                    int secondBest = 0;
                    bool seenMax = false;
                    for (int j = 0; j < 256; j++)
                    {
                        if (search[first].n[j] == startMaxN)
                        {
                            if (seenMax)
                            {
                                secondBest = startMaxN;
                            }

                            seenMax = true;
                        }
                        else if (search[first].n[j] > secondBest)
                        {
                            secondBest = search[first].n[j];
                        }
                    }

                    long estTime = search[second].offset_time - search[first].offset_time;
                    long estBasisTime = search[first].offset_time - search[zeroth].offset_time;
                    double nps = (double) (startSumN - zerothSumN) / estBasisTime;
                    if (nps < 0.1)
                    {
                        // Less than 100 nps - on an A100 - probably bad data from before tcmalloc enabled..
                        Console.Out.WriteLine(nps);
                        continue;
                    }

                    if (estTime * nps < 1)
                    {
                        // Avoid divide by 0. With the nps limit this should be rare.
                        continue;
                    }

                    int estNodes = (int) (startSumN + (double)estTime * (startSumN-zerothSumN)/ estBasisTime);
                    float threshold = (estNodes - startSumN) / 2.3f;
                    if (secondBest + threshold < startMaxN)
                    {
                        if (startIdx == finalIdx)
                        {
                            truePositives++;
                        }
                        else
                        {
                            falsePositives++;
                            sLoss += search[second].s[startIdx] - search[second].s[finalIdx];
                            error += Math.Pow(search[second].s[startIdx] - search[second].s[finalIdx], 2.0);
                        }
                    }
                    else
                    {
                        if (startIdx == finalIdx)
                        {
                            falseNegatives++;
                            error += Math.Pow(0.01f * (1.0f - (float)startSumN / finalSumN), 2.0);
                        }
                        else
                        {
                            trueNegatives++;
                        }

                    }

                    int[] ordering = new int[search[first].n.Length];
                    for (int j = 0; j < ordering.Length; j++)
                    {
                        ordering[j] = j;
                    }
                    int[] raw = (int[])search[first].n.Clone();
                    Array.Sort(raw, ordering);
                    Array.Reverse(ordering);

                    TD data = new TD();
                    int toCopy = Math.Min(nPartSize, search[first].n.Length);
                    data.n_part = new float[nPartSize];
                    int div = Math.Max(startSumN, 1);
                    for (int j = 0; j < toCopy; j++)
                    {
                        data.n_part[j] = (float) search[first].n[ordering[j]] / div;
                    }

                    toCopy = Math.Min(nPartSize, search[first].s.Length);
                    data.s_part = new float[nPartSize];
                    for (int j = 0; j < toCopy; j++)
                    {
                        data.s_part[j] = search[first].s[ordering[j]];
                    }

                    toCopy = Math.Min(nPartSize, search[first].p.Length);
                    data.p_part = new float[nPartSize];
                    for (int j = 0; j < toCopy; j++)
                    {
                        data.p_part[j] = search[first].p[ordering[j]];
                    }

                    toCopy = Math.Min(nPartSize, search[first].n.Length);
                    data.ns_part = new float[nPartSize];
                    //int div2 = Math.Max(search[first- backLook].n.Sum(), 1);
                    for (int j = 0; j < toCopy; j++)
                    {
                        //data.ns_part[j] = data.n_part[j] - (float)search[first- backLook].n[j] / div2;
                        data.ns_part[j] = search[first].s[ordering[j]] - search[first - backLook].s[ordering[j]];
                    }

                    data.fraction = (float) startSumN / estNodes;
                    data.scale = startSumN > 0 ? MathF.Log(startSumN) : 0.0f;
                    data.target = startIdx == finalIdx ? 1.0f : 0.0f;
                    data.ratio = (float) (startMaxN - secondBest) / (estNodes - startSumN);
                    if (double.IsPositiveInfinity(data.ratio))
                    {
                        Console.Out.WriteLine("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9}", startMaxN, secondBest, estNodes, startSumN, estTime, estBasisTime, zerothSumN, second, first, zeroth);
                        throw new ArgumentException("Infinity??");
                    }
                    data.dqtarget = 0.005f * (1.0f - data.fraction) + (startIdx == finalIdx
                        ? 0.0f
                        : search[second].s[startIdx] - search[second].s[finalIdx] - 0.01f);
                    if (startIdx != finalIdx)
                    {
                        //Console.Out.WriteLine("{0}", search[second].s[startIdx] - search[second].s[finalIdx]);
                    }
                    //Console.Out.WriteLine("{0} {1} {2} {3} {4}", data.dqtarget, data.target, data.fraction, startIdx, finalIdx);
                    pending.Add(data);
                    if (pending.Count > mixSize)
                    {
                        DrainOne(pending, rnd, gzs);
                    }
                }
            }

            while (pending.Count > 0)
            {
                DrainOne(pending, rnd, gzs);
            }
            Console.Out.WriteLine("{0} {1}", sLoss / falsePositives, error / (falsePositives + falseNegatives));
            Console.Out.WriteLine("{0} {1} {2} {3}", truePositives, falsePositives, trueNegatives, falseNegatives);
            Console.Out.WriteLine("{0} {1}", (float)truePositives / (truePositives + falseNegatives), (float)truePositives / (truePositives + falsePositives));
        }

        private static void DrainOne(List<TD> pending, Random rnd, GZipStream gzs)
        {
            int idx = rnd.Next(pending.Count);
            TD tmp = pending[idx];
            pending[idx] = pending[pending.Count - 1];
            pending.RemoveAt(pending.Count - 1);
            using var memory = new MemoryStream(100);
            using (var writer = new BinaryWriter(memory,Encoding.Default, true))
            {
                tmp.Write(writer);
            }

            memory.Seek(0, SeekOrigin.Begin);
            memory.CopyTo(gzs);
        }

        private static void Process(IEnumerable<List<Counter>> searches)
        {
            int[] finalPosCount = new int[256];
            int[] curPosCount = new int[256];
            float[] fractions = new float[30];
            float[] falsePositives = new float[30];
            int total = 0;
            foreach (var search in searches)
            {
                total++;
                int finalMaxN = search.Last().n.Max();
                int finalSumN = search.Last().n.Sum();
                int finalIdx = search.Last().n.ToList().IndexOf(finalMaxN);
                if (finalIdx > 18)
                {
                    //Console.WriteLine("{0} {1} {2} {3} {4}", finalIdx, search.Last().p[finalIdx], search.Last().s[finalIdx], search.Last().s[finalIdx-1], search.Last().s[0]);
                }
                finalPosCount[finalIdx]++;
                bool[] aborted = new bool[30];
                int initialSumN = search.First().n.Sum();
                foreach (var counter in search)
                {
                    if (aborted[0]) break;
                    int curSumN = counter.n.Sum();
                    int curMaxN = counter.n.Max();
                    int remaining = finalSumN - curSumN;
                    bool seenMax = false;
                    int secondBest = 0;
                    for (int i = 0; i < 256; i++)
                    {
                        if (counter.n[i] == curMaxN)
                        {
                            if (seenMax)
                            {
                                secondBest = curMaxN;
                            }
                            seenMax = true;
                        }
                        else if (counter.n[i] > secondBest)
                        {
                            secondBest = counter.n[i];
                        }
                    }
                    int curIdx2 = counter.n.ToList().IndexOf(curMaxN);
                    curPosCount[curIdx2]++;
                    for (int i = 0; i < 30; i++)
                    {
                        if (aborted[i]) break;
                        float spf = 1.0f + i * 0.1f;
                        float threshold = remaining / spf;
                        if (secondBest + threshold < curMaxN)
                        {
                            //Console.WriteLine("{0} {1} {2} {3} {4} {5}", spf, curMaxN, secondBest, curSumN, finalSumN, finalMaxN);
                            int curIdx = counter.n.ToList().IndexOf(curMaxN);
                            if (curIdx != finalIdx)
                            {
                                falsePositives[i] += 1.0f;
                            }

                            fractions[i] += (float)(curSumN - initialSumN) / (finalSumN - initialSumN);
                            aborted[i] = true;
                        }

                    }

                }
            }

            for (int i = 0; i < 30; i++)
            {
                fractions[i] /= total;
                falsePositives[i] /= total;
            }

            for (int i = 0; i < 256; i++)
            {
                Console.Write(finalPosCount[i]);
                Console.Write(" ");
            }
            Console.WriteLine();
            for (int i = 0; i < 256; i++)
            {
                Console.Write(curPosCount[i]);
                Console.Write(" ");
            }
            Console.WriteLine();
            for (int i = 0; i < 30; i++)
            {
                Console.Write(fractions[i]);
                Console.Write(" ");
            }
            Console.WriteLine();
            for (int i = 0; i < 30; i++)
            {
                Console.Write(falsePositives[i]);
                Console.Write(" ");
            }
            Console.WriteLine();
        }
    }
}
