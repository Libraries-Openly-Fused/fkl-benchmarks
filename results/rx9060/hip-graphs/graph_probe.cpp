// Where does the time of the HIP graph of the horizontal fusion benchmark go? The same per-crop kernel (crop + saturate + mul + sub + div + write, 60x120 uchar -> float of a
// 3840x2160 image) launched five ways, for BATCH crops:
//   plain        BATCH launches in the stream (the "iterated batch" of the benchmark)
//   children     the graph of the benchmark: BATCH child graph nodes, each captured from one launch (hipStreamBeginCapture, global mode), no dependencies between them
//   children_seq the same with every child depending on the previous one
//   flat         ONE stream capture of the BATCH launches: a graph of BATCH kernel nodes in a chain (serial)
//   flat_parallel ONE stream capture with a fork and a join over one stream per crop (events): BATCH kernel nodes with no dependencies between them (they may run side by side)
// per variant: submit = host time of the launch call(s) ONLY (probe5_run1 and probe5_run2 took it from before the first hipEventRecord to after the second one, which includes the two event
// records: that interval is printed as "old definition of submit" on a second line per batch), total = from before the first event record to the end of the stream synchronisation
// (launch, the two event records and the synchronisation), event = hipEvent time around the submission, in the stream (what the benchmark reports); medians of N repetitions after a
// warm-up of 1.5 s. The first line printed is the cost of two hipEventRecord calls on an idle stream. The graph building (capture, instantiate) is not timed. A strategy whose output
// differs from the plain launches stops the program with exit code 1 before it is timed.
// usage: graph_probe [repetitions [batch]]   (default 100 repetitions, at least 1, and the batches 11, 51, 101, 201, 301, 601 and 1191; with a batch argument only 101, 301 or 1191 are accepted)
#include <hip/hip_runtime.h>

#include <fused_kernel/algorithms/algorithms.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/data/ptr_utils.h>
#include <fused_kernel/fused_kernel.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#define HC(x) do { const hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s failed at line %d: %s\n", #x, __LINE__, hipGetErrorString(e_)); std::exit(1); } } while (0)

using Clock = std::chrono::steady_clock;
static double msSince(const Clock::time_point a, const Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
static double med(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

struct Result { double submit, total, event, submitWide; };

static Result timeIt(const std::function<void()>& submit, hipStream_t hs, const int reps, const double warmupSeconds) {
    hipEvent_t e0, e1;
    HC(hipEventCreate(&e0));
    HC(hipEventCreate(&e1));
    const auto t0 = Clock::now();
    while (msSince(t0, Clock::now()) < warmupSeconds * 1000.0) { submit(); HC(hipStreamSynchronize(hs)); }
    std::vector<double> sub, tot, ev, wide;
    for (int i = 0; i < reps; ++i) {
        const auto a = Clock::now();
        HC(hipEventRecord(e0, hs));
        const auto s0 = Clock::now();
        submit();
        const auto s1 = Clock::now();
        HC(hipEventRecord(e1, hs));
        const auto b = Clock::now();
        HC(hipStreamSynchronize(hs));
        const auto c = Clock::now();
        float f = 0;
        HC(hipEventElapsedTime(&f, e0, e1));
        sub.push_back(msSince(s0, s1));
        wide.push_back(msSince(a, b));
        tot.push_back(msSince(a, c));
        ev.push_back(f);
    }
    HC(hipEventDestroy(e0));
    HC(hipEventDestroy(e1));
    return {med(sub), med(tot), med(ev), med(wide)};
}

static bool wanted(const char* name) {
    const char* v = std::getenv("PROBE_VARIANT");
    return v == nullptr || std::string(v) == name;
}
static double warmupSeconds() {
    const char* v = std::getenv("PROBE_WARMUP");
    return v ? std::atof(v) : 1.5;
}

// the number of nodes of a graph and how many of them are kernel nodes / child graph nodes
static void describe(const char* name, hipGraph_t g) {
    size_t n = 0;
    HC(hipGraphGetNodes(g, nullptr, &n));
    std::vector<hipGraphNode_t> nodes(n);
    HC(hipGraphGetNodes(g, nodes.data(), &n));
    size_t kernels = 0, children = 0, others = 0;
    for (const auto node : nodes) {
        hipGraphNodeType t;
        HC(hipGraphNodeGetType(node, &t));
        if (t == hipGraphNodeTypeKernel) ++kernels; else if (t == hipGraphNodeTypeGraph) ++children; else ++others;
    }
    std::printf("   graph %-12s: %zu nodes (kernel %zu, child graph %zu, other %zu)\n", name, n, kernels, children, others);
}

template <size_t BATCH>
void probe(fk::Stream& stream, const int reps) {
    using InputType = uchar;
    using OutputType = float;
    const fk::Size cropSize(60, 120);
    fk::Ptr2D<InputType> d_input(3840, 2160);
    fk::setTo(static_cast<InputType>(10), d_input, stream);
    std::array<fk::Ptr2D<OutputType>, BATCH> d_output;
    std::array<fk::Ptr2D<InputType>, BATCH> crops;
    for (int i = 0; i < static_cast<int>(BATCH); i++) {
        crops[i] = d_input.crop(fk::Point{i, i, 0}, fk::PtrDims<fk::ND::_2D>{static_cast<uint>(cropSize.width), static_cast<uint>(cropSize.height), static_cast<uint>(d_input.dims().pitch)});
        d_output[i].Alloc(cropSize);
    }
    const auto read_array = fk::PerThreadRead<fk::ND::_2D, InputType>::build_batch(crops);
    const auto saturate = fk::SaturateCast<InputType, OutputType>::build();
    const auto mul = fk::Mul<OutputType>::build(1.0f);
    const auto sub = fk::Sub<OutputType>::build(1.f);
    const auto div = fk::Div<OutputType>::build(3.2f);
    const auto write_array = fk::PerThreadWrite<fk::ND::_2D, OutputType>::build_batch(d_output);
    stream.sync();
    const hipStream_t hs = stream.getHIPStream();

    const auto launchOne = [&](fk::Stream& s, const int i) { fk::executeOperations<fk::TransformDPP<>>(s, read_array[i], saturate, mul, sub, div, write_array[i]); };

    // hash of the bits of all the outputs; each strategy is run once on cleared outputs and must give the hash of the plain launches
    const auto clearOutputs = [&] { for (int i = 0; i < static_cast<int>(BATCH); ++i) fk::setTo(static_cast<OutputType>(0), d_output[i], stream); stream.sync(); };
    const auto hashOutputs = [&] {
        uint64_t h = 1469598103934665603ull;
        for (int i = 0; i < static_cast<int>(BATCH); ++i) {
            d_output[i].download(stream);
            stream.sync();
            for (int y = 0; y < cropSize.height; ++y)
                for (int x = 0; x < cropSize.width; ++x) {
                    uint32_t bits;
                    const float v = d_output[i].at(fk::Point{x, y, 0});
                    std::memcpy(&bits, &v, 4);
                    h = (h ^ bits) * 1099511628211ull;
                }
        }
        return h;
    };
    clearOutputs();
    for (int i = 0; i < static_cast<int>(BATCH); ++i) launchOne(stream, i);
    stream.sync();
    const uint64_t plainHash = hashOutputs();
    const auto checkStrategy = [&](const char* name, const std::function<void()>& run) {
        clearOutputs();
        run();
        HC(hipStreamSynchronize(stream.getHIPStream()));
        const bool same = hashOutputs() == plainHash;
        std::printf("   output of %-12s identical to the plain launches: %s\n", name, same ? "yes" : "NO");
        if (!same) {
            std::fprintf(stderr, "graph_probe: the output of %s differs from the plain launches: it is not timed\n", name);
            std::exit(1);
        }
    };

    // plain launches
    Result plain{0, 0, 0}, children{0, 0, 0}, childrenSeq{0, 0, 0}, flat{0, 0, 0}, flatPar{0, 0, 0};
    if (wanted("plain")) plain = timeIt([&] { for (int i = 0; i < static_cast<int>(BATCH); ++i) launchOne(stream, i); }, hs, reps, warmupSeconds());

    // children (the graph of the benchmark) and children_seq
    const auto buildChildren = [&](const bool sequential) {
        hipGraph_t mainGraph;
        HC(hipGraphCreate(&mainGraph, 0));
        std::vector<hipGraphNode_t> nodes(BATCH);
        for (int i = 0; i < static_cast<int>(BATCH); i++) {
            fk::Stream nodeStream;
            hipGraph_t subGraph;
            HC(hipStreamBeginCapture(nodeStream, hipStreamCaptureModeGlobal));
            launchOne(nodeStream, i);
            HC(hipStreamEndCapture(nodeStream, &subGraph));
            if (sequential && i > 0) {
                HC(hipGraphAddChildGraphNode(&nodes[i], mainGraph, &nodes[i - 1], 1, subGraph));
            } else {
                HC(hipGraphAddChildGraphNode(&nodes[i], mainGraph, nullptr, 0, subGraph));
            }
            HC(hipGraphDestroy(subGraph));
        }
        if (!sequential) describe("children", mainGraph); else describe("children_seq", mainGraph);
        hipGraphExec_t exec;
        HC(hipGraphInstantiate(&exec, mainGraph, NULL, NULL, 0));
        HC(hipGraphDestroy(mainGraph));
        return exec;
    };
    if (wanted("children")) {
        hipGraphExec_t childrenExec = buildChildren(false);
        checkStrategy("children", [&] { HC(hipGraphLaunch(childrenExec, hs)); });
        children = timeIt([&] { HC(hipGraphLaunch(childrenExec, hs)); }, hs, reps, warmupSeconds());
        HC(hipGraphExecDestroy(childrenExec));
    }
    if (wanted("children_seq")) {
        hipGraphExec_t seqExec = buildChildren(true);
        checkStrategy("children_seq", [&] { HC(hipGraphLaunch(seqExec, hs)); });
        childrenSeq = timeIt([&] { HC(hipGraphLaunch(seqExec, hs)); }, hs, reps, warmupSeconds());
        HC(hipGraphExecDestroy(seqExec));
    }

    // flat: one capture of all the launches
    if (wanted("flat")) {
    hipGraphExec_t flatExec;
    {
        fk::Stream captureStream;
        hipGraph_t g;
        HC(hipStreamBeginCapture(captureStream, hipStreamCaptureModeGlobal));
        for (int i = 0; i < static_cast<int>(BATCH); ++i) launchOne(captureStream, i);
        HC(hipStreamEndCapture(captureStream, &g));
        describe("flat", g);
        HC(hipGraphInstantiate(&flatExec, g, NULL, NULL, 0));
        HC(hipGraphDestroy(g));
    }
    checkStrategy("flat", [&] { HC(hipGraphLaunch(flatExec, hs)); });
    flat = timeIt([&] { HC(hipGraphLaunch(flatExec, hs)); }, hs, reps, warmupSeconds());
    HC(hipGraphExecDestroy(flatExec));
    }

    // flat parallel: ONE capture with a fork and a join over one stream per crop; the crops have no dependencies between them
    if (wanted("flat_parallel")) {
        hipGraphExec_t parExec;
        {
            fk::Stream origin;
            std::vector<std::unique_ptr<fk::Stream>> workers;
            for (int i = 0; i < static_cast<int>(BATCH); ++i) workers.push_back(std::make_unique<fk::Stream>());
            hipEvent_t fork;
            HC(hipEventCreateWithFlags(&fork, hipEventDisableTiming));
            std::vector<hipEvent_t> joins(BATCH);
            for (auto& e : joins) HC(hipEventCreateWithFlags(&e, hipEventDisableTiming));
            hipGraph_t g;
            HC(hipStreamBeginCapture(origin, hipStreamCaptureModeGlobal));
            HC(hipEventRecord(fork, origin));
            for (int i = 0; i < static_cast<int>(BATCH); ++i) {
                HC(hipStreamWaitEvent(*workers[i], fork, 0));
                launchOne(*workers[i], i);
                HC(hipEventRecord(joins[i], *workers[i]));
            }
            for (int i = 0; i < static_cast<int>(BATCH); ++i) HC(hipStreamWaitEvent(origin, joins[i], 0));
            HC(hipStreamEndCapture(origin, &g));
            describe("flat_parallel", g);
            HC(hipGraphInstantiate(&parExec, g, NULL, NULL, 0));
            HC(hipGraphDestroy(g));
            HC(hipEventDestroy(fork));
            for (auto& e : joins) HC(hipEventDestroy(e));
        }
        checkStrategy("flat_parallel", [&] { HC(hipGraphLaunch(parExec, hs)); });
        flatPar = timeIt([&] { HC(hipGraphLaunch(parExec, hs)); }, hs, reps, warmupSeconds());
        HC(hipGraphExecDestroy(parExec));
    }

    std::printf("batch %4zu | plain: submit %.4f total %.4f event %.4f | children: submit %.4f total %.4f event %.4f | children_seq: submit %.4f total %.4f event %.4f | flat: submit %.4f total %.4f event %.4f | flat_parallel: submit %.4f total %.4f event %.4f   (ms)\n",
                BATCH, plain.submit, plain.total, plain.event, children.submit, children.total, children.event, childrenSeq.submit, childrenSeq.total, childrenSeq.event, flat.submit, flat.total, flat.event,
                flatPar.submit, flatPar.total, flatPar.event);
    std::printf("batch %4zu | old definition of submit (two hipEventRecord calls included): plain %.4f children %.4f children_seq %.4f flat %.4f flat_parallel %.4f\n",
                BATCH, plain.submitWide, children.submitWide, childrenSeq.submitWide, flat.submitWide, flatPar.submitWide);
    std::fflush(stdout);
}

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 100;
    if (reps < 1) {
        std::fprintf(stderr, "graph_probe: the repetitions must be a positive integer (got '%s')\n", argv[1]);
        return 1;
    }
    const int batchArg = argc > 2 ? std::atoi(argv[2]) : 0;
    if (argc > 2 && batchArg != 101 && batchArg != 301 && batchArg != 1191) {
        std::fprintf(stderr, "graph_probe: the batch must be 101, 301 or 1191 (got '%s'); without the argument the probe runs 11, 51, 101, 201, 301, 601 and 1191\n", argv[2]);
        return 1;
    }
    fk::Stream stream;
    {
        const hipStream_t hs0 = stream.getHIPStream();
        hipEvent_t x0, x1;
        HC(hipEventCreate(&x0));
        HC(hipEventCreate(&x1));
        const auto t0 = Clock::now();
        while (msSince(t0, Clock::now()) < 1500.0) { HC(hipEventRecord(x0, hs0)); HC(hipEventRecord(x1, hs0)); HC(hipStreamSynchronize(hs0)); }
        std::vector<double> pair;
        for (int i = 0; i < 1000; ++i) {
            HC(hipStreamSynchronize(hs0));
            const auto a = Clock::now();
            HC(hipEventRecord(x0, hs0));
            HC(hipEventRecord(x1, hs0));
            const auto b = Clock::now();
            pair.push_back(msSince(a, b));
        }
        HC(hipStreamSynchronize(hs0));
        std::sort(pair.begin(), pair.end());
        std::printf("two hipEventRecord calls on an idle stream (host, ms): median %.5f p10 %.5f p90 %.5f min %.5f max %.5f over %zu pairs\n", pair[pair.size() / 2], pair[pair.size() / 10], pair[pair.size() * 9 / 10], pair.front(), pair.back(), pair.size());
        HC(hipEventDestroy(x0));
        HC(hipEventDestroy(x1));
    }
    if (argc > 2) {
        if (batchArg == 101) probe<101>(stream, reps);
        else if (batchArg == 301) probe<301>(stream, reps);
        else probe<1191>(stream, reps);
        return 0;
    }
    probe<11>(stream, reps);
    probe<51>(stream, reps);
    probe<101>(stream, reps);
    probe<201>(stream, reps);
    probe<301>(stream, reps);
    probe<601>(stream, reps);
    probe<1191>(stream, reps);
    return 0;
}
